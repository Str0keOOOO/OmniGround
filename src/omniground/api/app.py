"""FastAPI application exposing the model-independent OmniGround contract."""

from __future__ import annotations

import io
import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from starlette.exceptions import HTTPException as StarletteHTTPException

from ..backends.base import GenerationRequest
from ..backends.registry import ModelRegistry
from ..core.config import load_config
from ..core.errors import (
    BackendInferenceError,
    BackendUnavailableError,
    InputValidationError,
    OmniGroundError,
    RequestTooLargeError,
    UnsupportedImageError,
)

_LOG = logging.getLogger(__name__)
_SUPPORTED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png"}
_SUPPORTED_IMAGE_FORMATS = {"JPEG", "PNG"}
DEFAULT_MAX_REQUEST_BYTES = 16 * 1024 * 1024


def _error_response(error: OmniGroundError) -> JSONResponse:
    return JSONResponse(
        status_code=error.status_code,
        content={"error": {"code": error.code, "message": error.message}},
    )


def _decode_image(content: bytes) -> Image.Image:
    if not content:
        raise InputValidationError("image must not be empty")
    try:
        with Image.open(io.BytesIO(content)) as source:
            if source.format not in _SUPPORTED_IMAGE_FORMATS:
                raise UnsupportedImageError("image must be a valid PNG or JPEG")
            return source.convert("RGB")
    except UnidentifiedImageError as exc:
        raise UnsupportedImageError("image must be a valid PNG or JPEG") from exc


def create_app(
    *,
    config_path: str | None = None,
    default_model: str | None = None,
    max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES,
) -> FastAPI:
    """Build an app and eagerly load the selected local model at startup."""

    if max_request_bytes <= 0:
        raise ValueError("max_request_bytes must be positive")
    config = load_config(config_path)
    selected_default = default_model or config.default_model
    registry = ModelRegistry(config)
    registry.get_config(selected_default)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            selected_config = registry.get_config(selected_default)
            if selected_config.mode == "local":
                started_at = time.perf_counter()
                _LOG.info("Eagerly loading local model %s during startup", selected_default)
                registry.get_backend(selected_default)
                _LOG.info(
                    "Loaded local model %s during startup in %.1f ms",
                    selected_default,
                    (time.perf_counter() - started_at) * 1000,
                )
            yield
        finally:
            registry.unload_all()

    app = FastAPI(title="OmniGround", version="0.1.0", lifespan=lifespan)
    app.state.registry = registry
    app.state.default_model = selected_default
    app.state.max_request_bytes = max_request_bytes

    @app.exception_handler(OmniGroundError)
    async def omniground_error_handler(_: Request, exc: OmniGroundError) -> JSONResponse:
        return _error_response(exc)

    @app.exception_handler(RequestValidationError)
    async def request_validation_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
        issues = "; ".join(
            f"{'.'.join(str(part) for part in issue['loc'])}: {issue['msg']}" for issue in exc.errors()
        )
        return _error_response(InputValidationError(f"invalid multipart request: {issues}"))

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        return _error_response(InputValidationError(str(exc.detail)))

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    async def ready() -> dict[str, object]:
        is_ready, detail = registry.probe(selected_default)
        if not is_ready:
            raise BackendUnavailableError(detail)
        return {"status": "ready", "detail": detail}

    @app.get("/v1/models")
    async def models() -> dict[str, object]:
        return {"object": "list", "data": registry.describe_models()}

    async def generate(
        request: Request,
        image: UploadFile = File(...),
        prompt: str = Form(...),
        temperature: float | None = Form(None),
    ) -> JSONResponse:
        request_id = uuid.uuid4().hex
        started_at = time.perf_counter()
        image_object: Image.Image | None = None
        backend = None
        try:
            content_length = request.headers.get("content-length")
            if content_length:
                try:
                    if int(content_length) > max_request_bytes:
                        raise RequestTooLargeError(f"request exceeds the {max_request_bytes} byte size limit")
                except ValueError as exc:
                    raise InputValidationError("invalid Content-Length header") from exc
            if image.content_type and image.content_type.lower() not in _SUPPORTED_CONTENT_TYPES:
                raise UnsupportedImageError("image content type must be image/png or image/jpeg")
            if not prompt.strip():
                raise InputValidationError("prompt must not be empty")
            if temperature is not None and temperature < 0:
                raise InputValidationError("temperature must be greater than or equal to zero")

            image_content = await image.read(max_request_bytes + 1)
            if len(image_content) > max_request_bytes:
                raise RequestTooLargeError(f"image exceeds the {max_request_bytes} byte size limit")
            image_object = _decode_image(image_content)

            backend, first_load = registry.get_backend(selected_default)
            inference_started_at = time.perf_counter()
            result = backend.generate(
                GenerationRequest(
                    image=image_object,
                    prompt=prompt,
                    model_id=selected_default,
                    temperature=temperature,
                )
            )
            inference_ms = (time.perf_counter() - inference_started_at) * 1000
            elapsed_ms = (time.perf_counter() - started_at) * 1000
            _LOG.info(
                "generate success request_id=%s model_id=%s backend=%s first_load=%s elapsed_ms=%.1f "
                "inference_ms=%.1f prompt_length=%s parser_and_validation=success",
                request_id,
                selected_default,
                registry.get_config(selected_default).backend,
                first_load,
                elapsed_ms,
                inference_ms,
                len(prompt),
            )
            response = JSONResponse(content=result.model_dump(mode="json"))
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Backend-Inference-Ms"] = f"{inference_ms:.3f}"
            for phase_name, phase_seconds in backend.last_timing.items():
                response.headers[f"X-Backend-Timing-{phase_name.replace('_', '-')}"] = f"{phase_seconds * 1000:.3f}"
            return response
        except OmniGroundError as exc:
            elapsed_ms = (time.perf_counter() - started_at) * 1000
            _LOG.warning(
                "generate failed request_id=%s model_id=%s error_code=%s error_message=%s "
                "elapsed_ms=%.1f prompt_length=%s",
                request_id,
                selected_default,
                exc.code,
                exc.message,
                elapsed_ms,
                len(prompt),
            )
            if backend is not None and backend.last_raw_text is not None:
                return JSONResponse(
                    status_code=exc.status_code,
                    content={
                        "error": {"code": exc.code, "message": exc.message},
                        "raw_model_output": backend.last_raw_text,
                    },
                )
            raise
        except Exception as exc:
            _LOG.exception("generate failed request_id=%s model_id=%s", request_id, selected_default)
            raise BackendInferenceError("Selected backend failed unexpectedly") from exc
        finally:
            if image_object is not None:
                image_object.close()
            await image.close()

    app.post("/generate", response_model=None)(generate)
    app.post("/v1/generate", response_model=None)(generate)
    return app
