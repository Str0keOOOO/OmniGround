"""Typed errors that can be rendered safely by the HTTP layer."""

from __future__ import annotations


class OmniGroundError(Exception):
    """Base error with a client-safe code and HTTP status."""

    code = "OMNIGROUND_ERROR"
    status_code = 500

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class InputValidationError(OmniGroundError):
    code = "INVALID_REQUEST"
    status_code = 400


class UnsupportedImageError(OmniGroundError):
    code = "UNSUPPORTED_IMAGE_TYPE"
    status_code = 415


class RequestTooLargeError(OmniGroundError):
    code = "PAYLOAD_TOO_LARGE"
    status_code = 413


class UnknownModelError(OmniGroundError):
    code = "UNKNOWN_MODEL"
    status_code = 404


class BackendUnavailableError(OmniGroundError):
    code = "MODEL_NOT_READY"
    status_code = 503


class BackendInferenceError(OmniGroundError):
    code = "BACKEND_INFERENCE_FAILED"
    status_code = 502


class ModelOutputParseError(OmniGroundError):
    code = "INVALID_MODEL_OUTPUT"
    status_code = 502


class ModelOutputValidationError(ModelOutputParseError):
    pass


class PointOnlyOutputError(ModelOutputParseError):
    code = "POINT_OUTPUT_NOT_SUPPORTED"
