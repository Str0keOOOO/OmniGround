"""Start the OmniGround HTTP server."""

from __future__ import annotations

import argparse
import logging

from ..server import DEFAULT_MAX_REQUEST_BYTES, create_app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start the OmniGround unified VLM service.")
    parser.add_argument("--config", default=None, help="Path to models.yaml")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument("--model-id", default=None, help="Default model used by /ready when no query is supplied")
    parser.add_argument("--max-request-bytes", type=int, default=DEFAULT_MAX_REQUEST_BYTES)
    parser.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    app = create_app(
        config_path=args.config,
        default_model=args.model_id,
        max_request_bytes=args.max_request_bytes,
    )
    logging.getLogger(__name__).info(
        "Starting OmniGround on %s:%s with default model %s (models remain lazy-loaded)",
        args.host,
        args.port,
        app.state.default_model,
    )
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
