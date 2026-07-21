"""Run a self-contained server/request/schema demo and always stop the server."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from ..config import PROJECT_ROOT
from ..schemas import GroundingResult


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start OmniGround, submit one image, validate the JSON result, and stop.")
    parser.add_argument("--model-id", default="qwen3.7-plus")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument("--config", default=None)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import requests
    except ImportError as exc:  # pragma: no cover - declared core dependency
        raise SystemExit("Demo requires requests; run `pixi install` first.") from exc

    command = [
        sys.executable,
        "-m",
        "omniground.cli.run_server",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--model-id",
        args.model_id,
    ]
    if args.config:
        command.extend(["--config", args.config])
    process = subprocess.Popen(command, cwd=PROJECT_ROOT)
    base_url = f"http://{args.host}:{args.port}"
    deadline = time.monotonic() + args.timeout_seconds
    # A demo server is always local. Do not route its readiness/request traffic
    # through a workstation's HTTP(S) proxy.
    session = requests.Session()
    session.trust_env = False
    try:
        while time.monotonic() < deadline:
            try:
                response = session.get(base_url + "/ready", params={"model_id": args.model_id}, timeout=1)
                if response.status_code == 200:
                    break
            except requests.RequestException:
                pass
            time.sleep(0.2)
        else:
            raise SystemExit("Server did not become ready before the demo timeout")

        image_path = PROJECT_ROOT / "examples" / "demo_image.png"
        prompt = (PROJECT_ROOT / "examples" / "demo_prompt.txt").read_text(encoding="utf-8")
        with image_path.open("rb") as image_file:
            response = session.post(
                base_url + "/generate",
                files={"image": (image_path.name, image_file, "image/png")},
                data={"prompt": prompt, "model_id": args.model_id, "temperature": "0"},
                timeout=args.timeout_seconds,
            )
        response.raise_for_status()
        result = GroundingResult.model_validate(response.json())
        print(result.model_dump_json(indent=2))
    except (requests.RequestException, ValueError) as exc:
        raise SystemExit(f"Demo failed: {exc}") from exc
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


if __name__ == "__main__":
    main()
