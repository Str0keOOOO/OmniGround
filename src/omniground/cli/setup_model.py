"""Initialize optional model submodules without installing heavyweight extras."""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import subprocess

from ..config import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize OmniGround's optional Git submodules.")
    parser.add_argument("--skip-submodules", action="store_true")
    args = parser.parse_args()

    if not args.skip_submodules and (PROJECT_ROOT / ".git").exists() and shutil.which("git"):
        subprocess.run(["git", "submodule", "update", "--init", "--recursive"], cwd=PROJECT_ROOT, check=True)
        print("Git submodules initialized.")
    elif not args.skip_submodules:
        print("No Git checkout detected; skipped submodule initialization.")

    missing = [
        package
        for package in ("fastapi", "pydantic", "PIL", "yaml", "multipart", "uvicorn")
        if importlib.util.find_spec(package) is None
    ]
    if missing:
        raise SystemExit("Missing base dependencies: " + ", ".join(missing) + ". Run `pixi install` and retry.")
    print("OmniGround base dependencies are available. Optional model dependencies were not installed.")
    print("Next: set OPENAI_API_KEY and run `pixi run server -- --model-id qwen3.7-plus`, or configure a local model.")


if __name__ == "__main__":
    main()
