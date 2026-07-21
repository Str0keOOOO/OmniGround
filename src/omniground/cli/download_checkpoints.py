"""Download configured checkpoints only when explicitly requested."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a checkpoint for one configured local model.")
    parser.add_argument("model_id", nargs="?", default="molmo2-er")
    parser.add_argument("--config", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--token", default=None, help="Optional Hugging Face token; never stored in config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    try:
        model = config.models[args.model_id]
    except KeyError as exc:
        raise SystemExit(f"Unknown model_id '{args.model_id}'") from exc
    if model.backend != "molmo2":
        raise SystemExit(f"Model '{args.model_id}' has no local checkpoint downloader (backend: {model.backend})")
    repo_id = model.option("hf_repo")
    if not repo_id:
        raise SystemExit(f"Model '{args.model_id}' does not define hf_repo in its configuration")
    target = Path(args.output_dir) if args.output_dir else config.resolve_path(model.checkpoint)
    if target is None:
        raise SystemExit(f"Model '{args.model_id}' does not define checkpoint in its configuration")
    if target.is_dir() and any(target.iterdir()):
        print(f"Checkpoint already exists at {target}; skipping download.")
        return
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit("Checkpoint download requires `pip install -e .[download]`.") from exc
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=repo_id, local_dir=str(target), token=args.token)
    print(f"Downloaded {repo_id} to {target}")


if __name__ == "__main__":
    main()
