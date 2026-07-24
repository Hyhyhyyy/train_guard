#!/usr/bin/env python3
"""Offline text-generation smoke test for a local Transformers model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one offline local-model generation")
    parser.add_argument(
        "--model-path", required=True, help="Relative path to a local model directory"
    )
    parser.add_argument("--prompt", default="List three colors.")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args(argv)

    model_path = Path(args.model_path)
    if not model_path.is_dir() or not (model_path / "config.json").is_file():
        print("model directory is missing or incomplete", file=sys.stderr)
        return 2
    if args.max_new_tokens <= 0:
        print("--max-new-tokens must be positive", file=sys.stderr)
        return 2

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("required local framework dependencies are unavailable", file=sys.stderr)
        return 2

    try:
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(str(model_path), local_files_only=True)
        inputs = tokenizer(args.prompt, return_tensors="pt")
        generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
        print(tokenizer.decode(generated[0], skip_special_tokens=True))
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"offline inference failed: {type(exc).__name__}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
