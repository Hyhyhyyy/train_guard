"""Generate versioned benchmark JSONL files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .scenarios import build_dataset


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, allow_nan=False, sort_keys=True) + "\n")


def generate(output_dir: Path, seed: int = 20260724) -> tuple[Path, Path]:
    telemetry, expected = build_dataset(seed)
    telemetry_path = output_dir / "telemetry.jsonl"
    expected_path = output_dir / "expected_alerts.jsonl"
    write_jsonl(telemetry_path, telemetry)
    write_jsonl(expected_path, expected)
    return telemetry_path, expected_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark-output"))
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--json", action="store_true", help="print a machine-readable summary")
    args = parser.parse_args(argv)
    telemetry_path, expected_path = generate(args.output_dir, args.seed)
    result = {
        "seed": args.seed,
        "telemetry": str(telemetry_path),
        "expected_alerts": str(expected_path),
    }
    print(json.dumps(result, sort_keys=True) if args.json else result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
