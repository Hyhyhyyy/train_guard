"""Run manifests and comparisons."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

from .. import __version__
from ..adapters.huggingface import (
    get_framework_adapter,
)
from ..core.io_util import (
    utc_now_iso,
    write_json,
)
from ..core.privacy import redact_value
from .lifecycle import (
    PHASE_NONE,
    lifecycle_path,
    summarize_lifecycle,
)


def run_manifest(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Create run manifest + experiment fingerprint (no raw training content)."""
    output_dir = Path(cfg.get("output_dir") or ".")
    framework = str(cfg.get("framework") or "generic")
    manifest_out = cfg.get("manifest_out")
    input_errors: List[str] = []
    if not output_dir.exists():
        input_errors.append("output directory does not exist")
    elif not output_dir.is_dir():
        input_errors.append("output path is not a directory")
    if input_errors:
        manifest: Dict[str, Any] = {
            "tool": "train_guard",
            "command": "manifest",
            "version": __version__,
            "timestamp": utc_now_iso(),
            "framework": framework,
            "output_dir_name": output_dir.name,
            "overall_status": "FAIL",
            "input_status": "invalid",
            "reasons": input_errors,
            "checkpoint_names": [],
            "lifecycle": {},
            "experiment_fingerprint": None,
            "note": "Manifest was not created from an invalid run directory.",
        }
        if manifest_out:
            write_json(Path(manifest_out), redact_value(manifest), overwrite=True)
            manifest["manifest_written"] = Path(manifest_out).name
        return redact_value(manifest)

    life = summarize_lifecycle(lifecycle_path(output_dir))
    parts = [
        framework,
        output_dir.name,
        str(cfg.get("expected_steps") or ""),
        str(cfg.get("seed") or ""),
        str(life.get("phase") or ""),
        str(life.get("event_count") or 0),
    ]
    # Include sorted names of checkpoint dirs and adapter presence, not file contents
    adapter = get_framework_adapter(framework)
    ckpt_names = ",".join(p.name for p in adapter.list_checkpoints(output_dir))
    parts.append(ckpt_names)
    state = adapter.locate_trainer_state(output_dir)
    if state and state.is_file():
        try:
            data = json.loads(state.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("trainer_state root is not an object")
            state_step = data.get("global_step")
            if state_step is not None:
                state_step = int(state_step)
            parts.append(str(state_step))
            parts.append(str(data.get("best_metric")))
        except (OSError, json.JSONDecodeError, ValueError):
            parts.append("state_unreadable")
            input_errors.append("trainer_state.json is invalid or unreadable")
    fingerprint = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    manifest = {
        "tool": "train_guard",
        "command": "manifest",
        "version": __version__,
        "timestamp": utc_now_iso(),
        "framework": framework,
        "output_dir_name": output_dir.name,
        "overall_status": "FAIL" if input_errors else "PASS",
        "input_status": "invalid" if input_errors else "valid",
        "reasons": input_errors,
        "checkpoint_names": ckpt_names.split(",") if ckpt_names else [],
        "lifecycle": {
            "phase": life.get("phase"),
            "event_count": life.get("event_count"),
            "global_step": life.get("global_step"),
            "has_finish": life.get("has_finish"),
            "has_abort": life.get("has_abort"),
        },
        "experiment_fingerprint": fingerprint,
        "note": "Fingerprint excludes raw samples, logs text, and absolute paths.",
    }
    out = Path(manifest_out or (output_dir / "train_guard_manifest.json"))
    write_json(out, redact_value(manifest), overwrite=True)
    manifest["manifest_written"] = out.name
    return redact_value(manifest)


def run_run_compare(
    left_dir: Path,
    right_dir: Path,
    framework: str = "huggingface",
) -> Dict[str, Any]:
    """Compare two run output dirs by fingerprint/steps (no raw content)."""

    def snap(d: Path) -> Dict[str, Any]:
        errors: List[str] = []
        if not d.exists():
            return {
                "name": d.name,
                "input_status": "invalid",
                "status": "FAIL",
                "reasons": ["directory does not exist"],
                "global_step": None,
                "checkpoints": 0,
                "weights": 0,
                "has_adapter_config": False,
                "lifecycle_phase": PHASE_NONE,
                "lifecycle_events": 0,
            }
        if not d.is_dir():
            return {
                "name": d.name,
                "input_status": "invalid",
                "status": "FAIL",
                "reasons": ["path is not a directory"],
                "global_step": None,
                "checkpoints": 0,
                "weights": 0,
                "has_adapter_config": False,
                "lifecycle_phase": PHASE_NONE,
                "lifecycle_events": 0,
            }
        fw = get_framework_adapter(framework)
        state = fw.locate_trainer_state(d)
        step = None
        if state and state.is_file():
            try:
                data = json.loads(state.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    raise ValueError("root is not an object")
                raw_step = data.get("global_step")
                step = int(raw_step) if raw_step is not None else None
            except (OSError, json.JSONDecodeError, ValueError):
                step = None
                errors.append("trainer_state.json is invalid or unreadable")
        arts = fw.find_adapter_artifacts(d)
        life = summarize_lifecycle(lifecycle_path(d))
        return {
            "name": d.name,
            "input_status": "invalid" if errors else "valid",
            "status": "FAIL" if errors else "PASS",
            "reasons": errors,
            "global_step": step,
            "checkpoints": len(arts.checkpoints),
            "weights": len([w for w in arts.weight_files if w.get("ok")]),
            "has_adapter_config": bool(arts.adapter_configs),
            "lifecycle_phase": life.get("phase"),
            "lifecycle_events": life.get("event_count"),
        }

    a = snap(left_dir)
    b = snap(right_dir)
    report = {
        "tool": "train_guard",
        "command": "run compare",
        "version": __version__,
        "timestamp": utc_now_iso(),
        "framework": framework,
        "left": a,
        "right": b,
        "step_delta": None
        if a["global_step"] is None or b["global_step"] is None
        else (b["global_step"] - a["global_step"]),
        "overall_status": "FAIL" if a["status"] == "FAIL" or b["status"] == "FAIL" else "PASS",
        "reasons": [
            *[f"left: {reason}" for reason in a.get("reasons", [])],
            *[f"right: {reason}" for reason in b.get("reasons", [])],
        ],
        "disclaimer": "Compares metadata only.",
    }
    return redact_value(report)
