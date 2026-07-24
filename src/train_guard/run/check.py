"""Training completion evidence checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .. import __version__
from ..adapters.huggingface import (
    FULL_MODEL_WEIGHT_NAMES,
    find_weight_files,
    get_framework_adapter,
    is_finite_number,
)
from ..core.events import CheckItem, overall_status
from ..core.io_util import (
    utc_now_iso,
)
from ..core.privacy import redact_value
from ..domain import json_safe
from .lifecycle import (
    PHASE_ABORTED,
    PHASE_FINISHED,
    PHASE_NONE,
    lifecycle_path,
    summarize_lifecycle,
)


def run_run_check(
    output_dir: Path,
    expected_steps: Optional[int] = None,
    framework: str = "huggingface",
    training_type: str = "auto",
) -> Dict[str, Any]:
    """Post-training completion check for PEFT or full-model artifacts."""
    items: List[CheckItem] = []
    reasons: List[str] = []
    adapter = get_framework_adapter(framework)

    if not output_dir.exists():
        items.append(CheckItem("output_dir", "FAIL", "Output directory does not exist", {}))
        return _pack_check(
            items, ["Output directory does not exist"], None, expected_steps, output_dir.name
        )
    if not output_dir.is_dir():
        items.append(CheckItem("output_dir", "FAIL", "output_dir is not a directory", {}))
        return _pack_check(
            items, ["output_dir is not a directory"], None, expected_steps, output_dir.name
        )

    items.append(
        CheckItem("output_dir", "PASS", "Output directory readable", {"name": output_dir.name})
    )

    state_path = adapter.locate_trainer_state(output_dir)
    trainer_state: Dict[str, Any] = {}
    trainer_state_ok = False
    train_metrics_complete = False
    global_step = None
    if state_path is None:
        items.append(CheckItem("trainer_state", "FAIL", "trainer_state.json not found", {}))
        reasons.append("Missing trainer_state.json")
    else:
        try:
            trainer_state = json.loads(state_path.read_text(encoding="utf-8"))
            if not isinstance(trainer_state, dict):
                raise ValueError("root not object")
            trainer_state_ok = True
            items.append(CheckItem("trainer_state", "PASS", "trainer_state.json parsed", {}))
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            items.append(
                CheckItem("trainer_state", "FAIL", f"Invalid trainer_state.json: {exc}", {})
            )
            reasons.append("Invalid trainer_state.json")

    if trainer_state:
        try:
            global_step = (
                int(trainer_state["global_step"])
                if trainer_state.get("global_step") is not None
                else None
            )
        except (TypeError, ValueError):
            global_step = None
        if expected_steps is not None:
            if global_step is None:
                items.append(CheckItem("global_step", "FAIL", "Missing global_step", {}))
                reasons.append("Missing global_step")
            elif global_step >= int(expected_steps):
                items.append(
                    CheckItem(
                        "global_step",
                        "PASS",
                        f"global_step={global_step} >= {expected_steps}",
                        {"global_step": global_step},
                    )
                )
            else:
                items.append(
                    CheckItem(
                        "global_step",
                        "FAIL",
                        f"global_step={global_step} < {expected_steps}",
                        {"global_step": global_step},
                    )
                )
                reasons.append("Insufficient steps")
        history = trainer_state.get("log_history") or []
        bad = []
        loss_seen = eval_seen = 0
        if isinstance(history, list):
            for idx, entry in enumerate(history):
                if not isinstance(entry, dict):
                    continue
                if "loss" in entry:
                    loss_seen += 1
                    if not is_finite_number(entry["loss"]):
                        bad.append(f"loss@{idx}")
                if "eval_loss" in entry:
                    eval_seen += 1
                    if not is_finite_number(entry["eval_loss"]):
                        bad.append(f"eval_loss@{idx}")
        if bad:
            items.append(
                CheckItem("loss_finite", "FAIL", f"Non-finite losses: {', '.join(bad[:10])}", {})
            )
            reasons.append("Non-finite loss in log_history")
        else:
            items.append(
                CheckItem(
                    "loss_finite",
                    "PASS" if (loss_seen or eval_seen) else "WARN",
                    f"Finite losses (loss={loss_seen}, eval_loss={eval_seen})",
                    {},
                )
            )

        train_metrics: Dict[str, Any] = {}
        for key in ("train_runtime", "train_loss", "train_samples_per_second"):
            if key in trainer_state:
                train_metrics[key] = trainer_state[key]
        if isinstance(history, list):
            for entry in reversed(history):
                if isinstance(entry, dict):
                    for key in ("train_runtime", "train_loss", "train_samples_per_second"):
                        if key in entry and key not in train_metrics:
                            train_metrics[key] = entry[key]
                    if train_metrics:
                        break
        missing = [
            k
            for k in ("train_runtime", "train_loss", "train_samples_per_second")
            if k not in train_metrics
        ]
        if missing:
            items.append(
                CheckItem("train_metrics", "WARN", f"Missing end metrics: {', '.join(missing)}", {})
            )
            reasons.append("Missing train end metrics")
        else:
            train_metrics_complete = True
            items.append(
                CheckItem(
                    "train_metrics",
                    "PASS",
                    "train_runtime/train_loss/train_samples_per_second present",
                    train_metrics,
                )
            )

    artifacts = adapter.find_adapter_artifacts(output_dir)
    if artifacts.checkpoints:
        items.append(
            CheckItem(
                "checkpoints",
                "PASS",
                f"{len(artifacts.checkpoints)} non-empty checkpoint(s)",
                {"names": [p.name for p in artifacts.checkpoints]},
            )
        )
    else:
        items.append(CheckItem("checkpoints", "WARN", "No non-empty checkpoint-* dirs", {}))
        reasons.append("No checkpoints")

    requested_type = (training_type or "auto").lower()
    if requested_type in {"lora", "qlora", "adapter"}:
        requested_type = "peft"
    if requested_type not in {"auto", "peft", "full"}:
        items.append(
            CheckItem("training_type", "FAIL", f"Unsupported training_type={training_type}", {})
        )
        reasons.append("Unsupported training type")
        return _pack_check(items, reasons, global_step, expected_steps, output_dir.name)

    cfg_ok = False
    cfg_invalid = False
    for cfg_path in artifacts.adapter_configs:
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            if isinstance(cfg, dict):
                cfg_ok = True
                items.append(
                    CheckItem(
                        "adapter_config",
                        "PASS",
                        f"adapter_config.json ok (peft_type={cfg.get('peft_type')})",
                        {},
                    )
                )
                break
        except (OSError, json.JSONDecodeError):
            cfg_invalid = True
            items.append(CheckItem("adapter_config", "FAIL", "adapter_config.json invalid", {}))
            reasons.append("Invalid adapter_config.json")
            break

    adapter_nonzero = [w for w in artifacts.weight_files if w.get("ok")]
    adapter_zero = [w for w in artifacts.weight_files if w.get("size_bytes") == 0]
    roots = [output_dir, *artifacts.checkpoints]
    full_weights = find_weight_files(roots, FULL_MODEL_WEIGHT_NAMES)
    full_nonzero = [w for w in full_weights if w.get("ok")]
    full_zero = [w for w in full_weights if w.get("size_bytes") == 0]
    resolved_type = requested_type
    if requested_type == "auto":
        resolved_type = "peft" if (cfg_ok or adapter_nonzero or adapter_zero) else "full"
    items.append(
        CheckItem(
            "training_type",
            "INFO",
            f"requested={requested_type}, resolved={resolved_type}",
            {"requested": requested_type, "resolved": resolved_type},
        )
    )

    if resolved_type == "peft":
        if not cfg_ok and not cfg_invalid:
            items.append(CheckItem("adapter_config", "FAIL", "adapter_config.json not found", {}))
            reasons.append("Missing adapter_config.json")
        if adapter_nonzero:
            items.append(
                CheckItem(
                    "adapter_weights",
                    "PASS",
                    f"{len(adapter_nonzero)} non-zero adapter weight file(s)",
                    {"files": adapter_nonzero},
                )
            )
        elif adapter_zero:
            items.append(
                CheckItem("adapter_weights", "FAIL", "Adapter weight file(s) exist but size=0", {})
            )
            reasons.append("Empty adapter weights")
        else:
            items.append(CheckItem("adapter_weights", "FAIL", "No adapter weight files found", {}))
            reasons.append("Missing adapter weights")
    else:
        if not any(i.name == "adapter_config" for i in items):
            items.append(
                CheckItem(
                    "adapter_config", "INFO", "Adapter config not required for full training", {}
                )
            )
        if full_nonzero:
            items.append(
                CheckItem(
                    "model_weights",
                    "PASS",
                    f"{len(full_nonzero)} non-zero full-model weight file(s)",
                    {"files": full_nonzero},
                )
            )
        elif full_zero:
            items.append(
                CheckItem("model_weights", "FAIL", "Full-model weight file(s) exist but size=0", {})
            )
            reasons.append("Empty full-model weights")
        else:
            items.append(CheckItem("model_weights", "FAIL", "No full-model weight files found", {}))
            reasons.append("Missing full-model weights")

    items.append(
        CheckItem("resume_artifacts", "INFO", "Optional resume files are not required", {})
    )

    life = summarize_lifecycle(lifecycle_path(output_dir))
    lifecycle_finished = False
    if not life.get("present") or life.get("event_count", 0) == 0:
        items.append(
            CheckItem(
                "lifecycle",
                "INFO",
                "No train_guard_lifecycle.jsonl yet (optional; produced by run watch)",
                {},
            )
        )
    else:
        phase = str(life.get("phase") or PHASE_NONE)
        detail = {
            "phase": phase,
            "event_count": life.get("event_count"),
            "global_step": life.get("global_step"),
            "checkpoints": life.get("checkpoints") or [],
        }
        if not life.get("has_training_events"):
            watcher = life.get("watcher") or {}
            items.append(
                CheckItem(
                    "lifecycle",
                    "INFO",
                    "Watcher activity observed; no training lifecycle event recorded",
                    {**detail, "watcher": watcher},
                )
            )
        elif phase == PHASE_ABORTED:
            items.append(CheckItem("lifecycle", "FAIL", f"Lifecycle phase={phase}", detail))
            reasons.append("Lifecycle aborted")
        elif phase == PHASE_FINISHED:
            lifecycle_finished = True
            items.append(CheckItem("lifecycle", "PASS", f"Lifecycle phase={phase}", detail))
        else:
            items.append(
                CheckItem(
                    "lifecycle",
                    "WARN",
                    f"Lifecycle phase={phase} (watch did not record finish)",
                    detail,
                )
            )
            reasons.append(f"Lifecycle incomplete ({phase})")

    step_complete = global_step is not None and (
        expected_steps is None or global_step >= int(expected_steps)
    )
    completion_evidence = [
        name
        for name, present in (
            ("trainer_state", trainer_state_ok),
            ("step", step_complete),
            ("checkpoint", bool(artifacts.checkpoints)),
            ("trainer_end_metrics", train_metrics_complete),
            ("lifecycle_finished", lifecycle_finished),
        )
        if present
    ]
    terminal_evidence = lifecycle_finished or train_metrics_complete
    if len(completion_evidence) >= 3 and terminal_evidence:
        items.append(
            CheckItem(
                "completion_evidence",
                "PASS",
                "Independent completion evidence is sufficient",
                {"evidence": completion_evidence},
            )
        )
    else:
        items.append(
            CheckItem(
                "completion_evidence",
                "WARN",
                "Insufficient independent evidence to declare training complete",
                {"evidence": completion_evidence},
            )
        )
        reasons.append("Insufficient completion evidence")

    return _pack_check(items, reasons, global_step, expected_steps, output_dir.name)


def _pack_check(
    items: List[CheckItem],
    reasons: List[str],
    global_step: Optional[int],
    expected_steps: Optional[int],
    output_name: str,
) -> Dict[str, Any]:
    overall = overall_status(items)
    uniq = []
    for r in reasons:
        if r not in uniq:
            uniq.append(r)
    if overall == "PASS" and not uniq:
        uniq = ["All critical checks passed"]
    return dict(
        json_safe(
            redact_value(
                {
                    "tool": "train_guard",
                    "command": "run check",
                    "version": __version__,
                    "timestamp": utc_now_iso(),
                    "overall_status": overall,
                    "output_dir_name": output_name,
                    "expected_steps": expected_steps,
                    "global_step": global_step,
                    "reasons": uniq,
                    "checks": [i.__dict__ for i in items],
                    "disclaimer": "Artifact completeness only; not a quality or domain validity claim.",
                }
            )
        )
    )
