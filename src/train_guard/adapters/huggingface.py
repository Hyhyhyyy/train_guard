"""Log metric parsing and HuggingFace / LLaMAFactory framework adapters."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from ..core.events import TrainEvent
from .base import AdapterArtifacts

_METRIC_PATTERNS: Dict[str, re.Pattern[str]] = {
    "eval_loss": re.compile(
        r"""(?<![A-Za-z0-9_])(?:['"]?eval_loss['"]?)\s*[:=]\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|nan|NaN|inf|Inf|Infinity|-inf|-Inf|-Infinity)""",
        re.IGNORECASE,
    ),
    "loss": re.compile(
        r"""(?<![A-Za-z0-9_])(?:['"]?loss['"]?)\s*[:=]\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|nan|NaN|inf|Inf|Infinity|-inf|-Inf|-Infinity)""",
        re.IGNORECASE,
    ),
    "learning_rate": re.compile(
        r"""(?<![A-Za-z0-9_])(?:['"]?(?:learning_rate|lr)['"]?)\s*[:=]\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|nan|NaN|inf|Inf|Infinity)""",
        re.IGNORECASE,
    ),
    "grad_norm": re.compile(
        r"""(?<![A-Za-z0-9_])(?:['"]?grad_norm['"]?)\s*[:=]\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|nan|NaN|inf|Inf|Infinity)""",
        re.IGNORECASE,
    ),
    "epoch": re.compile(
        r"""(?<![A-Za-z0-9_])(?:['"]?epoch['"]?)\s*[:=]\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)""",
        re.IGNORECASE,
    ),
    "step": re.compile(
        r"""(?<![A-Za-z0-9_])(?:['"]?(?:global_step|step)['"]?)\s*[:=]\s*(\d+)""",
        re.IGNORECASE,
    ),
    "throughput": re.compile(
        r"""(?<![A-Za-z0-9_])(?:['"]?(?:train_samples_per_second|samples_per_second|throughput)['"]?)\s*[:=]\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)""",
        re.IGNORECASE,
    ),
}

LORA_WEIGHT_NAMES = (
    "adapter_model.safetensors",
    "adapter_model.bin",
    "adapter_model.pt",
    "pytorch_model.bin",
    "model.safetensors",
)


def parse_numeric_token(token: str) -> Optional[float]:
    """Parse numeric token including nan/inf."""
    t = token.strip()
    lower = t.lower()
    if lower == "nan":
        return float("nan")
    if lower in {"inf", "infinity", "+inf", "+infinity"}:
        return float("inf")
    if lower in {"-inf", "-infinity"}:
        return float("-inf")
    try:
        return float(t)
    except ValueError:
        return None


def parse_training_metrics(line: str) -> Dict[str, float]:
    """Parse common trainer log formats; never map eval_loss to loss."""
    metrics: Dict[str, float] = {}
    for key in ("eval_loss", "loss", "learning_rate", "grad_norm", "epoch", "step", "throughput"):
        for m in _METRIC_PATTERNS[key].finditer(line):
            if key == "loss":
                prefix = line[max(0, m.start() - 5) : m.start()]
                if prefix.lower().endswith("eval_"):
                    continue
            value = parse_numeric_token(m.group(1))
            if value is not None:
                metrics[key] = value
                break
    return metrics


def is_bad_loss(value: Optional[float]) -> bool:
    """True if NaN or Inf."""
    if value is None:
        return False
    return math.isnan(value) or math.isinf(value)


def is_finite_number(value: Any) -> bool:
    """True if value is a finite float."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(f)


def list_checkpoint_dirs(output_dir: Path) -> List[Path]:
    """Non-empty checkpoint-* directories."""
    if not output_dir.is_dir():
        return []
    found: List[Path] = []
    try:
        for p in sorted(output_dir.iterdir()):
            if p.is_dir() and p.name.startswith("checkpoint-"):
                try:
                    if any(p.iterdir()):
                        found.append(p)
                except OSError:
                    continue
    except OSError:
        return []
    return found


def find_lora_weights(search_roots: List[Path]) -> List[Dict[str, Any]]:
    """Find adapter weight files."""
    results: List[Dict[str, Any]] = []
    seen = set()
    for root in search_roots:
        if not root.is_dir():
            continue
        for name in LORA_WEIGHT_NAMES:
            path = root / name
            key = str(path)
            if key in seen or not path.is_file():
                continue
            seen.add(key)
            try:
                size = path.stat().st_size
            except OSError as exc:
                results.append({"path": path.name, "parent": root.name, "size_bytes": None, "ok": False, "error": str(exc)})
                continue
            results.append({"path": path.name, "parent": root.name or ".", "size_bytes": size, "ok": size > 0})
    return results


class HuggingFaceFrameworkAdapter:
    """Hugging Face Trainer layout."""

    name = "huggingface"

    def locate_trainer_state(self, output_dir: Path) -> Optional[Path]:
        """Locate trainer_state.json in root or latest checkpoint."""
        direct = output_dir / "trainer_state.json"
        if direct.is_file():
            return direct
        checkpoints = list_checkpoint_dirs(output_dir)
        if checkpoints:
            cand = checkpoints[-1] / "trainer_state.json"
            if cand.is_file():
                return cand
        return None

    def iter_log_events(
        self, log_path: Optional[Path], state_path: Optional[Path]
    ) -> Iterator[TrainEvent]:
        """Yield events from log lines and/or trainer_state log_history."""
        if log_path and log_path.is_file():
            with log_path.open("r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    m = parse_training_metrics(line)
                    if m:
                        yield TrainEvent(
                            step=m.get("step"),
                            epoch=m.get("epoch"),
                            loss=m.get("loss"),
                            eval_loss=m.get("eval_loss"),
                            learning_rate=m.get("learning_rate"),
                            grad_norm=m.get("grad_norm"),
                            throughput=m.get("throughput"),
                            source="log",
                        )
        if state_path and state_path.is_file():
            try:
                data = json.loads(state_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return
            history = data.get("log_history") or []
            if isinstance(history, list):
                for item in history:
                    if not isinstance(item, dict):
                        continue
                    yield TrainEvent(
                        step=item.get("step") or item.get("global_step"),
                        epoch=_as_float(item.get("epoch")),
                        loss=_as_float(item.get("loss")),
                        eval_loss=_as_float(item.get("eval_loss")),
                        learning_rate=_as_float(item.get("learning_rate")),
                        grad_norm=_as_float(item.get("grad_norm")),
                        throughput=_as_float(
                            item.get("train_samples_per_second") or item.get("samples_per_second")
                        ),
                        source="trainer_state",
                    )

    def list_checkpoints(self, output_dir: Path) -> List[Path]:
        """List checkpoints."""
        return list_checkpoint_dirs(output_dir)

    def find_adapter_artifacts(self, output_dir: Path) -> AdapterArtifacts:
        """Discover adapter configs and weights."""
        checkpoints = list_checkpoint_dirs(output_dir)
        roots = [output_dir, *checkpoints]
        configs: List[Path] = []
        for root in roots:
            cfg = root / "adapter_config.json"
            if cfg.is_file():
                configs.append(cfg)
        return AdapterArtifacts(
            adapter_configs=configs,
            weight_files=find_lora_weights(roots),
            checkpoints=checkpoints,
        )


class LLaMAFactoryFrameworkAdapter(HuggingFaceFrameworkAdapter):
    """LLaMAFactory uses HF Trainer artifacts with similar layout."""

    name = "llamafactory"


class GenericFrameworkAdapter(HuggingFaceFrameworkAdapter):
    """Generic fallback = HF layout conventions."""

    name = "generic"


def get_framework_adapter(name: str) -> HuggingFaceFrameworkAdapter:
    """Factory for framework adapters."""
    key = (name or "generic").lower()
    if key in {"llamafactory", "llama-factory", "lf"}:
        return LLaMAFactoryFrameworkAdapter()
    if key in {"huggingface", "hf", "transformers", "trainer"}:
        return HuggingFaceFrameworkAdapter()
    return GenericFrameworkAdapter()


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
