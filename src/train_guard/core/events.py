"""Unified training event and dataset record types."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TrainEvent:
    """Normalized training metric event."""

    step: Optional[float] = None
    epoch: Optional[float] = None
    loss: Optional[float] = None
    eval_loss: Optional[float] = None
    learning_rate: Optional[float] = None
    grad_norm: Optional[float] = None
    throughput: Optional[float] = None
    source: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return asdict(self)


@dataclass
class MediaRef:
    """Reference to a media file path (relative or absolute)."""

    path: str
    field: str = "media"


@dataclass
class MessagesOrIO:
    """Either chat messages or input/output pair."""

    messages: Optional[List[Dict[str, Any]]] = None
    input_text: Optional[str] = None
    output_text: Optional[str] = None


@dataclass
class NormalizedRecord:
    """Normalized annotation record."""

    index: int
    raw_keys: List[str] = field(default_factory=list)
    group_id: Optional[str] = None
    split: Optional[str] = None
    media: List[MediaRef] = field(default_factory=list)
    content: MessagesOrIO = field(default_factory=MessagesOrIO)


@dataclass
class CheckItem:
    """Single check result."""

    name: str
    status: str  # PASS / WARN / FAIL / INFO
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


def overall_status(items: List[CheckItem]) -> str:
    """Aggregate PASS/WARN/FAIL."""
    statuses = {i.status for i in items}
    if "FAIL" in statuses:
        return "FAIL"
    if "WARN" in statuses:
        return "WARN"
    return "PASS"
