"""Adapter protocol interfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Protocol

from ..core.events import MediaRef, MessagesOrIO, NormalizedRecord, TrainEvent


@dataclass
class FieldMap:
    """Configurable field names for datasets."""

    messages: str = "messages"
    media: str = "media"
    input_field: str = "input"
    output_field: str = "output"
    group_id: str = "group_id"
    split: str = "split"
    prediction: str = "prediction"
    reference: str = "reference"


@dataclass
class AdapterArtifacts:
    """LoRA / adapter artifacts discovered under an output dir."""

    adapter_configs: List[Path] = field(default_factory=list)
    weight_files: List[Dict[str, Any]] = field(default_factory=list)
    checkpoints: List[Path] = field(default_factory=list)


class DatasetAdapter(Protocol):
    """Dataset format adapter."""

    name: str

    def iter_records(
        self, path: Path, *, sample_limit: Optional[int] = None
    ) -> Iterator[NormalizedRecord]:
        """Stream normalized records with bounded sample_limit."""

    def extract_media_refs(self, record: Mapping[str, Any]) -> List[MediaRef]:
        """Extract media paths from a raw record."""

    def extract_group_id(self, record: Mapping[str, Any]) -> Optional[str]:
        """Extract group id (leakage unit)."""

    def extract_split(self, record: Mapping[str, Any]) -> Optional[str]:
        """Extract split label."""

    def extract_messages_or_io(self, record: Mapping[str, Any]) -> MessagesOrIO:
        """Extract messages or input/output."""


class FrameworkAdapter(Protocol):
    """Training framework adapter."""

    name: str

    def locate_trainer_state(self, output_dir: Path) -> Optional[Path]:
        """Find trainer_state.json if present."""

    def iter_log_events(
        self, log_path: Optional[Path], state_path: Optional[Path]
    ) -> Iterator[TrainEvent]:
        """Yield normalized training events."""

    def list_checkpoints(self, output_dir: Path) -> List[Path]:
        """List non-empty checkpoint-* dirs."""

    def find_adapter_artifacts(self, output_dir: Path) -> AdapterArtifacts:
        """Discover adapter config/weights."""
