"""Generic JSON / JSONL dataset adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional

from ..core.events import MediaRef, MessagesOrIO, NormalizedRecord
from .base import FieldMap


class GenericDatasetAdapter:
    """Domain-neutral JSON/JSONL adapter with no framework inheritance."""

    name = "generic"

    def __init__(self, field_map: Optional[FieldMap] = None) -> None:
        self.fields = field_map or FieldMap()

    def iter_objects(
        self, path: Path, *, sample_limit: Optional[int] = None
    ) -> Iterator[Dict[str, Any]]:
        """Yield raw objects, streaming JSONL one physical line at a time."""
        count = 0
        if sample_limit is not None and int(sample_limit) <= 0:
            return
        if path.suffix.lower() == ".jsonl":
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    text = line.strip()
                    if not text:
                        continue
                    obj = json.loads(text)
                    if isinstance(obj, dict):
                        if sample_limit is not None and count >= int(sample_limit):
                            break
                        count += 1
                        yield obj
            return

        text = path.read_text(encoding="utf-8")
        stripped = text.lstrip()
        if not stripped:
            return
            yield  # pragma: no cover
        if not stripped.startswith("[") and not stripped.startswith("{"):
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict):
                    if sample_limit is not None and count >= int(sample_limit):
                        break
                    count += 1
                    yield obj
            return
        data = json.loads(text)
        if isinstance(data, list):
            for obj in data:
                if isinstance(obj, dict):
                    if sample_limit is not None and count >= int(sample_limit):
                        break
                    count += 1
                    yield obj
            return
        if isinstance(data, dict):
            for key in ("data", "samples", "instances", "annotations"):
                if isinstance(data.get(key), list):
                    for obj in data[key]:
                        if isinstance(obj, dict):
                            if sample_limit is not None and count >= int(sample_limit):
                                break
                            count += 1
                            yield obj
                    return
            if sample_limit is None or int(sample_limit) > 0:
                yield data

    def iter_records(
        self, path: Path, *, sample_limit: Optional[int] = None
    ) -> Iterator[NormalizedRecord]:
        """Stream records with optional sample limit."""
        for idx, raw in enumerate(self.iter_objects(path, sample_limit=sample_limit)):
            rec = NormalizedRecord(
                index=idx,
                raw_keys=sorted(raw.keys()),
                group_id=self.extract_group_id(raw),
                split=self.extract_split(raw),
                media=self.extract_media_refs(raw),
                content=self.extract_messages_or_io(raw),
            )
            yield rec

    def extract_media_refs(self, record: Mapping[str, Any]) -> List[MediaRef]:
        """Extract image/media paths."""
        raw = None
        field = self.fields.media
        # ``images`` remains a read-only compatibility input, while ``media``
        # is the public configuration field.
        for key in (self.fields.media, "images", "image", "image_path", "file_name"):
            if key in record:
                raw = record[key]
                field = key
                break
        if raw is None:
            return []
        paths: List[str] = []
        if isinstance(raw, str):
            if raw:
                paths.append(raw)
        elif isinstance(raw, list):
            for item in raw:
                if isinstance(item, str):
                    paths.append(item)
                elif isinstance(item, dict):
                    for k in ("path", "image", "file_name", "url"):
                        if isinstance(item.get(k), str):
                            paths.append(item[k])
                            break
        return [MediaRef(path=p, field=field) for p in paths]

    def extract_group_id(self, record: Mapping[str, Any]) -> Optional[str]:
        """Extract the configured group id, falling back to a generic id."""
        for key in (self.fields.group_id, "id"):
            if key in record and record[key] is not None:
                return str(record[key])
        return None

    def extract_split(self, record: Mapping[str, Any]) -> Optional[str]:
        """Extract split field."""
        if self.fields.split in record and record[self.fields.split] is not None:
            return str(record[self.fields.split])
        return None

    def extract_messages_or_io(self, record: Mapping[str, Any]) -> MessagesOrIO:
        """Extract messages or generic input/output."""
        messages = record.get(self.fields.messages)
        if isinstance(messages, list):
            return MessagesOrIO(messages=[m for m in messages if isinstance(m, dict)])
        inp = record.get(self.fields.input_field)
        out = record.get(self.fields.output_field)
        if out is None and "answer" in record:  # compatibility input only
            out = record.get("answer")
        return MessagesOrIO(
            input_text=None if inp is None else str(inp),
            output_text=None if out is None else str(out),
        )

    def extract_answer_text(self, record: Mapping[str, Any]) -> Optional[str]:
        """Assistant/output answer text for emptiness checks."""
        content = self.extract_messages_or_io(record)
        if content.output_text is not None:
            return content.output_text
        if content.messages:
            for msg in reversed(content.messages):
                if str(msg.get("role", "")).lower() in {"assistant", "gpt"}:
                    c = msg.get("content")
                    return "" if c is None else str(c)
        return None
