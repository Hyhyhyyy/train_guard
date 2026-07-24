"""data check / inventory / compare commands."""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .. import __version__
from ..adapters.base import FieldMap
from ..adapters.generic import GenericDatasetAdapter
from ..core.exitcodes import EXIT_CONFIG, EXIT_FAIL, EXIT_OK, EXIT_USAGE, EXIT_WARN
from ..core.io_util import sha256_file, utc_now_iso, write_json
from ..core.optional import try_import_pil
from ..core.privacy import group_id_hash, redact_value
from ..report.html import render_html_report
from .cache import FileCheckCache

LOGGER = logging.getLogger("train_guard.data")


def _field_map_from_cfg(cfg: Dict[str, Any]) -> FieldMap:
    return FieldMap(
        messages=str(cfg.get("messages_field") or "messages"),
        media=str(cfg.get("media_field") or "media"),
        input_field=str(cfg.get("input_field") or "input"),
        output_field=str(cfg.get("output_field") or "output"),
        group_id=str(cfg.get("group_id_field") or "group_id"),
        split=str(cfg.get("split_field") or "split"),
    )


def run_data_check(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Read-only dataset/media/message integrity check."""
    annotation = cfg.get("annotation")
    if not annotation:
        raise ValueError("annotation path is required")
    ann_path = Path(annotation)
    data_root = Path(cfg["data_root"]) if cfg.get("data_root") else ann_path.parent
    sample_limit = cfg.get("sample_limit", 1000)
    if cfg.get("full_scan"):
        sample_limit = None
    max_examples = int(cfg.get("max_examples") or 20)
    compute_hash = bool(cfg.get("compute_hash"))
    verify_images = bool(cfg.get("verify_images", True))
    Image = try_import_pil() if verify_images else None
    if verify_images and Image is None:
        LOGGER.warning("Pillow not installed; skipping image verify")

    cache = None
    cache_path = cfg.get("cache_db")
    if cache_path:
        cache = FileCheckCache(Path(cache_path))

    adapter = GenericDatasetAdapter(_field_map_from_cfg(cfg))
    if not data_root.exists():
        raise FileNotFoundError(f"data_root does not exist: {data_root}")

    issues: Dict[str, List[str]] = defaultdict(list)
    ext_counter: Counter[str] = Counter()
    media_per_sample: Counter[int] = Counter()
    hashes: Dict[str, List[int]] = defaultdict(list)
    group_splits: Dict[str, Set[str]] = defaultdict(set)

    stats = {
        "total_seen": 0,
        "empty_records": 0,
        "empty_answers": 0,
        "missing_media": 0,
        "zero_byte_files": 0,
        "media_checked": 0,
        "media_verify_failed": 0,
        "duplicate_hash_groups": 0,
        "group_leak_count": 0,
        "pillow_available": Image is not None,
        "hash_enabled": compute_hash,
        "sample_limit": sample_limit,
        "full_scan": bool(cfg.get("full_scan")),
    }

    # Single streaming pass
    stats["total_samples"] = None  # unknown without full scan
    for rec in adapter.iter_records(ann_path, sample_limit=sample_limit):
        stats["total_seen"] += 1
        raw_empty = not rec.raw_keys
        if raw_empty:
            stats["empty_records"] += 1
            if len(issues["empty_records"]) < max_examples:
                issues["empty_records"].append(f"index={rec.index}")
            continue

        answer = None
        if rec.content.output_text is not None:
            answer = rec.content.output_text
        elif rec.content.messages:
            for msg in reversed(rec.content.messages):
                if str(msg.get("role", "")).lower() in {"assistant", "gpt"}:
                    c = msg.get("content")
                    answer = "" if c is None else str(c)
                    break
        if answer is not None and str(answer).strip() == "":
            stats["empty_answers"] += 1
            if len(issues["empty_answers"]) < max_examples:
                issues["empty_answers"].append(f"index={rec.index}")

        media_per_sample[len(rec.media)] += 1
        for ref in rec.media:
            p = Path(ref.path)
            if not p.is_absolute():
                p = data_root / ref.path
            ext = p.suffix.lower() or "<none>"
            ext_counter[ext] += 1
            stats["media_checked"] += 1
            if not p.exists():
                stats["missing_media"] += 1
                if len(issues["missing_media"]) < max_examples:
                    issues["missing_media"].append(f"index={rec.index} media={Path(ref.path).name}")
                continue
            try:
                size = p.stat().st_size
            except OSError as exc:
                if len(issues["stat_errors"]) < max_examples:
                    issues["stat_errors"].append(f"index={rec.index} err={type(exc).__name__}")
                continue
            if size == 0:
                stats["zero_byte_files"] += 1
                if len(issues["zero_byte_files"]) < max_examples:
                    issues["zero_byte_files"].append(f"index={rec.index}")

            if Image is not None and verify_images and size > 0:
                cached = cache.get(p, "verify") if cache else None
                if cached is not None:
                    ok = bool(cached.get("ok"))
                else:
                    try:
                        with Image.open(p) as im:
                            im.verify()
                        ok = True
                        err = None
                    except Exception as exc:  # noqa: BLE001
                        ok = False
                        err = type(exc).__name__
                    if cache:
                        cache.set(p, "verify", {"ok": ok, "error": err})
                if not ok:
                    stats["media_verify_failed"] += 1
                    if len(issues["media_verify_failed"]) < max_examples:
                        issues["media_verify_failed"].append(f"index={rec.index}")

            if compute_hash and size > 0:
                cached = cache.get(p, "sha256") if cache else None
                if cached and cached.get("digest"):
                    digest = str(cached["digest"])
                else:
                    digest = sha256_file(p)
                    if cache:
                        cache.set(p, "sha256", {"digest": digest, "ok": True})
                hashes[digest].append(rec.index)

        if rec.group_id is not None and rec.split is not None:
            group_splits[rec.group_id].add(str(rec.split))

    stats["scanned_samples"] = stats["total_seen"]
    if sample_limit is None:
        stats["total_samples"] = stats["total_seen"]
    else:
        stats["total_samples"] = stats["total_seen"]  # scanned bound

    if cache:
        cache.close()

    leak_examples: List[str] = []
    for gid, splits in group_splits.items():
        normalized = {s.lower() for s in splits}
        trainish = {"train", "training"}
        evalish = {"val", "valid", "validation", "test", "dev", "eval"}
        if (normalized & trainish) and (normalized & evalish):
            stats["group_leak_count"] += 1
            if len(leak_examples) < max_examples:
                leak_examples.append(f"{group_id_hash(gid)} splits={sorted(splits)}")
    issues["group_leak"] = leak_examples

    dup_groups = {h: idxs for h, idxs in hashes.items() if len(set(idxs)) > 1}
    stats["duplicate_hash_groups"] = len(dup_groups)
    for h, idxs in list(dup_groups.items())[:max_examples]:
        issues["duplicate_media"].append(f"sha256={h[:12]}… indices={sorted(set(idxs))[:10]}")

    overall = "PASS"
    if stats["missing_media"] or stats["empty_answers"] or stats["group_leak_count"] or stats["zero_byte_files"]:
        overall = "FAIL"
    elif stats["media_verify_failed"] or stats["duplicate_hash_groups"] or stats["empty_records"]:
        overall = "WARN"

    report = {
        "tool": "train_guard",
        "command": "data check",
        "version": __version__,
        "timestamp": utc_now_iso(),
        "overall_status": overall,
        "stats": stats,
        "extension_counts": dict(ext_counter),
        "media_per_sample_distribution": {str(k): v for k, v in sorted(media_per_sample.items())},
        "issues": dict(issues),
        "privacy_note": "Public fields are redacted; group ids hashed; no raw media content.",
        "disclaimer": "Integrity check only; does not validate task quality or domain correctness.",
    }
    return redact_value(report)


def run_data_inventory(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Lightweight streaming inventory (bounded memory)."""
    annotation = cfg.get("annotation")
    if not annotation:
        raise ValueError("annotation path is required")
    ann_path = Path(annotation)
    sample_limit = cfg.get("sample_limit")
    adapter = GenericDatasetAdapter(_field_map_from_cfg(cfg))
    key_counter: Counter[str] = Counter()
    media_hist: Counter[int] = Counter()
    split_counter: Counter[str] = Counter()
    n = 0
    for rec in adapter.iter_records(ann_path, sample_limit=sample_limit):
        n += 1
        for k in rec.raw_keys:
            key_counter[k] += 1
        media_hist[len(rec.media)] += 1
        if rec.split:
            split_counter[rec.split] += 1
    report = {
        "tool": "train_guard",
        "command": "data inventory",
        "version": __version__,
        "timestamp": utc_now_iso(),
        "overall_status": "PASS" if n else "FAIL",
        "sample_count": n,
        "sample_limit": sample_limit,
        "field_presence": dict(key_counter),
        "media_count_histogram": {str(k): v for k, v in sorted(media_hist.items())},
        "split_counts": dict(split_counter),
    }
    return redact_value(report)


def run_data_compare(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Compare two annotation files by group_id and media basenames."""
    left = cfg.get("left")
    right = cfg.get("right")
    if not left or not right:
        raise ValueError("left and right annotation paths are required")
    adapter = GenericDatasetAdapter(_field_map_from_cfg(cfg))
    sample_limit = cfg.get("sample_limit")

    def collect(path: Path) -> Dict[str, Any]:
        groups: Set[str] = set()
        media_names: Set[str] = set()
        n = 0
        for rec in adapter.iter_records(path, sample_limit=sample_limit):
            n += 1
            if rec.group_id:
                groups.add(rec.group_id)
            for m in rec.media:
                media_names.add(Path(m.path).name)
        return {"count": n, "groups": groups, "media_names": media_names}

    a = collect(Path(left))
    b = collect(Path(right))
    overlap_groups = a["groups"] & b["groups"]
    only_a = a["groups"] - b["groups"]
    only_b = b["groups"] - a["groups"]
    media_overlap = a["media_names"] & b["media_names"]
    report = {
        "tool": "train_guard",
        "command": "data compare",
        "version": __version__,
        "timestamp": utc_now_iso(),
        "overall_status": "WARN" if overlap_groups else "PASS",
        "left_count": a["count"],
        "right_count": b["count"],
        "group_overlap_count": len(overlap_groups),
        "group_only_left": len(only_a),
        "group_only_right": len(only_b),
        "media_basename_overlap": len(media_overlap),
        "group_overlap_hashes": [group_id_hash(g) for g in sorted(overlap_groups)[:20]],
        "disclaimer": "Compares identifiers and media basenames only; no raw text emitted.",
    }
    return redact_value(report)


def write_data_reports(report: Dict[str, Any], report_dir: Path, stem: str) -> None:
    """Write JSON + HTML for a data command report."""
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / f"{stem}.json"
    write_json(json_path, report, overwrite=True)
    cards = [
        {"title": "Status", "value": report.get("overall_status"), "status": report.get("overall_status", "INFO")},
    ]
    rows = [[k, v] for k, v in report.items() if k not in {"issues"} and not isinstance(v, (dict, list))]
    if isinstance(report.get("stats"), dict):
        rows.extend([[f"stats.{k}", v] for k, v in report["stats"].items()])
    html_doc = render_html_report(
        f"Train Guard — {report.get('command')}",
        cards,
        [{"title": "Summary", "headers": ["Key", "Value"], "rows": rows}],
        str(report.get("disclaimer") or "Read-only dataset report."),
    )
    (report_dir / f"{stem}.html").write_text(html_doc, encoding="utf-8")
