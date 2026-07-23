"""Train Guard CLI — argparse with new command groups and legacy aliases."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import traceback
from pathlib import Path
from typing import Optional, Sequence

from . import __min_python__, __version__
from .core.exitcodes import (
    EXIT_CONFIG,
    EXIT_FAIL,
    EXIT_OK,
    EXIT_RUNTIME,
    EXIT_USAGE,
    EXIT_WARN,
)
from .core.io_util import write_json
from .data.commands import run_data_check, run_data_compare, run_data_inventory, write_data_reports
from .env.doctor import run_doctor, status_to_exit
from .eval.metrics import run_eval
from .report.html import render_html_report
from .run.commands import cmd_run_watch, run_manifest, run_run_check, run_run_compare


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def _warn_deprecated(old: str, new: str) -> None:
    print(f"[DEPRECATED] {old} → {new}; see docs/MIGRATION.md", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(
        prog="train_guard",
        description="Train Guard — read-only LLM/VLM training quality toolkit",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    # doctor
    p = sub.add_parser("doctor", help="Environment and model integrity check")
    p.add_argument("--model-path", default=None)
    p.add_argument("--expected-gpus", type=int, default=None)
    p.add_argument("--json-output", default=None)
    p.set_defaults(_handler="doctor")

    # data group
    p_data = sub.add_parser("data", help="Dataset commands")
    data_sub = p_data.add_subparsers(dest="data_command", required=True)

    p_dc = data_sub.add_parser("check", help="Dataset/media/message integrity check")
    p_dc.add_argument("--data-root", default=None)
    p_dc.add_argument("--annotation", required=True)
    p_dc.add_argument("--sample-limit", type=int, default=1000)
    p_dc.add_argument("--full-scan", action="store_true")
    p_dc.add_argument("--compute-hash", action="store_true")
    p_dc.add_argument("--verify-images", action=argparse.BooleanOptionalAction, default=True)
    p_dc.add_argument("--group-id-field", default=None)
    p_dc.add_argument("--split-field", default=None)
    p_dc.add_argument("--images-field", default=None)
    p_dc.add_argument("--messages-field", default=None)
    p_dc.add_argument("--answer-field", default=None)
    p_dc.add_argument("--report-dir", default="reports/data_check")
    p_dc.add_argument("--cache-db", default=None)
    p_dc.set_defaults(_handler="data_check")

    p_di = data_sub.add_parser("inventory", help="Streaming dataset inventory")
    p_di.add_argument("--annotation", required=True)
    p_di.add_argument("--sample-limit", type=int, default=None)
    p_di.add_argument("--group-id-field", default=None)
    p_di.add_argument("--report-dir", default="reports/data_inventory")
    p_di.set_defaults(_handler="data_inventory")

    p_dcmp = data_sub.add_parser("compare", help="Compare two annotation files")
    p_dcmp.add_argument("--left", required=True)
    p_dcmp.add_argument("--right", required=True)
    p_dcmp.add_argument("--sample-limit", type=int, default=None)
    p_dcmp.add_argument("--group-id-field", default=None)
    p_dcmp.add_argument("--report-dir", default="reports/data_compare")
    p_dcmp.set_defaults(_handler="data_compare")

    # run group
    p_run = sub.add_parser("run", help="Training run commands")
    run_sub = p_run.add_subparsers(dest="run_command", required=True)

    p_rw = run_sub.add_parser("watch", help="Read-only training watch")
    p_rw.add_argument("--once", action="store_true")
    p_rw.add_argument("--interval", type=int, default=30)
    p_rw.add_argument("--pid", type=int, default=None)
    p_rw.add_argument("--log-file", default=None)
    p_rw.add_argument("--framework", default="generic")
    p_rw.add_argument("--output-dir", default="reports/watch")
    p_rw.add_argument("--expected-gpus", type=int, default=None)
    p_rw.add_argument("--stale-log-minutes", type=float, default=15)
    p_rw.add_argument("--disk-free-gb-threshold", type=float, default=10.0)
    p_rw.set_defaults(_handler="run_watch")

    p_rc = run_sub.add_parser("check", help="Check whether a training run completed")
    p_rc.add_argument("--output-dir", required=True)
    p_rc.add_argument("--expected-steps", type=int, default=None)
    p_rc.add_argument("--json-output", default=None)
    p_rc.add_argument("--html-output", default=None)
    p_rc.set_defaults(_handler="run_check")

    p_rcomp = run_sub.add_parser("compare", help="Compare two run output directories")
    p_rcomp.add_argument("--left", required=True)
    p_rcomp.add_argument("--right", required=True)
    p_rcomp.add_argument("--json-output", default=None)
    p_rcomp.set_defaults(_handler="run_compare")

    # eval
    p_ev = sub.add_parser("eval", help="Evaluate predictions vs references")
    p_ev.add_argument("--predictions", required=True)
    p_ev.add_argument("--references", default=None)
    p_ev.add_argument("--prediction-field", default=None)
    p_ev.add_argument("--reference-field", default=None)
    p_ev.add_argument("--group-id-field", default=None)
    p_ev.add_argument("--label-field", default=None)
    p_ev.add_argument("--predicted-label-field", default=None)
    p_ev.add_argument("--keywords", default=None, help="Comma-separated keywords")
    p_ev.add_argument("--report-dir", default="reports/eval")
    p_ev.set_defaults(_handler="eval")

    # manifest
    p_mf = sub.add_parser("manifest", help="Write run manifest and experiment fingerprint")
    p_mf.add_argument("--output-dir", required=True)
    p_mf.add_argument("--framework", default="generic")
    p_mf.add_argument("--manifest-out", default=None)
    p_mf.add_argument("--expected-steps", type=int, default=None)
    p_mf.add_argument("--seed", default=None)
    p_mf.set_defaults(_handler="manifest")

    # bundle-info
    p_b = sub.add_parser("bundle-info", help="Show deploy / version info")
    p_b.set_defaults(_handler="bundle_info")

    # top-level compare alias → run compare
    p_cmp = sub.add_parser("compare", help="Alias for 'run compare'")
    p_cmp.add_argument("--left", required=True)
    p_cmp.add_argument("--right", required=True)
    p_cmp.add_argument("--json-output", default=None)
    p_cmp.set_defaults(_handler="run_compare")

    # ---- legacy aliases ----
    for legacy, handler, help_txt in [
        ("precheck", "legacy_precheck", "Deprecated alias for 'data check'"),
        ("monitor", "legacy_monitor", "Deprecated alias for 'run watch'"),
        ("postcheck", "legacy_postcheck", "Deprecated alias for 'run check'"),
        ("evaluate", "legacy_evaluate", "Deprecated alias for 'eval'"),
    ]:
        lp = sub.add_parser(legacy, help=help_txt)
        if legacy == "precheck":
            lp.add_argument("--data-root", default=None)
            lp.add_argument("--annotation", required=True)
            lp.add_argument("--sample-limit", type=int, default=1000)
            lp.add_argument("--full-scan", action="store_true")
            lp.add_argument("--compute-hash", action="store_true")
            lp.add_argument("--verify-images", action=argparse.BooleanOptionalAction, default=True)
            lp.add_argument("--group-id-field", default=None)
            lp.add_argument("--split-field", default=None)
            lp.add_argument("--images-field", default=None)
            lp.add_argument("--messages-field", default=None)
            lp.add_argument("--answer-field", default=None)
            lp.add_argument("--report-dir", default="reports/data_check")
            lp.add_argument("--cache-db", default=None)
        elif legacy == "monitor":
            lp.add_argument("--once", action="store_true")
            lp.add_argument("--interval", type=int, default=30)
            lp.add_argument("--pid", type=int, default=None)
            lp.add_argument("--log-file", default=None)
            lp.add_argument("--framework", default="generic")
            lp.add_argument("--output-dir", default="reports/watch")
            lp.add_argument("--expected-gpus", type=int, default=None)
            lp.add_argument("--stale-log-minutes", type=float, default=15)
            lp.add_argument("--disk-free-gb-threshold", type=float, default=10.0)
        elif legacy == "postcheck":
            lp.add_argument("--output-dir", required=True)
            lp.add_argument("--expected-steps", type=int, default=None)
            lp.add_argument("--json-output", default=None)
            lp.add_argument("--html-output", default=None)
        elif legacy == "evaluate":
            lp.add_argument("--predictions", required=True)
            lp.add_argument("--references", default=None)
            lp.add_argument("--prediction-field", default=None)
            lp.add_argument("--reference-field", default=None)
            lp.add_argument("--group-id-field", default=None)
            lp.add_argument("--label-field", default=None)
            lp.add_argument("--predicted-label-field", default=None)
            lp.add_argument("--keywords", default=None)
            lp.add_argument("--report-dir", default="reports/eval")
        lp.set_defaults(_handler=handler)

    return parser


def _dispatch(args: argparse.Namespace) -> int:
    handler = getattr(args, "_handler", None)

    if handler in {"legacy_precheck", "data_check"}:
        if handler == "legacy_precheck":
            _warn_deprecated("precheck", "data check")
        cfg = {
            "data_root": args.data_root,
            "annotation": args.annotation,
            "sample_limit": args.sample_limit,
            "full_scan": args.full_scan,
            "compute_hash": args.compute_hash,
            "verify_images": args.verify_images,
            "group_id_field": args.group_id_field,
            "split_field": getattr(args, "split_field", None),
            "images_field": getattr(args, "images_field", None),
            "messages_field": getattr(args, "messages_field", None),
            "answer_field": getattr(args, "answer_field", None),
            "cache_db": getattr(args, "cache_db", None),
        }
        report = run_data_check(cfg)
        write_data_reports(report, Path(args.report_dir), "data_check_report")
        print(f"data check — {report['overall_status']}")
        print(f"scanned={report['stats'].get('scanned_samples')} missing_media={report['stats'].get('missing_media')} "
              f"empty_answers={report['stats'].get('empty_answers')} group_leak={report['stats'].get('group_leak_count')}")
        return status_to_exit(report["overall_status"])

    if handler == "data_inventory":
        report = run_data_inventory({"annotation": args.annotation, "sample_limit": args.sample_limit, "group_id_field": args.group_id_field})
        write_data_reports(report, Path(args.report_dir), "data_inventory_report")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return status_to_exit(report["overall_status"])

    if handler == "data_compare":
        report = run_data_compare({"left": args.left, "right": args.right, "sample_limit": args.sample_limit, "group_id_field": args.group_id_field})
        write_data_reports(report, Path(args.report_dir), "data_compare_report")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return status_to_exit(report["overall_status"])

    if handler in {"legacy_monitor", "run_watch"}:
        if handler == "legacy_monitor":
            _warn_deprecated("monitor", "run watch")
        cfg = {
            "once": args.once,
            "interval": args.interval,
            "pid": args.pid,
            "log_file": args.log_file,
            "framework": args.framework,
            "output_dir": args.output_dir,
            "expected_gpu_count": args.expected_gpus,
            "stale_log_minutes": args.stale_log_minutes,
            "disk_free_gb_threshold": args.disk_free_gb_threshold,
        }
        return cmd_run_watch(cfg)

    if handler in {"legacy_postcheck", "run_check"}:
        if handler == "legacy_postcheck":
            _warn_deprecated("postcheck", "run check")
        report = run_run_check(Path(args.output_dir), expected_steps=args.expected_steps)
        print("=" * 60)
        print(f"run check — {report['overall_status']}")
        for item in report.get("checks") or []:
            print(f"[{item['status']:4}] {item['name']}: {item['message']}")
        for reason in report.get("reasons") or []:
            print(f"  - {reason}")
        if args.json_output:
            write_json(Path(args.json_output), report, overwrite=True)
            print(f"JSON: {args.json_output}")
        if getattr(args, "html_output", None):
            html_doc = render_html_report(
                "Train Guard — Run Check",
                [{"title": "Status", "value": report["overall_status"], "status": report["overall_status"]}],
                [
                    {"title": "Checks", "headers": ["Status", "Name", "Message"], "rows": [[c["status"], c["name"], c["message"]] for c in report.get("checks") or []]},
                    {"title": "Reasons", "headers": ["Reason"], "rows": [[r] for r in report.get("reasons") or []]},
                ],
                report.get("disclaimer") or "",
            )
            Path(args.html_output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.html_output).write_text(html_doc, encoding="utf-8")
            print(f"HTML: {args.html_output}")
        return status_to_exit(report["overall_status"])

    if handler == "run_compare":
        report = run_run_compare(Path(args.left), Path(args.right))
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if getattr(args, "json_output", None):
            write_json(Path(args.json_output), report, overwrite=True)
        return EXIT_OK

    if handler in {"legacy_evaluate", "eval"}:
        if handler == "legacy_evaluate":
            _warn_deprecated("evaluate", "eval")
        keywords = None
        if getattr(args, "keywords", None):
            keywords = args.keywords.split(",")
        report = run_eval(
            {
                "predictions": args.predictions,
                "references": args.references,
                "prediction_field": args.prediction_field,
                "reference_field": args.reference_field,
                "group_id_field": args.group_id_field,
                "label_field": args.label_field,
                "predicted_label_field": args.predicted_label_field,
                "keywords": keywords or [],
                "report_dir": args.report_dir,
            }
        )
        print(f"eval — {report['overall_status']}")
        print(report.get("disclaimer"))
        return status_to_exit(report["overall_status"])

    if handler == "doctor":
        report = run_doctor(
            model_path=Path(args.model_path) if args.model_path else None,
            expected_gpus=args.expected_gpus,
        )
        print("=" * 60)
        print(f"doctor — {report['overall_status']}")
        for item in report["checks"]:
            print(f"[{item['status']:4}] {item['name']}: {item['message']}")
        if args.json_output:
            write_json(Path(args.json_output), report, overwrite=True)
            print(f"JSON: {args.json_output}")
        return status_to_exit(report["overall_status"])

    if handler == "manifest":
        report = run_manifest(
            {
                "output_dir": args.output_dir,
                "framework": args.framework,
                "manifest_out": args.manifest_out,
                "expected_steps": args.expected_steps,
                "seed": args.seed,
            }
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return EXIT_OK

    if handler == "bundle_info":
        from .core.io_util import sha256_file
        from .core.optional import package_version

        self_path = Path(sys.argv[0]).resolve() if sys.argv and sys.argv[0] else Path.cwd() / "train_guard.py"
        digest = None
        try:
            if self_path.is_file():
                digest = sha256_file(self_path)
        except OSError:
            digest = None
        info = {
            "name": "Train Guard",
            "version": __version__,
            "min_python": f"{__min_python__[0]}.{__min_python__[1]}",
            "file": self_path.name,
            "sha256": digest,
            "commands": [
                "doctor",
                "data check",
                "data inventory",
                "data compare",
                "run watch",
                "run check",
                "run compare",
                "eval",
                "manifest",
                "bundle-info",
            ],
            "optional_dependencies": {
                "PyYAML": package_version("PyYAML"),
                "Pillow": package_version("Pillow"),
                "psutil": package_version("psutil"),
            },
            "deploy": "Copy release/train_guard.py to the target host and run: python train_guard.py doctor",
            "note": "Read-only by default; does not stop training processes.",
        }
        print(json.dumps(info, ensure_ascii=False, indent=2))
        return EXIT_OK

    print(f"Unknown handler: {handler}", file=sys.stderr)
    return EXIT_USAGE


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entrypoint."""
    if sys.version_info < __min_python__:
        print(
            f"Error: Python >= {__min_python__[0]}.{__min_python__[1]} required, "
            f"got {sys.version.split()[0]}",
            file=sys.stderr,
        )
        return EXIT_FAIL
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    setup_logging(verbose=bool(getattr(args, "verbose", False)))
    try:
        return int(_dispatch(args))
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logging.getLogger("train_guard").error("%s", exc)
        msg = str(exc).lower()
        if "config" in msg:
            return EXIT_CONFIG
        return EXIT_FAIL
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return EXIT_RUNTIME
    except Exception as exc:  # noqa: BLE001
        logging.getLogger("train_guard").error("Unhandled error: %s", exc)
        logging.getLogger("train_guard").debug(traceback.format_exc())
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_RUNTIME
