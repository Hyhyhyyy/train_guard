"""Train Guard command-line interface."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import traceback
from pathlib import Path
from typing import Any, Optional, Sequence

from . import __min_python__, __version__
from ._compat import LEGACY_HANDLERS, deprecation_message
from .cli_parser import HANDLER_SECTIONS, STATUS_HELP, build_parser
from .control import ControlToken
from .core.config import (
    ConfigError,
    resolve_command_config,
    write_config_template,
)
from .core.exitcodes import (
    EXIT_CONFIG,
    EXIT_FAIL,
    EXIT_OK,
    EXIT_REFUSED,
    EXIT_RUNTIME,
    EXIT_USAGE,
    EXIT_WARN,
)
from .core.io_util import atomic_write_text, write_json
from .data.commands import run_data_check, run_data_compare, run_data_inventory, write_data_reports
from .dashboard import serve as serve_dashboard
from .env.doctor import run_bundle_info, run_doctor, status_to_exit
from .eval.metrics import run_eval
from .report.html import render_html_report
from .run.commands import cmd_run_watch, run_manifest, run_run_check, run_run_compare
from .run.launch import run_launch
from .state import AuditLog, StateStore
from .status import build_status_snapshot
from .supervisor import (
    FileCheckpointValidator,
    FileHeartbeatProbe,
    ProcessSpec,
    RecoveryGuard,
    RecoveryPolicy,
    supervise,
)
from .tui import run_tui


def setup_logging(verbose: bool = False) -> None:
    """Configure CLI logging."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def _apply_config(args: argparse.Namespace) -> None:
    handler = getattr(args, "_handler", "")
    parts = HANDLER_SECTIONS.get(handler)
    if parts is None:
        return
    resolved = resolve_command_config(
        parts,
        getattr(args, "config", None),
        vars(args),
    )
    for key, value in resolved.items():
        setattr(args, key, value)


def _dataset_cfg(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "data_root": getattr(args, "data_root", None),
        "annotation": args.annotation,
        "sample_limit": args.sample_limit,
        "full_scan": getattr(args, "full_scan", False),
        "compute_hash": getattr(args, "compute_hash", False),
        "verify_images": getattr(args, "verify_images", True),
        "max_image_pixels": getattr(args, "max_image_pixels", None),
        "max_media_files": getattr(args, "max_media_files", None),
        "max_scan_bytes": getattr(args, "max_scan_bytes", None),
        "allow_external_media": getattr(args, "allow_external_media", False),
        "group_id_field": getattr(args, "group_id", None),
        "split_field": getattr(args, "split", None),
        "media_field": getattr(args, "media", None),
        "messages_field": getattr(args, "messages", None),
        "cache_db": getattr(args, "cache_db", None),
        "issues_jsonl": getattr(args, "issues_jsonl", None),
    }


def _dispatch(args: argparse.Namespace) -> int:
    handler = getattr(args, "_handler", None)
    legacy_command = getattr(args, "_legacy_command", None)
    if handler in LEGACY_HANDLERS:
        legacy_command, handler = LEGACY_HANDLERS[handler]
    if legacy_command is not None:
        print(deprecation_message(legacy_command), file=sys.stderr)
    if handler == "init":
        write_config_template(args.output, args.template, args.force)
        print(f"PASS config written: {args.output}")
        print("Next:")
        print(f"  1. Edit placeholder paths in {args.output}")
        print(f"  2. train-guard doctor --config {args.output}")
        print(f"  3. train-guard data check --config {args.output}")
        print(f"  4. After training: train-guard run check --config {args.output}")
        return EXIT_OK

    if handler == "data_check":
        report = run_data_check(_dataset_cfg(args))
        write_data_reports(report, Path(args.report_dir), "data_check_report")
        print(f"data check — {report['overall_status']}")
        print(
            f"scanned={report['stats'].get('scanned_samples')} "
            f"missing_media={report['stats'].get('missing_media')} "
            f"empty_answers={report['stats'].get('empty_answers')} "
            f"group_leak={report['stats'].get('group_leak_count')}"
        )
        return status_to_exit(report["overall_status"])

    if handler == "data_inventory":
        report = run_data_inventory(
            {
                "annotation": args.annotation,
                "sample_limit": args.sample_limit,
                "group_id_field": args.group_id,
            }
        )
        write_data_reports(report, Path(args.report_dir), "data_inventory_report")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return status_to_exit(report["overall_status"])

    if handler == "data_compare":
        report = run_data_compare(
            {
                "left": args.left,
                "right": args.right,
                "sample_limit": args.sample_limit,
                "group_id_field": args.group_id,
            }
        )
        write_data_reports(report, Path(args.report_dir), "data_compare_report")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return status_to_exit(report["overall_status"])

    if handler in {"run_watch", "run_snapshot"}:
        return cmd_run_watch(
            {
                "once": True if handler == "run_snapshot" else args.once,
                "interval": args.interval,
                "pid": args.pid,
                "log_file": args.log_file,
                "framework": args.framework,
                "output_dir": args.output_dir,
                "expected_gpu_count": args.expected_gpus,
                "stale_log_minutes": args.stale_log_minutes,
                "disk_free_gb_threshold": args.disk_free_gb_threshold,
                "run_id": args.run_id,
                "state_db": args.state_db,
                "webhook_url": args.webhook_url,
                "reliability": args.reliability,
                "prometheus_file": getattr(args, "prometheus_file", None),
                "otel_file": getattr(args, "otel_file", None),
                "notification_every": args.notification_every,
                "step_stall_seconds": args.step_stall_seconds,
                "gpu_overheat_celsius": args.gpu_overheat_celsius,
                "checkpoint_stale_seconds": args.checkpoint_stale_seconds,
            }
        )

    if handler == "run_check":
        report = run_run_check(
            Path(args.output_dir),
            expected_steps=args.expected_steps,
            framework=getattr(args, "framework", None) or "huggingface",
            training_type=getattr(args, "training_type", "auto"),
        )
        print("=" * 60)
        print(f"run check — {report['overall_status']}")
        for item in report.get("checks") or []:
            print(f"[{item['status']:4}] {item['name']}: {item['message']}")
        for reason in report.get("reasons") or []:
            print(f"  - {reason}")
        if args.json_output:
            write_json(Path(args.json_output), report, overwrite=True)
            print(f"JSON: {args.json_output}")
        if args.html_output:
            html_doc = render_html_report(
                "Train Guard — Run Check",
                [
                    {
                        "title": "Status",
                        "value": report["overall_status"],
                        "status": report["overall_status"],
                    }
                ],
                [
                    {
                        "title": "Checks",
                        "headers": ["Status", "Name", "Message"],
                        "rows": [
                            [c["status"], c["name"], c["message"]]
                            for c in report.get("checks") or []
                        ],
                    },
                    {
                        "title": "Reasons",
                        "headers": ["Reason"],
                        "rows": [[r] for r in report.get("reasons") or []],
                    },
                ],
                report.get("disclaimer") or "",
            )
            path = Path(args.html_output)
            atomic_write_text(path, html_doc, overwrite=True)
            print(f"HTML: {args.html_output}")
        return status_to_exit(report["overall_status"])

    if handler == "run_compare":
        report = run_run_compare(
            Path(args.left),
            Path(args.right),
            framework=getattr(args, "framework", None) or "huggingface",
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if args.json_output:
            write_json(Path(args.json_output), report, overwrite=True)
        return status_to_exit(report["overall_status"])

    if handler == "run_status":
        with StateStore(args.state_db) as store:
            snapshot = build_status_snapshot(store, args.run_id).to_dict()
        print(json.dumps(snapshot, ensure_ascii=False, indent=2, allow_nan=False))
        if args.json_output:
            write_json(args.json_output, snapshot, overwrite=True)
        severities = {
            str((alert.get("event") or {}).get("severity"))
            for alert in snapshot.get("active_alerts", ())
            if isinstance(alert, dict)
        }
        if snapshot.get("phase") in {"aborted", "watch_error"} or severities.intersection(
            {"error", "critical"}
        ):
            return EXIT_FAIL
        return EXIT_WARN if "warning" in severities else EXIT_OK

    if handler == "run_launch":
        summary = run_launch(vars(args))
        print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
        return status_to_exit(str(summary.get("overall_status") or "FAIL"))

    if handler == "run_supervise":
        command = list(args.training_command)
        if command and command[0] == "--":
            command = command[1:]
        if not command:
            raise ValueError("a training command is required after '--'")
        if args.max_restarts < 0 or args.restart_window_seconds <= 0:
            raise ValueError("restart limits must be non-negative and window positive")
        if (
            getattr(args, "health_max_age", 30.0) <= 0
            or getattr(args, "health_timeout", 120.0) <= 0
            or getattr(args, "health_interval", 2.0) <= 0
        ):
            raise ValueError("health probe intervals must be positive")
        if args.restart and (
            args.max_restarts < 1
            or args.checkpoint_dir is None
            or not args.required_checkpoint_file
        ):
            raise ValueError(
                "--restart requires --max-restarts >= 1, --checkpoint-dir, "
                "and at least one --required-checkpoint-file"
            )
        audit_log = AuditLog(args.audit_log)
        state_db = getattr(args, "state_db", None) or args.audit_log.with_suffix(".sqlite")
        run_id = getattr(args, "run_id", None) or (
            "supervise-" + hashlib.sha256("\0".join(command).encode("utf-8")).hexdigest()[:12]
        )
        with StateStore(state_db) as recovery_store:
            stored_times = recovery_store.get_run_state(run_id, "supervisor.restart_times", [])
            restart_times = (
                [float(value) for value in stored_times] if isinstance(stored_times, list) else []
            )
            recovery_guard = RecoveryGuard(
                RecoveryPolicy(
                    max_restarts=args.max_restarts,
                    window_seconds=args.restart_window_seconds,
                    probe_timeout_seconds=getattr(args, "health_timeout", 120.0),
                    probe_interval_seconds=getattr(args, "health_interval", 2.0),
                ),
                restart_times=restart_times,
                on_change=lambda values: recovery_store.set_run_state(
                    run_id, "supervisor.restart_times", list(values)
                ),
            )
            result = supervise(
                ProcessSpec(command[0], tuple(command[1:])),
                restart_enabled=bool(args.restart),
                recovery_guard=recovery_guard,
                checkpoint_path=args.checkpoint_dir,
                checkpoint_validator=(
                    FileCheckpointValidator(args.required_checkpoint_file)
                    if args.required_checkpoint_file
                    else None
                ),
                health_probe=(
                    FileHeartbeatProbe(
                        args.health_file,
                        getattr(args, "health_max_age", 30.0),
                    )
                    if getattr(args, "health_file", None)
                    else None
                ),
                audit=audit_log.append,
                run_id=run_id,
                control_store=recovery_store,
                control_enabled=bool(getattr(args, "enable_control", False)),
            )
        print(
            json.dumps(
                {
                    "exit_code": result.exit_code,
                    "restart_count": result.restart_count,
                    "stopped_reason": result.stopped_reason,
                    "checkpoint_errors": list(result.checkpoint_errors),
                    "audit_log": args.audit_log.name,
                    "state_db": state_db.name,
                    "run_id": run_id,
                    "control_enabled": bool(getattr(args, "enable_control", False)),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return (
            EXIT_OK
            if result.exit_code == 0 or result.stopped_reason.startswith("control_")
            else EXIT_FAIL
        )

    if handler == "eval":
        keywords = args.keywords
        if isinstance(keywords, str):
            keywords = [item.strip() for item in keywords.split(",") if item.strip()]
        report = run_eval(
            {
                "predictions": args.predictions,
                "references": args.references,
                "prediction_field": args.prediction_field,
                "reference_field": args.reference_field,
                "group_id_field": args.group_id,
                "label_field": args.label_field,
                "predicted_label_field": args.predicted_label_field,
                "keywords": keywords,
                "sample_limit": args.sample_limit,
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
        return status_to_exit(report["overall_status"])

    if handler == "bundle_info":
        self_path = (
            Path(sys.argv[0]).resolve()
            if sys.argv and sys.argv[0]
            else Path.cwd() / "train_guard.py"
        )
        info = run_bundle_info(self_path)
        info["status_and_exit_codes"] = STATUS_HELP
        print(json.dumps(info, ensure_ascii=False, indent=2))
        return EXIT_OK
    if handler == "show":
        if args.port < 1 or args.port > 65535:
            raise ValueError("port must be between 1 and 65535")
        enable_control = bool(getattr(args, "enable_control", False))
        authorization = ControlToken() if enable_control else None
        with StateStore(args.state_db) as store:
            print(f"Dashboard: http://{args.host}:{args.port}")
            if authorization is not None:
                print(f"One-time local authorization: {authorization.plain}")
            if enable_control:
                serve_dashboard(
                    store,
                    args.host,
                    args.port,
                    enable_control=True,
                    authorization=authorization,
                )
            else:
                serve_dashboard(store, args.host, args.port)
        return EXIT_OK
    if handler == "tui":
        run_tui(
            args.state_db,
            run_id=args.run_id,
            enable_control=bool(args.enable_control),
        )
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
        _apply_config(args)
        return int(_dispatch(args))
    except ConfigError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_CONFIG
    except FileExistsError as exc:
        print(
            "Initialization refused\n"
            f"Problem: output file already exists\n"
            f"Location: {exc}\n"
            "Fix: choose another --output path or pass --force to overwrite it",
            file=sys.stderr,
        )
        return EXIT_REFUSED
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(
            f"FAIL\nProblem: {exc}\nLocation: command input\nFix: correct the path or value and retry",
            file=sys.stderr,
        )
        return EXIT_FAIL
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return EXIT_RUNTIME
    except Exception as exc:  # noqa: BLE001
        logging.getLogger("train_guard").error("Unhandled error: %s", exc)
        logging.getLogger("train_guard").debug(traceback.format_exc())
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_RUNTIME
