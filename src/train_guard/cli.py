"""Train Guard command-line interface."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import traceback
from pathlib import Path
from typing import Any, Optional, Sequence

from . import __min_python__, __version__
from .core.config import (
    ConfigError,
    TEMPLATE_NAMES,
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
)
from .core.io_util import write_json
from .data.commands import run_data_check, run_data_compare, run_data_inventory, write_data_reports
from .env.doctor import run_doctor, status_to_exit
from .eval.metrics import run_eval
from .report.html import render_html_report
from .run.commands import cmd_run_watch, run_manifest, run_run_check, run_run_compare


STATUS_HELP = (
    "Results: PASS=checks passed (exit 0), WARN=review recommended (exit 1), "
    "FAIL=check failed (exit 2). Usage=3, configuration=4, runtime=5, refused overwrite=6."
)


class TrainGuardArgumentParser(argparse.ArgumentParser):
    """Argparse with the same actionable error contract as config validation."""

    def error(self, message: str) -> None:
        self.print_usage(sys.stderr)
        self.exit(
            EXIT_USAGE,
            "FAIL\n"
            f"Problem: {message}\n"
            "Location: command line\n"
            "Fix: review the command help and correct the shown option\n",
        )


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def _warn_deprecated(old: str, new: str) -> None:
    print(f"[DEPRECATED] {old} → {new}; see docs/MIGRATION.md", file=sys.stderr)


def _configured(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        help="JSON config (YAML also works when PyYAML is installed)",
    )


def _add(
    parser: argparse.ArgumentParser,
    *flags: str,
    **kwargs: Any,
) -> None:
    kwargs.setdefault("default", argparse.SUPPRESS)
    parser.add_argument(*flags, **kwargs)


def _data_check_args(parser: argparse.ArgumentParser) -> None:
    _configured(parser)
    _add(parser, "--data-root")
    _add(parser, "--annotation")
    _add(parser, "--sample-limit", type=int)
    _add(parser, "--full-scan", action=argparse.BooleanOptionalAction)
    _add(parser, "--compute-hash", action=argparse.BooleanOptionalAction)
    _add(parser, "--verify-images", action=argparse.BooleanOptionalAction)
    _add(parser, "--group-id-field", dest="group_id")
    _add(parser, "--split-field", dest="split")
    _add(parser, "--media-field", dest="media")
    _add(parser, "--messages-field", dest="messages")
    _add(parser, "--report-dir")
    _add(parser, "--cache-db")


def _run_watch_args(parser: argparse.ArgumentParser) -> None:
    _configured(parser)
    _add(parser, "--once", action=argparse.BooleanOptionalAction)
    _add(parser, "--interval", type=int)
    _add(parser, "--pid", type=int)
    _add(parser, "--log-file")
    _add(parser, "--framework", choices=("generic", "huggingface", "transformers", "llamafactory"))
    _add(parser, "--output-dir")
    _add(parser, "--expected-gpus", type=int)
    _add(parser, "--stale-log-minutes", type=float)
    _add(parser, "--disk-free-gb-threshold", type=float)


def _run_check_args(parser: argparse.ArgumentParser) -> None:
    _configured(parser)
    _add(parser, "--output-dir")
    _add(parser, "--expected-steps", type=int)
    _add(parser, "--json-output")
    _add(parser, "--html-output")


def _run_compare_args(parser: argparse.ArgumentParser) -> None:
    _configured(parser)
    _add(parser, "--left")
    _add(parser, "--right")
    _add(parser, "--json-output")


def _eval_args(parser: argparse.ArgumentParser) -> None:
    _configured(parser)
    _add(parser, "--predictions")
    _add(parser, "--references")
    _add(parser, "--prediction-field")
    _add(parser, "--reference-field")
    _add(parser, "--group-id-field", dest="group_id")
    _add(parser, "--label-field")
    _add(parser, "--predicted-label-field")
    _add(parser, "--keywords", help="Comma-separated keywords")
    _add(parser, "--report-dir")


def build_parser() -> argparse.ArgumentParser:
    """Build the dependency-free argparse parser."""
    parser = TrainGuardArgumentParser(
        prog="train-guard",
        description="Read-only, domain-neutral LLM/VLM training quality checks.",
        epilog=STATUS_HELP,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init", help="Generate a validated starter configuration", epilog=STATUS_HELP)
    p.add_argument("--output", type=Path, default=Path("train-guard.json"))
    p.add_argument("--template", choices=TEMPLATE_NAMES, default="generic")
    p.add_argument("--force", action="store_true", help="Overwrite an existing output file")
    p.set_defaults(_handler="init")

    p = sub.add_parser("doctor", help="Environment and model integrity check", epilog=STATUS_HELP)
    _configured(p)
    _add(p, "--model-path")
    _add(p, "--expected-gpus", type=int)
    _add(p, "--json-output")
    p.set_defaults(_handler="doctor")

    p_data = sub.add_parser("data", help="Dataset commands")
    data_sub = p_data.add_subparsers(dest="data_command", required=True)
    p = data_sub.add_parser("check", help="Dataset/media/message integrity check", epilog=STATUS_HELP)
    _data_check_args(p)
    p.set_defaults(_handler="data_check")
    p = data_sub.add_parser("inventory", help="Streaming dataset inventory", epilog=STATUS_HELP)
    _configured(p)
    _add(p, "--annotation")
    _add(p, "--sample-limit", type=int)
    _add(p, "--group-id-field", dest="group_id")
    _add(p, "--report-dir")
    p.set_defaults(_handler="data_inventory")
    p = data_sub.add_parser("compare", help="Compare two annotation files", epilog=STATUS_HELP)
    _configured(p)
    _add(p, "--left")
    _add(p, "--right")
    _add(p, "--sample-limit", type=int)
    _add(p, "--group-id-field", dest="group_id")
    _add(p, "--report-dir")
    p.set_defaults(_handler="data_compare")

    p_run = sub.add_parser("run", help="Training run commands")
    run_sub = p_run.add_subparsers(dest="run_command", required=True)
    p = run_sub.add_parser("watch", help="Read-only training watch", epilog=STATUS_HELP)
    _run_watch_args(p)
    p.set_defaults(_handler="run_watch")
    p = run_sub.add_parser("check", help="Check whether a training run completed", epilog=STATUS_HELP)
    _run_check_args(p)
    p.set_defaults(_handler="run_check")
    p = run_sub.add_parser("compare", help="Compare two run output directories", epilog=STATUS_HELP)
    _run_compare_args(p)
    p.set_defaults(_handler="run_compare")

    p = sub.add_parser("eval", help="Evaluate predictions vs references", epilog=STATUS_HELP)
    _eval_args(p)
    p.set_defaults(_handler="eval")

    p = sub.add_parser("manifest", help="Write run manifest and experiment fingerprint", epilog=STATUS_HELP)
    _configured(p)
    _add(p, "--output-dir")
    _add(p, "--framework", choices=("generic", "huggingface", "transformers", "llamafactory"))
    _add(p, "--manifest-out")
    _add(p, "--expected-steps", type=int)
    _add(p, "--seed")
    p.set_defaults(_handler="manifest")

    p = sub.add_parser("bundle-info", help="Show deploy/version info")
    p.set_defaults(_handler="bundle_info")

    p = sub.add_parser("compare", help="Alias for 'run compare'", epilog=STATUS_HELP)
    _run_compare_args(p)
    p.set_defaults(_handler="run_compare")

    aliases = (
        ("precheck", "legacy_precheck", _data_check_args, "Deprecated alias for 'data check'"),
        ("monitor", "legacy_monitor", _run_watch_args, "Deprecated alias for 'run watch'"),
        ("postcheck", "legacy_postcheck", _run_check_args, "Deprecated alias for 'run check'"),
        ("evaluate", "legacy_evaluate", _eval_args, "Deprecated alias for 'eval'"),
    )
    for name, handler, add_args, help_text in aliases:
        p = sub.add_parser(name, help=help_text, epilog=STATUS_HELP)
        add_args(p)
        p.set_defaults(_handler=handler)
    return parser


HANDLER_SECTIONS = {
    "doctor": ("doctor",),
    "data_check": ("data", "check"),
    "legacy_precheck": ("data", "check"),
    "data_inventory": ("data", "inventory"),
    "data_compare": ("data", "compare"),
    "run_watch": ("run", "watch"),
    "legacy_monitor": ("run", "watch"),
    "run_check": ("run", "check"),
    "legacy_postcheck": ("run", "check"),
    "run_compare": ("run", "compare"),
    "eval": ("eval",),
    "legacy_evaluate": ("eval",),
    "manifest": ("manifest",),
}


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
        "group_id_field": getattr(args, "group_id", None),
        "split_field": getattr(args, "split", None),
        "media_field": getattr(args, "media", None),
        "messages_field": getattr(args, "messages", None),
        "cache_db": getattr(args, "cache_db", None),
    }


def _dispatch(args: argparse.Namespace) -> int:
    handler = getattr(args, "_handler", None)
    if handler == "init":
        write_config_template(args.output, args.template, args.force)
        print(f"PASS config written: {args.output}")
        print("Next: edit paths, then run `train-guard doctor --config <file>`.")
        return EXIT_OK

    if handler in {"legacy_precheck", "data_check"}:
        if handler == "legacy_precheck":
            _warn_deprecated("precheck", "data check")
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
        report = run_data_inventory({
            "annotation": args.annotation, "sample_limit": args.sample_limit,
            "group_id_field": args.group_id,
        })
        write_data_reports(report, Path(args.report_dir), "data_inventory_report")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return status_to_exit(report["overall_status"])

    if handler == "data_compare":
        report = run_data_compare({
            "left": args.left, "right": args.right, "sample_limit": args.sample_limit,
            "group_id_field": args.group_id,
        })
        write_data_reports(report, Path(args.report_dir), "data_compare_report")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return status_to_exit(report["overall_status"])

    if handler in {"legacy_monitor", "run_watch"}:
        if handler == "legacy_monitor":
            _warn_deprecated("monitor", "run watch")
        return cmd_run_watch({
            "once": args.once, "interval": args.interval, "pid": args.pid,
            "log_file": args.log_file, "framework": args.framework,
            "output_dir": args.output_dir, "expected_gpu_count": args.expected_gpus,
            "stale_log_minutes": args.stale_log_minutes,
            "disk_free_gb_threshold": args.disk_free_gb_threshold,
        })

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
        if args.html_output:
            html_doc = render_html_report(
                "Train Guard — Run Check",
                [{"title": "Status", "value": report["overall_status"], "status": report["overall_status"]}],
                [
                    {"title": "Checks", "headers": ["Status", "Name", "Message"], "rows": [[c["status"], c["name"], c["message"]] for c in report.get("checks") or []]},
                    {"title": "Reasons", "headers": ["Reason"], "rows": [[r] for r in report.get("reasons") or []]},
                ],
                report.get("disclaimer") or "",
            )
            path = Path(args.html_output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(html_doc, encoding="utf-8")
            print(f"HTML: {args.html_output}")
        return status_to_exit(report["overall_status"])

    if handler == "run_compare":
        report = run_run_compare(Path(args.left), Path(args.right))
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if args.json_output:
            write_json(Path(args.json_output), report, overwrite=True)
        return EXIT_OK

    if handler in {"legacy_evaluate", "eval"}:
        if handler == "legacy_evaluate":
            _warn_deprecated("evaluate", "eval")
        keywords = args.keywords
        if isinstance(keywords, str):
            keywords = [item.strip() for item in keywords.split(",") if item.strip()]
        report = run_eval({
            "predictions": args.predictions, "references": args.references,
            "prediction_field": args.prediction_field,
            "reference_field": args.reference_field,
            "group_id_field": args.group_id, "label_field": args.label_field,
            "predicted_label_field": args.predicted_label_field,
            "keywords": keywords, "report_dir": args.report_dir,
        })
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
        report = run_manifest({
            "output_dir": args.output_dir, "framework": args.framework,
            "manifest_out": args.manifest_out, "expected_steps": args.expected_steps,
            "seed": args.seed,
        })
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
            pass
        info = {
            "name": "Train Guard", "version": __version__,
            "min_python": f"{__min_python__[0]}.{__min_python__[1]}",
            "file": self_path.name, "sha256": digest,
            "commands": ["init", "doctor", "data check", "data inventory", "data compare",
                         "run watch", "run check", "run compare", "eval", "manifest", "bundle-info"],
            "optional_dependencies": {
                "PyYAML": package_version("PyYAML"),
                "Pillow": package_version("Pillow"),
                "psutil": package_version("psutil"),
            },
            "status_and_exit_codes": STATUS_HELP,
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
        print(f"FAIL\nProblem: {exc}\nLocation: command input\nFix: correct the path or value and retry", file=sys.stderr)
        return EXIT_FAIL
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return EXIT_RUNTIME
    except Exception as exc:  # noqa: BLE001
        logging.getLogger("train_guard").error("Unhandled error: %s", exc)
        logging.getLogger("train_guard").debug(traceback.format_exc())
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_RUNTIME
