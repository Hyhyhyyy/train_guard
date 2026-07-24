from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from train_guard import cli
from train_guard.core.config import ConfigError
from train_guard.core.exitcodes import (
    EXIT_CONFIG,
    EXIT_FAIL,
    EXIT_OK,
    EXIT_REFUSED,
    EXIT_RUNTIME,
    EXIT_USAGE,
)


def ns(handler: str, **values: object) -> Namespace:
    defaults: dict[str, object] = {
        "_handler": handler,
        "annotation": "items.jsonl",
        "sample_limit": None,
        "group_id": "group_id",
        "report_dir": "reports",
        "left": "left",
        "right": "right",
        "framework": "generic",
        "json_output": None,
        "html_output": None,
    }
    defaults.update(values)
    return Namespace(**defaults)


@pytest.mark.parametrize(
    ("handler", "runner_name", "runner_result"),
    [
        ("data_inventory", "run_data_inventory", {"overall_status": "PASS", "count": 1}),
        ("data_compare", "run_data_compare", {"overall_status": "WARN", "count": 2}),
    ],
)
def test_dispatch_data_commands(
    tmp_path: Path, handler: str, runner_name: str, runner_result: dict[str, object]
) -> None:
    args = ns(handler, report_dir=str(tmp_path))
    with (
        mock.patch.object(cli, runner_name, return_value=runner_result) as runner,
        mock.patch.object(cli, "write_data_reports") as write_reports,
    ):
        code = cli._dispatch(args)
    assert code in {EXIT_OK, 1}
    runner.assert_called_once()
    write_reports.assert_called_once()


def test_dispatch_watch_snapshot_and_deprecated_alias(capsys: pytest.CaptureFixture[str]) -> None:
    values = {
        "once": False,
        "interval": 2,
        "pid": None,
        "log_file": None,
        "output_dir": "out",
        "expected_gpus": 0,
        "stale_log_minutes": 1.0,
        "disk_free_gb_threshold": 2.0,
        "run_id": "r",
        "state_db": "state.db",
        "webhook_url": None,
        "reliability": False,
        "notification_every": 3,
        "step_stall_seconds": 4.0,
        "gpu_overheat_celsius": 80.0,
        "checkpoint_stale_seconds": 5.0,
    }
    with mock.patch.object(cli, "cmd_run_watch", return_value=1) as watch:
        assert cli._dispatch(ns("run_snapshot", **values)) == 1
        assert watch.call_args.args[0]["once"] is True
        assert cli._dispatch(ns("legacy_monitor", **values)) == 1
    assert "DEPRECATED" in capsys.readouterr().err


def test_dispatch_run_check_writes_optional_reports(tmp_path: Path) -> None:
    report = {
        "overall_status": "WARN",
        "checks": [{"status": "WARN", "name": "weights", "message": "missing"}],
        "reasons": ["missing"],
        "disclaimer": "synthetic",
    }
    json_path = tmp_path / "result.json"
    html_path = tmp_path / "result.html"
    args = ns(
        "legacy_postcheck",
        output_dir=str(tmp_path),
        expected_steps=2,
        training_type="full",
        json_output=str(json_path),
        html_output=str(html_path),
    )
    with mock.patch.object(cli, "run_run_check", return_value=report):
        assert cli._dispatch(args) == 1
    assert json.loads(json_path.read_text(encoding="utf-8"))["overall_status"] == "WARN"
    assert "Run Check" in html_path.read_text(encoding="utf-8")


def test_dispatch_compare_eval_doctor_manifest_and_bundle(tmp_path: Path) -> None:
    comparison = {"overall_status": "PASS", "value": 1}
    eval_report = {"overall_status": "PASS", "disclaimer": "metrics only"}
    doctor_report = {
        "overall_status": "PASS",
        "checks": [{"status": "PASS", "name": "python", "message": "ok"}],
    }
    manifest = {"overall_status": "PASS", "fingerprint": "x"}
    compare_out = tmp_path / "compare.json"
    doctor_out = tmp_path / "doctor.json"
    with (
        mock.patch.object(cli, "run_run_compare", return_value=comparison),
        mock.patch.object(cli, "run_eval", return_value=eval_report) as evaluate,
        mock.patch.object(cli, "run_doctor", return_value=doctor_report),
        mock.patch.object(cli, "run_manifest", return_value=manifest),
        mock.patch.object(cli, "run_bundle_info", return_value={"version": "test"}),
    ):
        assert cli._dispatch(ns("run_compare", json_output=str(compare_out))) == EXIT_OK
        assert (
            cli._dispatch(
                ns(
                    "legacy_evaluate",
                    predictions="p",
                    references="r",
                    prediction_field="prediction",
                    reference_field="reference",
                    label_field="label",
                    predicted_label_field="predicted_label",
                    keywords=" alpha, ,beta ",
                )
            )
            == EXIT_OK
        )
        assert evaluate.call_args.args[0]["keywords"] == ["alpha", "beta"]
        assert (
            cli._dispatch(
                ns("doctor", model_path=None, expected_gpus=0, json_output=str(doctor_out))
            )
            == EXIT_OK
        )
        assert (
            cli._dispatch(
                ns(
                    "manifest",
                    output_dir=str(tmp_path),
                    manifest_out=str(tmp_path / "manifest.json"),
                    expected_steps=1,
                    seed=3,
                )
            )
            == EXIT_OK
        )
        assert cli._dispatch(ns("bundle_info")) == EXIT_OK
    assert compare_out.exists()
    assert doctor_out.exists()


def test_dispatch_supervise_validation_and_success(tmp_path: Path) -> None:
    base: dict[str, object] = {
        "restart": False,
        "max_restarts": 0,
        "restart_window_seconds": 10.0,
        "checkpoint_dir": None,
        "required_checkpoint_file": [],
        "audit_log": tmp_path / "audit.jsonl",
    }
    with pytest.raises(ValueError, match="training command"):
        cli._dispatch(ns("run_supervise", training_command=[], **base))
    with pytest.raises(ValueError, match="restart limits"):
        cli._dispatch(
            ns(
                "run_supervise",
                training_command=["python", "train.py"],
                **{**base, "max_restarts": -1},
            )
        )
    with pytest.raises(ValueError, match="--restart requires"):
        cli._dispatch(ns("run_supervise", training_command=["python"], **{**base, "restart": True}))

    result = SimpleNamespace(
        exit_code=0,
        restart_count=1,
        stopped_reason="completed",
        checkpoint_errors=(),
    )
    with mock.patch.object(cli, "supervise", return_value=result) as supervise:
        code = cli._dispatch(
            ns("run_supervise", training_command=["--", "python", "train.py"], **base)
        )
    assert code == EXIT_OK
    assert supervise.call_args.args[0].command == ("python", "train.py")


def test_dispatch_show_and_unknown(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="port"):
        cli._dispatch(ns("show", state_db=tmp_path / "state.db", host="localhost", port=0))
    fake_store = mock.MagicMock()
    fake_store.__enter__.return_value = fake_store
    with (
        mock.patch.object(cli, "StateStore", return_value=fake_store),
        mock.patch.object(cli, "serve_dashboard") as serve,
    ):
        assert (
            cli._dispatch(ns("show", state_db=tmp_path / "state.db", host="127.0.0.1", port=8765))
            == EXIT_OK
        )
    serve.assert_called_once_with(fake_store, "127.0.0.1", 8765)
    assert cli._dispatch(ns("does-not-exist")) == EXIT_USAGE


@pytest.mark.parametrize(
    ("exception", "expected"),
    [
        (ConfigError("bad config", "test", "fix"), EXIT_CONFIG),
        (FileExistsError("existing"), EXIT_REFUSED),
        (FileNotFoundError("missing"), EXIT_FAIL),
        (KeyboardInterrupt(), EXIT_RUNTIME),
        (LookupError("unexpected"), EXIT_RUNTIME),
    ],
)
def test_main_maps_expected_failures(exception: BaseException, expected: int) -> None:
    with mock.patch.object(cli, "_dispatch", side_effect=exception):
        assert cli.main(["bundle-info"]) == expected


def test_parser_error_and_old_python(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        cli.main(["not-a-command"])
    assert caught.value.code == EXIT_USAGE
    assert "Problem:" in capsys.readouterr().err

    with mock.patch.object(cli.sys, "version_info", (3, 9)):
        assert cli.main(["bundle-info"]) == EXIT_FAIL
