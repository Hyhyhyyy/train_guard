from __future__ import annotations

import importlib.metadata
import runpy
import sys
import time
from pathlib import Path
from unittest import mock

import pytest

import train_guard
import train_guard.frameworks as frameworks
from train_guard import api
from train_guard.control import ControlRequest, ControlToken, bearer_token, origin_is_local
from train_guard.core import optional
from train_guard.core.config import ConfigError, resolve_inline_config
from train_guard.dashboard import DashboardOptions
from train_guard.state import StateStore
from train_guard.supervisor import FileHeartbeatProbe
from train_guard.tui import run_tui


def test_optional_dependency_absence_and_version_aliases() -> None:
    with mock.patch.dict(
        sys.modules,
        {
            "yaml": None,
            "PIL": None,
            "psutil": None,
            "torch": None,
        },
    ):
        assert optional.try_import_yaml() is None
        assert optional.try_import_pil() is None
        assert optional.try_import_psutil() is None
        assert optional.try_import_torch() is None

    missing = importlib.metadata.PackageNotFoundError()
    with mock.patch(
        "importlib.metadata.version",
        side_effect=[missing, missing, "1.2.3"],
    ):
        assert optional.package_version("pillow") == "1.2.3"
    with mock.patch("importlib.metadata.version", side_effect=RuntimeError("broken")):
        assert optional.package_version("unknown") is None


def test_lazy_public_exports_and_unknown_attributes() -> None:
    assert "check_run" in dir(train_guard)
    assert train_guard.check_run is api.check_run
    with pytest.raises(AttributeError):
        getattr(train_guard, "missing_export")
    assert "TrainGuardCallback" in dir(frameworks)
    assert frameworks.TrainGuardCallback.__name__ == "TrainGuardCallback"
    with pytest.raises(AttributeError):
        getattr(frameworks, "missing_export")


def test_cli_module_entrypoint_and_tui_dependency_error(tmp_path: Path) -> None:
    with mock.patch("train_guard.cli.main", return_value=7):
        with pytest.raises(SystemExit) as exit_info:
            runpy.run_module("train_guard.__main__", run_name="__main__")
    assert exit_info.value.code == 7

    with mock.patch.dict(sys.modules, {"textual.app": None}):
        with pytest.raises(RuntimeError, match="optional dependency"):
            run_tui(tmp_path / "state.sqlite")


def test_config_api_aliases_and_validation() -> None:
    with pytest.raises(ConfigError, match="unknown configuration field"):
        resolve_inline_config(("eval",), {"predictions": "p.jsonl", "typo": True})
    with pytest.raises(ConfigError, match="unsupported command section"):
        resolve_inline_config(("missing",), {})

    with mock.patch.object(api, "run_data_check", return_value={}) as check:
        api.check_dataset(
            {
                "annotation": "data.jsonl",
                "group_id_field": "batch",
                "messages_field": "turns",
            }
        )
    resolved = check.call_args.args[0]
    assert resolved["group_id_field"] == "batch"
    assert resolved["messages_field"] == "turns"


def test_state_and_control_validation_edges(tmp_path: Path) -> None:
    request = ControlRequest.create("run-1", "pause")
    assert request.command_id
    assert ControlToken().plain
    assert bearer_token(None) == ""
    assert not origin_is_local(None, "localhost", 8765)
    with pytest.raises(ValueError, match="run_id"):
        ControlRequest.create("", "pause")
    with pytest.raises(ValueError, match="ttl"):
        ControlRequest.create("run-1", "pause", ttl_seconds=0)

    with StateStore(tmp_path / "state.sqlite") as store:
        with pytest.raises(ValueError, match="retention"):
            store.record_sample("run-1", 1.0, {}, retention=0)
        with pytest.raises(ValueError, match="limit"):
            store.metric_series("run-1", limit=0)
        with pytest.raises(TypeError, match="to_dict"):
            store.enqueue_control(object())
        with pytest.raises(ValueError, match="outcome"):
            store.complete_control("missing", "unknown", {})


def test_dashboard_options_and_file_heartbeat_probe(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="localhost"):
        DashboardOptions(host="external")
    with pytest.raises(ValueError, match="port"):
        DashboardOptions(port=70000)
    with pytest.raises(ValueError, match="token"):
        DashboardOptions(enable_control=True)

    heartbeat = tmp_path / "heartbeat"
    probe = FileHeartbeatProbe(heartbeat, max_age_seconds=1)
    assert not probe.healthy()
    heartbeat.write_text("ok", encoding="utf-8")
    assert probe.healthy()
    old = time.time() - 5
    heartbeat.touch()
    import os

    os.utime(heartbeat, (old, old))
    assert not probe.healthy()
    with pytest.raises(ValueError, match="positive"):
        FileHeartbeatProbe(heartbeat, 0)
