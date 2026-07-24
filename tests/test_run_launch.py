from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from train_guard import cli
from train_guard.cli_parser import build_parser
from train_guard.run import launch
from train_guard.state import StateStore


def test_launch_parser_exposes_single_workflow_command(tmp_path: Path) -> None:
    args = build_parser().parse_args(
        [
            "run",
            "launch",
            "--output-dir",
            str(tmp_path),
            "--framework",
            "generic",
            "--",
            sys.executable,
            "-c",
            "pass",
        ]
    )
    assert args._handler == "run_launch"
    assert args.training_command[0] == "--"
    assert args.monitor_interval == 5


def test_launch_dispatch_maps_summary_status() -> None:
    summary = {"overall_status": "WARN", "run_id": "run-1"}
    with mock.patch.object(cli, "run_launch", return_value=summary) as run:
        code = cli._dispatch(SimpleNamespace(_handler="run_launch"))
    assert code == 1
    assert run.call_args.args[0]["_handler"] == "run_launch"


def test_strict_preflight_rejects_before_process_start(tmp_path: Path) -> None:
    doctor = {"overall_status": "FAIL", "checks": []}
    with (
        mock.patch.object(launch, "run_doctor", return_value=doctor),
        mock.patch.object(launch, "supervise") as supervise,
    ):
        summary = launch.run_launch(
            {
                "training_command": [sys.executable, "-c", "pass"],
                "output_dir": tmp_path,
                "strict_preflight": True,
            }
        )
    assert summary["overall_status"] == "FAIL"
    assert summary["phase"] == "preflight_rejected"
    assert not supervise.called
    assert (tmp_path / "train_guard_run_summary.json").is_file()


def test_launch_runs_monitored_training_and_writes_summary(tmp_path: Path) -> None:
    output = tmp_path / "run"
    code = (
        "import json,sys,time;"
        "from pathlib import Path;"
        "out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True);"
        "state={'global_step':1,'train_runtime':1.0,'train_loss':0.5,"
        "'train_samples_per_second':1.0,'log_history':[{'loss':0.5}]};"
        "(out/'trainer_state.json').write_text(json.dumps(state),encoding='utf-8');"
        "(out/'model.safetensors').write_bytes(b'weights');"
        "ckpt=out/'checkpoint-1';ckpt.mkdir(exist_ok=True);"
        "(ckpt/'state.json').write_text('{}',encoding='utf-8');"
        "time.sleep(0.2)"
    )
    with mock.patch.object(
        launch,
        "run_doctor",
        return_value={"overall_status": "PASS", "checks": []},
    ):
        summary = launch.run_launch(
            {
                "training_command": [sys.executable, "-c", code, str(output)],
                "output_dir": output,
                "framework": "generic",
                "training_type": "full",
                "expected_steps": 1,
                "monitor_interval": 1,
                "run_id": "launch-e2e",
            }
        )

    assert summary["overall_status"] == "PASS"
    assert summary["phase"] == "finished"
    assert summary["execution"]["status"] == "completed"
    assert summary["postcheck"]["overall_status"] == "PASS"
    assert summary["manifest"]["overall_status"] == "PASS"
    persisted = json.loads((output / "train_guard_run_summary.json").read_text(encoding="utf-8"))
    assert persisted["training_command"]["executable"] == Path(sys.executable).name
    assert "time.sleep" not in json.dumps(persisted)
    with StateStore(output / "train_guard_state.sqlite") as store:
        assert store.latest_sample("launch-e2e")
