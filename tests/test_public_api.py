from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from train_guard import api


def test_public_function_wrappers_copy_inputs(tmp_path: Path) -> None:
    dataset_cfg = {"annotation": "records.jsonl"}
    eval_cfg = {"predictions": "predictions.jsonl"}
    watch_cfg = {"once": True}
    state: dict[str, object] = {"offset": 3}

    with (
        mock.patch.object(api, "run_data_check", return_value={"kind": "data"}) as data_check,
        mock.patch.object(api, "run_eval", return_value={"kind": "eval"}) as run_eval,
        mock.patch.object(api, "collect_watch_sample", return_value={"kind": "watch"}) as watch,
        mock.patch.object(api, "run_run_check", return_value={"kind": "run"}) as run_check,
    ):
        assert api.check_dataset(dataset_cfg) == {"kind": "data"}
        assert api.evaluate_predictions(eval_cfg) == {"kind": "eval"}
        assert api.watch_snapshot(watch_cfg, state) == {"kind": "watch"}
        assert api.watch_snapshot(watch_cfg) == {"kind": "watch"}
        assert api.check_run(
            tmp_path,
            expected_steps=7,
            framework="generic",
            training_type="full",
        ) == {"kind": "run"}

    resolved_dataset = data_check.call_args.args[0]
    assert resolved_dataset["annotation"] == "records.jsonl"
    assert resolved_dataset["group_id_field"] == "group_id"
    resolved_eval = run_eval.call_args.args[0]
    assert resolved_eval["predictions"] == "predictions.jsonl"
    assert resolved_eval["group_id_field"] == "group_id"
    assert [call.args[0]["once"] for call in watch.call_args_list] == [True, True]
    assert [call.args[0]["expected_gpu_count"] for call in watch.call_args_list] == [None, None]
    assert watch.call_args_list[0].args[1] is state
    assert watch.call_args_list[1].args[1] == {}
    assert dataset_cfg == {"annotation": "records.jsonl"}
    assert eval_cfg == {"predictions": "predictions.jsonl"}
    assert watch_cfg == {"once": True}
    run_check.assert_called_once_with(
        tmp_path,
        expected_steps=7,
        framework="generic",
        training_type="full",
    )


def test_reliability_session_context_and_closed_guard(tmp_path: Path) -> None:
    with api.ReliabilitySession(
        run_id="public-api-run",
        state_dir=tmp_path / "state",
        notification_every=1,
    ) as session:
        result = session.observe(
            {"loss": 1.0, "step": 1},
            timestamp=123.0,
            source="test",
        )
        assert result.events == ()
        session.close()
        session.close()

    assert (tmp_path / "state" / "train_guard_state.sqlite").exists()
    with pytest.raises(RuntimeError, match="closed"):
        session.observe({"step": 2})


def test_public_exports_are_stable() -> None:
    assert api.__all__ == [
        "ReliabilitySession",
        "check_dataset",
        "check_run",
        "evaluate_predictions",
        "watch_snapshot",
    ]
