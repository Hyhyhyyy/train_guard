"""Optional Textual terminal dashboard for SSH-friendly monitoring."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

from .control import CONTROL_ACTIONS, ControlRequest
from .state import StateStore
from .status import build_status_snapshot


def render_terminal_snapshot(snapshot: Mapping[str, Any]) -> str:
    """Render a deterministic plain-text fallback and test representation."""
    latest = dict(snapshot.get("latest_sample") or {})
    metrics = dict(latest.get("metrics") or {})
    alerts = list(snapshot.get("active_alerts") or [])
    process = dict(snapshot.get("managed_process") or {})
    lines = [
        "TRAIN GUARD",
        f"run={snapshot.get('run_id') or '-'} phase={snapshot.get('phase') or 'unknown'}",
        (
            f"step={metrics.get('step', latest.get('global_step', '-'))} "
            f"loss={metrics.get('loss', '-')} "
            f"throughput={metrics.get('throughput', '-')}"
        ),
        f"alerts={len(alerts)} process={process.get('status', 'unmanaged')}",
    ]
    for alert in alerts[:8]:
        event = dict(alert.get("event") or {})
        lines.append(
            f"[{str(event.get('severity', 'warning')).upper()}] "
            f"{event.get('kind', 'unknown')}: {event.get('message', '')}"
        )
    return "\n".join(lines)


def run_tui(
    state_db: Path,
    *,
    run_id: Optional[str] = None,
    enable_control: bool = False,
) -> None:
    """Run the optional interactive dashboard without burdening core installs."""
    try:
        from textual.app import App, ComposeResult
        from textual.binding import Binding
        from textual.containers import Horizontal, VerticalScroll
        from textual.widgets import Footer, Header, Static
    except ImportError as exc:
        raise RuntimeError(
            'terminal UI requires the optional dependency: pip install "train-guard[tui]"'
        ) from exc

    store = StateStore(state_db)

    class TrainGuardTui(App[None]):
        CSS = """
        Screen { background: #090d12; }
        #summary { border: round #2c4157; padding: 1 2; height: 8; }
        #alerts { border: round #70404a; padding: 1 2; }
        #help { color: #8ea0b5; height: 3; padding: 1 2; }
        """
        BINDINGS = [
            Binding("q", "quit", "Quit"),
            Binding("p", "control('pause')", "Pause"),
            Binding("r", "control('resume')", "Resume"),
            Binding("s", "control('graceful_stop')", "Stop"),
            Binding("x", "control('terminate')", "Terminate"),
            Binding("v", "control('validated_restart')", "Restart"),
        ]

        def compose(self) -> ComposeResult:
            yield Header()
            with Horizontal():
                yield Static(id="summary")
            with VerticalScroll(id="alerts"):
                yield Static("Loading...", id="content")
            yield Static(
                "Control keys are active only for supervised runs started with --enable-control.",
                id="help",
            )
            yield Footer()

        def on_mount(self) -> None:
            self.set_interval(1.0, self.refresh_status)
            self.refresh_status()

        def refresh_status(self) -> None:
            snapshot = build_status_snapshot(
                store,
                run_id,
                control_enabled=enable_control,
            ).to_dict()
            lines = render_terminal_snapshot(snapshot).splitlines()
            self.query_one("#summary", Static).update("\n".join(lines[:4]))
            self.query_one("#content", Static).update("\n".join(lines[4:]) or "No active alerts")

        def action_control(self, action: str) -> None:
            if not enable_control or action not in CONTROL_ACTIONS:
                self.notify("Control mode is disabled", severity="warning")
                return
            snapshot = build_status_snapshot(store, run_id, control_enabled=True)
            if not snapshot.run_id or not snapshot.managed_process:
                self.notify("No managed process for this run", severity="warning")
                return
            enqueue = getattr(store, "enqueue_control", None)
            if enqueue is None:
                self.notify("Control queue is unavailable", severity="error")
                return
            try:
                accepted = enqueue(ControlRequest.create(snapshot.run_id, action))
            except (TypeError, ValueError):
                self.notify("Action is unavailable for this process", severity="warning")
                return
            self.notify("Command queued" if accepted else "Command already queued")

        def on_unmount(self) -> None:
            store.close()

    try:
        TrainGuardTui().run()
    finally:
        try:
            store.close()
        except Exception:
            pass


__all__ = ["render_terminal_snapshot", "run_tui"]
