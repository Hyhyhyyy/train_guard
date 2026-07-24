# Reliability and recovery — 0.6.0rc1

Train Guard observes by default. `run watch` collects local process, GPU, log, metric,
checkpoint, and disk evidence; deterministic rules produce versioned events and deduplicated
alerts. State is persisted in local SQLite and audit/export records use JSONL.

Watcher lifecycle and training lifecycle are separate. A snapshot, watcher shutdown, or
Ctrl-C is not completion evidence. Use `run check` after training.

## Outputs

Webhook delivery, Prometheus textfile output, and OTel JSONL output are opt-in. Delivery
failures must not mutate or stop training. Users are responsible for securing endpoints and
files and for deciding whether metadata may leave the machine.

The Web dashboard reads a state database:

```bash
train-guard show --state-db ./.train-guard/state.sqlite
```

It binds only to `127.0.0.1`, `::1`, or `localhost`. Observation is read-only by default.
With `--enable-control`, it prints an in-memory token once and accepts only local-origin,
allowlisted commands for processes registered by `run supervise --enable-control`. Never
expose it through a public proxy. Install `train-guard[tui]` and run `train-guard tui
--state-db PATH` for the SSH-friendly terminal view.

## Controlled recovery

`run supervise` is a separate, explicit policy boundary. Without `--restart`, it does not
restart. With restart enabled, it:

1. launches an argv directly, never a shell string;
2. waits for a non-zero process exit;
3. validates the selected checkpoint and required relative files;
4. enforces a persistent finite restart budget and time window;
5. records the decision and outcome in an audit log; and
6. relaunches the same argv.

The argv must include the framework's resume option. A valid checkpoint and a permitted
restart do not guarantee model correctness. Use conservative budgets and test the flow with
synthetic failures before production use.

## Rule interpretation

Events follow `symptom -> evidence -> root cause -> action -> outcome`. A root cause is
guidance derived from evidence, not proof. Review raw framework logs and infrastructure
telemetry before acting on destructive recommendations.
