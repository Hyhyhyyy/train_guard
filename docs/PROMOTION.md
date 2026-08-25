# Train Guard promotion playbook

## Positioning

**One sentence:** Train Guard is a local-first reliability layer that checks LLM/VLM training
inputs, observes runs, records structured incidents, validates checkpoints, and only performs
recovery when the operator explicitly enables a bounded policy.

**Primary audience:** engineers and researchers running Hugging Face, Transformers, or
LLaMAFactory training who need reproducible checks and local observability without adopting a
hosted platform.

**Do not claim:** autonomous production healing, clinical safety, guaranteed training
correctness, measured GPU-hour savings, or real-hardware accuracy without separate evidence.

## Proof points

- Zero required runtime dependencies for the core CLI.
- Linux, macOS, and Windows CI across Python 3.10–3.14.
- CLI, Python API, Hugging Face callback, loopback-only Web dashboard, and SSH-friendly TUI.
- Explicit safety boundary: observation by default; restart requires opt-in, a finite budget,
  and checkpoint validation.
- Public synthetic fault-injection benchmark with fixed seed, schema, evaluator, and limitations.
- Checksummed single-file artifact, wheel, sdist, SBOM, and privacy gates for releases.

## Launch sequence

1. Publish the GitHub prerelease and verify all install and quickstart commands from a clean
   environment.
2. Ask 5–10 LLM/VLM practitioners for private trial feedback before a broad launch.
3. Convert reproducible feedback into public Issues and ship one follow-up release.
4. Publish a technical article focused on one failure story and the evidence/recovery boundary.
5. Share the article and repository in relevant engineering communities; answer technical
   questions and record limitations openly.

## Channel-specific copy

### GitHub / developer communities

> I built Train Guard, a local-first reliability toolkit for LLM/VLM training. It checks data
> and environments, observes runs, records structured incidents, validates checkpoints, and
> keeps recovery explicitly opt-in and bounded. The core has zero required dependencies, and
> the repository includes a reproducible synthetic fault-injection benchmark. Feedback on real
> training workflows is especially welcome.

### Chinese technical communities

> 我做了一个本地优先的 LLM/VLM 训练可靠性工具 Train Guard：覆盖训练前检查、运行时观测、
> 结构化事件、检查点验收和显式受控恢复。核心零必需依赖，默认只观察、不上传遥测、不擅自
> 停止训练；仓库提供固定种子的合成故障 benchmark，欢迎用真实训练流程试用并反馈边界问题。

## Content ideas

- “一次错误 checkpoint 为什么不应该被自动恢复”——展示验证门禁和有限重启预算。
- “训练监控不等于训练控制”——解释默认只读、显式控制与本地数据边界。
- “如何用固定种子验证 15 类训练故障规则”——完整复现 benchmark，主动说明合成局限。
- “从训练日志到可审计事件”——展示症状、证据、根因、动作、结果的数据契约。

## Success metrics

Track real trial users, successful clean installs, external Issues, repeat users,
non-documentation PRs, release downloads, and time-to-first-response. Stars and followers are
outcomes, not targets.
