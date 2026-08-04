# Train Guard 0.6.0rc1

[![CI](https://github.com/Hyhyhyyy/train_guard/actions/workflows/ci.yml/badge.svg)](https://github.com/Hyhyhyyy/train_guard/actions/workflows/ci.yml)
[![Python 3.10–3.14](https://img.shields.io/badge/python-3.10--3.14-blue)](https://www.python.org/)
[![许可证：Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

[English](README.md)

Train Guard 是面向 LLM/VLM 训练的本地优先可靠性工具，提供检查、监控、告警、可视化和显式受控恢复。核心安装没有必选依赖。默认只观察：不会自动安装包、上传遥测、停止训练或修改训练数据。

## 功能概览

- **训练前：**检查环境、GPU、依赖、模型路径、数据、媒体文件和数据划分。
- **训练中：**观察生命周期、日志、Loss、GPU、磁盘、检查点和训练进程。
- **可靠性：**生成结构化事件、去重告警、持久状态，并执行有界受控恢复。
- **训练后：**验收输出、对比运行、评估结果，并生成 SHA256 清单。
- **使用界面：**CLI、Python API、Hugging Face 回调、本地 Web 面板和 SSH TUI。
- **交付方式：**核心零必选依赖，同时提供经过校验的单文件发行版。

## 安装

任选一种方式：

```bash
# 隔离 CLI（推荐）
pipx install train-guard==0.6.0rc1
uv tool install train-guard==0.6.0rc1

# 当前 Python 环境
python -m pip install train-guard==0.6.0rc1

# 源码安装
git clone https://github.com/Hyhyhyyy/train_guard.git
cd train_guard
python -m pip install -e .
```

可选 extras 为 `yaml`、`image`、`psutil`、`tui` 和 `all`：

```bash
python -m pip install "train-guard[all]==0.6.0rc1"
python -m pip install "train-guard[tui]==0.6.0rc1"
```

升级或卸载：

```bash
pipx upgrade train-guard              # 或：uv tool upgrade train-guard
python -m pip install --upgrade train-guard
pipx uninstall train-guard            # 或：uv tool uninstall train-guard
python -m pip uninstall train-guard
```

## 三分钟流程

```bash
# 1. 生成通用、Transformers 或 LLaMAFactory 配置。
train-guard init --template transformers --output train-guard.json

# 2. 替换占位路径，然后检查环境和数据。
train-guard doctor --config train-guard.json
train-guard data check --config train-guard.json

# 3. 观察训练并验证输出。
train-guard run watch --config train-guard.json
train-guard run check --config train-guard.json
train-guard manifest --config train-guard.json
```

需要逐条保存全部缺图、坏图和空答案时，增加 `--issues-jsonl`：

```bash
train-guard data check --config train-guard.json --issues-jsonl ./reports/data-issues.jsonl
```

该台账可能包含媒体引用；如果原始数据使用绝对路径，绝对路径也会写入。请作为敏感
本地文件保管，不要随公开报告发布。

源码目录和下载的单文件发行资产都可使用 `python train_guard.py ...`。

## 界面与恢复

- **CLI：**稳定接口；参见 [docs/CLI.md](docs/CLI.md)。
- **统一启动：**`train-guard run launch` 使用同一 run ID 串联环境预检、受监督训练、实时可靠性采集、有限恢复、训练后检查和清单生成。
- **Web：**`train-guard show --state-db PATH` 仅在回环地址展示指标、GPU 状态、告警、检查点、恢复历史和受监督进程。
- **TUI：**`train-guard tui --state-db PATH` 通过 SSH 提供同一持久状态视图。
- **状态快照：**`train-guard run status --state-db PATH` 输出适合脚本使用的单次状态。
- **受控恢复：**`run supervise` 只启动显式 argv；只有传入 `--restart`、检查点验证通过且未耗尽有限预算时才自动重启。它不会执行 shell 字符串。

新训练建议使用统一入口：

```bash
train-guard run launch --output-dir ./outputs/run-1 --framework huggingface \
  --expected-steps 100 -- python train.py --output_dir ./outputs/run-1
```

该命令会生成统一摘要、环境报告、监控记录、审计日志、训练后检查、运行清单和 SQLite 状态。
只有希望环境检查失败时阻止训练启动，才需要添加 `--strict-preflight`。框架恢复参数仍由训练 argv 显式提供。

仅需独立监督功能时可使用：

```bash
train-guard run supervise --restart --max-restarts 1 \
  --checkpoint-dir ./checkpoint-100 \
  --required-checkpoint-file trainer_state.json \
  -- python train.py --resume-from-checkpoint ./checkpoint-100
```

训练 argv 必须包含框架对应的恢复选项。

控制功能默认关闭。受监督训练与 Web 面板必须都显式传入 `--enable-control`，并使用同一状态数据库。面板只在启动时显示一次内存令牌，只接受该受监督进程声明支持的操作：

```bash
train-guard run supervise --enable-control --state-db ./guard.sqlite -- python train.py
train-guard show --enable-control --state-db ./guard.sqlite
```

支持的控制操作为暂停、恢复、优雅停止、终止和验证后重启。检查点创建依赖具体训练框架，
因此不作为通用控制操作。自动重启尝试和结果会与人工控制一起进入恢复历史。

控制接口拒绝非回环客户端、非本地来源、过期或重复命令、未受监督进程和不支持的能力。请勿通过公网代理暴露面板。

## 安全边界与退出码

报告会脱敏绝对路径、用户名、主机名和凭据特征值。公开 HTML/JSON 不应包含原始样本或媒体字节。可选包绝不自动安装；Webhook 和遥测导出均为主动选择。Train Guard 是辅助工具，不是安全沙箱、合规系统，也不保证训练结果正确。

稳定退出码：`0` PASS、`1` WARN、`2` FAIL、`3` 用法错误、`4` 配置错误、`5` 运行时错误、`6` 拒绝覆盖。

## 专题文档

- [CLI 参考](docs/CLI.md)
- [配置与优先级](docs/CONFIGURATION.md)
- [可靠性、Web、TUI 与恢复](docs/RELIABILITY.md)
- [发行工程与产物](docs/RELEASE.md)
- [迁移与一个候选周期的别名窗口](docs/MIGRATION.md)
- [架构](ARCHITECTURE.md)
- [贡献](CONTRIBUTING.md)、[安全](SECURITY.md)和[支持](SUPPORT.md)
- [变更日志](CHANGELOG.md)

采用 Apache License 2.0；参见 [LICENSE](LICENSE)。
