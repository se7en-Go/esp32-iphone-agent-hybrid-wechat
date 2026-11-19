# esp32-iphone-agent-hybrid-wechat

独立的微信业务操作仓库，通过底座服务的 CLI（`/Users/seven/Documents/esp32-iphone-agent-hybrid-new/server/.venv/bin/iphone-agent`）与 `HIDAGENT_SKILLS` 挂载机制驱动 iPhone 真机微信操作。

## 架构约束

- **零代码拷贝**：本仓库不复制、不导入底座仓库源码，仅通过受管子进程调用底座 `iphone-agent` CLI 并遵守 `PI_CLI.md` 接口契约。
- **不修改底座**：不改动底座仓库代码，也不在未经确认的情况下改动底座的 `server/.local-data/runtime.env`。
- **全动态定位**：不设置任何静态 `layout.json`；聊天标题、桌面图标本体、底部输入框、键盘蓝色「发送」按钮与气泡均由本仓库独立 OCR + Pillow 像素几何从当前最新帧动态识别。

## 目录结构

```text
├── config/
│   ├── wechat-send.example.yaml     # M1 通用发消息配置模板
│   ├── group-monitor.example.yaml   # M2 群监控配置模板（默认 observe-only）
│   └── rules.example.yaml           # M2 确定性 YAML 规则模板
├── skills/                          # 通过 HIDAGENT_SKILLS 优先挂载到底座的 Skill 定义
│   ├── wechat-send/                 # M1 通用微信发消息规范
│   └── wechat-group-reply/          # M2 群监控与自动回复规范（优先覆盖底座占位）
├── scripts/
│   └── check-mount.sh               # 只读检查底座 8890 状态、staged-v1 与 HIDAGENT_SKILLS 挂载
├── src/wechat_agent/
│   ├── adapters/                    # 唯一底座 CLI 适配层
│   ├── vision/                      # 本仓独立 OCR（RapidOCR + Apple Vision 协议）与微信界面几何解析
│   ├── guards/                      # 目标白名单、Skill 熔断限流、M2 双层频控与违禁词守卫
│   ├── workflows/                   # M1 紧连点击链、微信导航、发消息安全状态机
│   ├── monitor/                     # M2 群聊提取、SQLite 跨重启去重、两级决策/生成框架、监控循环
│   │   ├── decision/                # Layer 1: DecisionGate (rule / llm / hybrid / jev 扩展槽)
│   │   └── generators/              # Layer 2: ReplyGenerator (rule_template / llm / hybrid)
│   └── cli.py                       # `wechat-agent` CLI（doctor / verify-skills / send / monitor）
└── tests/                           # 完整离线单元测试与多帧 Mock 回放测试
```

## 安装与环境准备

本仓库使用独立的 `.venv`（推荐 Python 3.12 + `rapidocr-onnxruntime`）：

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
```

## 挂载 `HIDAGENT_SKILLS` 到底座（只读检查与说明）

在底座服务侧配置环境变量指向本仓 `skills/` 目录，然后重启底座 worker：

```sh
export HIDAGENT_SKILLS=/Users/seven/Documents/esp32-iphone-agent-hybrid-wechat/skills
```

随后在本仓执行只读检查脚本或 `doctor` 验证是否生效：

```sh
./scripts/check-mount.sh
.venv/bin/wechat-agent verify-skills
.venv/bin/wechat-agent doctor
```

## M1 CLI 用法（通用发消息）

1. **默认 `--dry-run` 模式（只贴不发，核对正文后安全停在发送前）**：

```sh
.venv/bin/wechat-agent send --target "文件传输助手" --text "测试消息：仅粘贴不发送"
```

2. **显式 `--send` 模式（获明确授权后执行单次受守卫发送并见新绿色气泡确认）**：

```sh
.venv/bin/wechat-agent send --target "文件传输助手" --text "测试消息：真机验收" --send
```

## M2 CLI 用法（指定群监控与两级混合自动回复）

M2 采用两级架构：**Layer 1 `DecisionGate`**（输出统一结构化决策 `should_reply / intent / confidence / risk_score / matched_rule / reason`，预留 `jev` 扩展槽位兼容 `Noul / Choice / Score` 语义）+ **Layer 2 `ReplyGenerator`**（YAML 规则模板 + 可选 OpenAI 兼容 LLM 生成器，经 `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL` 启用）。

- **默认 `observe-only` 只看不发**（仅记录提取消息、意图判定与拟回复文案到 `.local-data/state/monitor_audit.jsonl`）：

```sh
.venv/bin/wechat-agent monitor --config config/group-monitor.local.yaml --once
```

- **安全红线**：
  - 对任何真实微信群，`observe-only` 和 `dry-run` 都**只做只读观察与审计日志记录，绝不聚焦输入框或粘贴草稿**。
  - 只有显式传 `--mode live --allow-live-send` 且配置了 `target_groups` 与非 `observe_only` 决策器时才会通过 M1 安全状态机发出回复。
  - **长时挂机注意事项**：如需长时间无人值守驻留监控，请提前将 iPhone「设置 → 显示与亮度 → 自动锁定」设为 **「永不」**；监控循环遇到锁屏、离开目标群或非微信界面会自动安全暂停，绝不盲目点击。

## 隐私与证据留存

- 运行截图与回执保存在 `.local-data/runs/<run_id>/`（目录 `0700`、文件 `0600`），M2 SQLite WAL 状态库与审计日志保存在 `.local-data/state/`（`0700` / `0600`），严禁写入 `/tmp`。
- 导出日志自动对 `transaction_id` 及敏感字段做脱敏处理；`.local-data/` 已被 `.gitignore` 忽略。
