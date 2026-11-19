# esp32-iphone-agent-hybrid-wechat

独立的微信业务操作仓库，通过底座服务的 CLI（`/Users/seven/Documents/esp32-iphone-agent-hybrid-new/server/.venv/bin/iphone-agent`）与 `HIDAGENT_SKILLS` 挂载机制驱动 iPhone 真机微信操作。

## 架构约束

- **零代码拷贝**：本仓库不复制、不导入底座仓库源码，仅通过受管子进程调用底座 `iphone-agent` CLI 并遵守 `PI_CLI.md` 接口契约。
- **不修改底座**：不改动底座仓库代码，也不在未经确认的情况下改动底座的 `server/.local-data/runtime.env`。
- **全动态定位**：不设置任何静态 `layout.json`；聊天标题、桌面图标本体、底部输入框、键盘蓝色「发送」按钮与气泡均由本仓库独立 OCR + Pillow 像素几何从当前最新帧动态识别。

## 目录结构

```text
├── config/                  # 配置模板（M1 默认白名单仅「文件传输助手」）
├── skills/                  # 通过 HIDAGENT_SKILLS 优先挂载到底座的 Skill 定义
│   ├── wechat-send/         # M1 通用微信发消息规范
│   └── wechat-group-reply/  # M2 群监控与自动回复规范（优先覆盖底座占位）
├── scripts/
│   └── check-mount.sh       # 只读检查底座 8890 状态、staged-v1 与 HIDAGENT_SKILLS 挂载
├── src/wechat_agent/
│   ├── adapters/            # 唯一底座 CLI 适配层
│   ├── vision/              # 本仓独立 OCR（RapidOCR + 预留 Apple Vision 接口）与微信界面几何解析
│   ├── guards/              # 目标白名单硬门禁、Skill 熔断/限流守卫
│   ├── workflows/           # M1 紧连点击链、微信导航、发消息安全状态机
│   └── cli.py               # `wechat-agent` CLI（doctor / verify-skills / send）
└── tests/                   # M1 完整离线单元测试与合成页面夹具
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

## M1 CLI 用法

1. **默认 `--dry-run` 模式（只贴不发，核对正文后安全停在发送前）**：

```sh
.venv/bin/wechat-agent send --target "文件传输助手" --text "测试消息：仅粘贴不发送"
```

2. **显式 `--send` 模式（获明确授权后执行单次受守卫发送并见新绿色气泡确认）**：

```sh
.venv/bin/wechat-agent send --target "文件传输助手" --text "测试消息：真机验收" --send
```

## 隐私与证据留存

- 所有截图、OCR 核对摘要与 `paste-flow` 完整 JSON 回执均保存在仓库 `.local-data/runs/<run_id>/`（目录权限 `0700`，文件权限 `0600`），严禁写入 `/tmp`。
- 导出日志自动对 `transaction_id` 及敏感字段做脱敏处理；`.local-data/` 已被 `.gitignore` 忽略。

## 路线图

- **M1（当前阶段）**：通用微信发消息 Skill + CLI + 完整离线单测 + 「文件传输助手」真机验收。
- **M2（M1 稳固后推进）**：指定群监控循环、消息增量提取、SQLite 跨重启去重、单群/全局冷却频控与可插拔 `Responder` 策略框架（默认 `observe-only`）。
