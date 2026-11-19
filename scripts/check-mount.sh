#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLI="${IPHONE_AGENT_CLI:-/Users/seven/Documents/esp32-iphone-agent-hybrid-new/server/.venv/bin/iphone-agent}"
URL="${IPHONE_AGENT_URL:-http://127.0.0.1:8890}"
SKILLS_DIR="${REPO_ROOT}/skills"

if [[ ! -x "${CLI}" ]]; then
  echo "[ERROR] 底座 CLI 不存在或不可执行: ${CLI}" >&2
  exit 1
fi

echo "== 1. 检查底座状态 (${URL}) =="
STATUS_JSON="$("${CLI}" --url "${URL}" status)"
echo "${STATUS_JSON}"

echo ""
echo "== 2. 检查底座能力 (clipboard / host_observed_click) =="
CAPS_JSON="$("${CLI}" --url "${URL}" capabilities)"
echo "${CAPS_JSON}" | python3 -c '
import json, sys
data = json.load(sys.stdin)
clip = data.get("features", {}).get("clipboard", {})
hoc = data.get("features", {}).get("host_observed_click", {})
print(json.dumps({"clipboard": clip, "host_observed_click": {"supported": hoc.get("supported")}}, ensure_ascii=False))
'

echo ""
echo "== 3. 检查 HIDAGENT_SKILLS 挂载 =="
SKILLS_JSON="$("${CLI}" --url "${URL}" skill-list)"
echo "${SKILLS_JSON}"

if ! echo "${SKILLS_JSON}" | python3 -c 'import json, sys; sys.exit(0 if "wechat-send" in json.load(sys.stdin).get("skills", []) else 1)'; then
  echo ""
  echo "[NOT MOUNTED] 底座当前未发现 wechat-send（尚未挂载本仓 skills 目录）。"
  echo "请由你确认后，为底座 worker 配置以下环境变量并重启 worker（例如 restart-worker）："
  echo ""
  echo "  export HIDAGENT_SKILLS=\"${SKILLS_DIR}\""
  echo ""
  echo "或将其写入底座独立运行配置后执行底座重启命令；本仓库不会自动修改底座仓库。"
  exit 2
fi

echo ""
echo "== 4. 核对已挂载 wechat-send Skill 内容与限制 =="
"${CLI}" --url "${URL}" skill-show wechat-send
echo ""
echo "[OK] 底座就绪且本仓 skills/ 已成功挂载！"
