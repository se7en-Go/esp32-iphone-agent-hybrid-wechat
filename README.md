# esp32-iphone-agent-hybrid-wechat

独立的微信业务操作仓库，通过底座服务的 CLI（`iphone-agent`）与 `HIDAGENT_SKILLS` 挂载机制驱动真机 iPhone 上的官方微信 App，完成单条精准发送（M1）以及指定微信群的驻留监控与两级混合自动回复（M2）。

整套方案走 **物理外设（ESP32-S3 蓝牙键鼠） + AirPlay 投屏画面 + 剪贴板快捷键** 路线：**不修改微信客户端、不做重签名、不注入/不扫内存**，尽可能贴近真人操作习惯。

---

## 一、iPhone 与微信端设置要求（必读清单）

本仓库没有任何写死的静态坐标（无 `layout.json`），所有群标题、气泡颜色、输入栏位置、蓝色「发送」按钮、绿光标与右下角常驻图标均由本仓 OCR + Pillow 像素几何从实时画面动态识别。为了让视觉识别稳定、不误触、不误判真人接管，请在上机前按以下清单完成设置。

### 1. 微信 App 内设置

| 设置项 | 要求 | 原因说明 |
|---|---|---|
| **外观颜色模式** | **必须使用浅色模式**（关闭深色/暗黑模式） | 气泡分类与输入框识别依赖浅灰聊天底、白色对方气泡、绿色己方气泡与白色底部输入栏的稳定对比度；深色模式会直接破坏检测精度。 |
| **聊天背景** | **保持微信默认浅灰纯色背景**（不要设自定义图片/壁纸） | 自定义照片背景容易干扰白/绿气泡边缘分割与同行文字像素桥接，导致消息切分或误触发 `human_takeover`。 |
| **字体大小** | **使用系统/微信标准默认字体** | 过大或过小字体会改变多行气泡行距与输入区高度比例，影响行合并与输入栏扩展框判定。 |
| **群聊「显示群成员昵称」** | **建议开启** | 开启后每条对方白色气泡上方会有小字昵称，能准确区分是谁在说话、是否同一人在 5 秒内连续发言，以及精确生成或提取上下文。 |
| **本号群昵称（`self_aliases`）** | 把机器人在群里的昵称/简称填入配置的 `self_aliases` | 用于识别别人何时在 `@本号`（如 `良策AI`、`良策`）；`@本号` 拥有更高优先响应权和独立置信度阈值。 |
| **输入栏状态** | **保持键盘文字输入栏状态**（左侧应为声波/键盘图标，中间为白色输入条，不要停在长条「按住说话」语音模式） | 本系统通过文字粘贴 + 发送键发消息；若处于「按住说话」模式，点击底部不会弹出键盘。 |
| **「输入框内按住转文字」** | **建议在微信设置里关闭该手势浮层**；新版微信 iOS 输入框右下角常驻的灰色 `🎤` 图标及其上方半透明白色渐隐**无法关闭，无需去掉** | 针对常驻的右下角 `🎤` 图标与渐隐导致的末尾单字漂白/漏字，本仓与底座已通过「静止图标像素过滤 + `corner_fade` 物理对比 + 细标点容错」自动兼容，保留图标也能稳定识别通过。 |
| **启动前页面与草稿** | 打开并停留在**目标微信群聊窗口最底部**，确认输入框**空白无未发送草稿** | 遇到已有草稿时会安全停机、原样保留不覆盖；监控会在未进目标群时安全暂停，不会乱翻聊天列表。 |
| **挂机期间人工操作禁忌** | 无人值守挂机时，**不要在手机或同号桌面端随意手动回群消息**（除非你就是想临时人工接管） | 一旦检测到并非 AI 发出的己方新绿色气泡，系统会自动判定为「真人接管」，立即进入 10 分钟人工接管静默期。 |

### 2. iPhone 系统、键盘与投屏/外设准备

| 设置项 | 要求 | 原因说明 |
|---|---|---|
| **自动锁定** | **「设置 → 显示与亮度 → 自动锁定」设为「永不」** | 一旦锁屏或黑屏，监控会自动进入 `paused_lock_screen` 安全暂停并停止发送，无法自行解锁。 |
| **系统外观** | 设为 **「浅色」** 并关闭 **「自动（日出/日落切换深色）」** | 防止傍晚自动切深色模式导致视觉检测失效。 |
| **输入法键盘** | **使用 iOS 系统自带原生键盘**（保留顶部候选词条与右下角蓝色「发送」/「Send」按钮；不要用第三方花哨皮肤或隐藏候选栏的键盘） | 本仓依赖候选词条下缘与输入栏位置、绿色光标 `caret_scope` 以及右下角蓝色「发送」按钮做多重发送前核验（默认使用受保护 `key enter` 发送）。 |
| **「从其他 App 粘贴」权限** | **「设置 → 微信 → 从其他 App 粘贴」强烈建议设为「允许」** | 若保留「询问」，每次 `Cmd+V` 都会弹出系统权限框，底座虽会自动识别并点击「允许粘贴」，但会额外增加 3～6 秒耗时和失败风险。 |
| **辅助触控（AssistiveTouch）** | **开启辅助触控**；空闲小圆点建议停在屏幕边缘（不要长时间压在顶部群标题文字或右下角「发送」按钮上） | 底座通过指针定位执行点击和快捷指令抄送，避免遮挡标题可减少 OCR 误读。 |
| **系统通知与弹窗** | **挂机时建议开启「勿扰模式」**（关闭顶部横幅通知预览与低电量提醒弹窗） | 顶部横幅弹窗会短暂遮挡聊天群标题，导致监控在弹窗期间自动暂停（`paused_left_group` / `paused_non_wechat`）。 |
| **底座物理链路** | 确认 AirPlay 投屏、ESP32-S3 串口与蓝牙连接就绪 | 执行 `.venv/bin/wechat-agent doctor` 可一键确认：`ready: true`、`staged_v1_ready: true`、`wechat_send_mounted: true`。 |

---

## 二、监控时间是怎么算的？（通俗解读与参数说明）

很多人刚开始调参时容易把 **「多久看一次屏幕」** 和 **「多久允许发一条回复」** 混在一起。其实系统里的时间分 **6 条独立计时线**，互不混淆：

### 1. 巡检节奏：每 2～3 秒看一次画面，没变化就不跑 OCR
- **`poll_interval_min_seconds`（默认 `2.0` 秒） ～ `poll_interval_max_seconds`（默认 `3.0` 秒）**：
  每一轮巡检完成后，随机休息 `2.0 ～ 3.0` 秒（随机抖动更贴近人工浏览），再抓下一帧画面。
- **静帧跳过 OCR 省算力优化**：
  如果当前帧聊天区域（顶部到底部输入栏之上）与上一帧相比，平均亮度变化 `< 0.5` 且像素变化比例 `< 0.2%`，并且没有待办消息需要排队，**这一轮会直接跳过耗时的 RapidOCR 识别**，在审计日志里记为 `Static chat viewport; skipped OCR`，几乎零 CPU 占用。
- **临时异常退避（指数退避）**：
  如果某次抓图或接口临时报错（如网络/投屏短暂抖动），不会死循环猛刷，而是按 `min(max(poll_interval_max_seconds, 2.0) * 2^(连续错误次数 - 1), 30.0)` 退避等待（默认约 **`3s → 6s → 12s → 24s → 30s`**）；只要成功恢复 1 轮就立刻清零计数器，连续失败 **5 次** 自动停机保护。

### 2. 基线对齐与连发合并：不回旧消息，同人连发先拼成一句话
- **首次启动「只记基线不回旧账」**：
  新启动 `monitor` 对当前群执行的第 1 帧（`BASELINE_SEEDED`）只把屏幕上已经存在的聊天记录记作已读基线，**绝不会对启动前的历史消息补发回复**。
- **`burst_merge_window_seconds`（默认 `5.0` 秒）—— 同人连发合并窗口**：
  群友经常把一句话拆成两三条连发（例如第 0 秒发「@良策AI」，第 2 秒发「这个怎么下单」）。同一发信人在 **5 秒内** 相邻到达的入群消息会在本地 `pending_queue` 自动合并成一条再交由 AI 评估，避免一问三答或答非所问。

### 3. 待办排队超时丢弃：`pending_ttl_seconds`（默认 `180.0` 秒 / 3 分钟）
- 当消息因为冷却未到而暂留在待办队列 `pending_queue` 中时，一旦距离它最新到达/合并更新的时间超过 **180 秒（3 分钟）**，系统就会把它标记为 `expired` 自动丢弃。
- **为什么这样设计**：防止冷却结束后或夜晚静音解除后，机器人突然对十多分钟前早已过时的聊天“诈尸回复”。

### 4. 核心频控、冷却与静默期：按顺序过 6 道闸，SQLite 持久化不因重启清零

每次从队列里取出消息准备发出去之前，系统会严格按以下 **6 道时间闸** 依次检查（数据存在 `.local-data/state/wechat_monitor.db` 的 `send_events` 与 `group_state` 表里，**重启监控进程不会清空计时**）：

1. **夜间静默时段 `quiet_hours`（默认 `"23:00-07:00"`）**
   - 按 Mac 本机时钟判断，支持跨午夜（`[23:00, 07:00)` 左闭右开）；处于静默时段内只看不发（`skipped_quiet_hours`）。填 `"off"` 可关闭。
2. **人工接管冷却 `human_takeover_cooldown_seconds`（默认 `600.0` 秒 / 10 分钟）**
   - 一旦屏幕右上侧出现**新的己方绿色气泡**，系统会拿它跟最近 **300 秒** 内 AI 刚发过且尚未在群流里核销的记录（`sent_confirmed`）做比对；若匹配不上，就说明是**真人客服/店主亲自插手回复了**！
   - 立刻进入 **600 秒（10 分钟）人工接管静默期**，停止所有自动回复，把对话完全让给真人。如果在 10 分钟内真人再次发言，10 分钟倒计时会向后顺延刷新。
3. **单群发送冷却 `per_group_cooldown_seconds`（正式默认 `60.0` 秒）**
   - 看本群上一次真正发送成功（见新绿气泡确认）的时间：如果**距今不足 60 秒**，暂不回复（`skipped_cooldown`），让新来的消息先在 `pending_queue` 里待着，等满 60 秒且还没超过 `pending_ttl_seconds`（180 秒）的下一轮巡检再自动拉起来发。
4. **单群每小时上限 `per_group_max_per_hour`（正式默认 `6` 次/小时，合法范围 `1..30`）**
   - ⚠️ **是「最近 3600 秒滚动滑动窗口」，不是整点清零！**
   - 举个例子：如果在 `14:05`、`14:10`、`14:20`、`14:30`、`14:40`、`14:50` 本群一共发了 6 次，那么到了 **`15:00` 整点不会解禁**，必须等到 **`15:05`（最早那条超过 3600 秒滚出窗口）** 才会释放第 1 个名额！
5. **全局最小发送间隔 `global_min_interval_seconds`（默认 `15.0` 秒，代码硬下限 `3.0` 秒）**
   - 跨群级别的防高频间隔，距全账号任意群上次发送不足 `15.0` 秒时暂缓。
6. **全局每小时上限 `global_max_per_hour`（默认 `15` 次/小时，硬上限 `30` 次/小时）**
   - 同为最近 `3600 秒` 滚动窗口统计全账号总发送次数；此外底座侧挂载的 `wechat-group-reply` Skill 还兜底了一道「最小间隔 `12s`、每小时最多 `15` 次、`23:00-07:00` 静默、`skill-prepare-send` 占用名额有效期 `30s`」的二次锁，**本地与底座永远取更严格的一侧生效**。

### 5. 短断层恢复窗口：`short_discontinuity_window_seconds`（默认 `90.0` 秒）
- 有时因为群里突然刷屏太快，或一次发送耗时几十秒期间刷过了好几屏，当前屏幕跟上次记住的最后几条记录对不上重叠（无共同后缀，记为 `discontinuity`）。
- 只要距上次见到该群正常画面 **不超过 `90.0` 秒**，系统会核对过去 1 小时去重库，精准捞回期间漏掉的 **`@本号` 消息 + 画面最底部最多 3 条未见消息** 以及任何己方新气泡（防止漏判人工接管）；
- 如果断层间隔 **超过 `90.0` 秒**（例如手机切出去好几分钟才切回群里），系统会直接用当前屏幕重新锚定基线，绝不翻几分钟前的旧账乱回复。

### 6. 单次回复真实耗时参考（端到端约 27～35 秒）
从监测到群里出现新提问，到绿气泡出现在屏幕上，实测典型耗时分解如下：
- **LLM 意图判定 + 拟回复生成**：约 `2 ～ 5` 秒（配置上限 `llm.timeout_seconds: 30.0`）
- **导航/聚焦/防草稿 + 底座 `paste-flow`（快捷键抄送 + `Cmd+V` + 剪贴板 Confirm 多帧确认）**：约 `20 ～ 25` 秒（超时保护 `paste_flow_timeout_seconds: 180.0`）
- **简化版 Enter 发送前置单帧核对 + 受保护 `key enter` + 确认新绿色气泡**：约 `5 ～ 6` 秒（在信任底座 `confirmation_evidence` 的 `enter` 实发路径上已省去一次重复 `04_post_paste` OCR 往返）
- 整体单轮端到端 **约 `27 ～ 35` 秒**；同时因为每条真实发出后才写入 `send_events` 并更新发后视口基线，下一次 2～3 秒巡检看到刚才自己发的绿气泡会精确核销，**不会把自己误判成人工接管**。

---

### 💡 两个通俗时间场景示例

**场景 A：同群连续有人提问（按正式默认配置：`冷却 60s`、`上限 6 次/小时`、`待办有效期 180s`）**
- `10:00:00` 群友甲问「@良策AI 什么时候发货」 → `10:00:30` 成功回复发出（记录本群第 1 次发送）。
- `10:00:50` 群友乙紧接着问「@良策AI 偏远地区包邮吗」 → 当前距 `10:00:30` 只有 `20 秒 < 60 秒` 冷却，**先留在 `pending_queue` 不回**；
- `10:01:31` 轮询触发，已满 `61 秒 ≥ 60 秒` 且排队仅 41 秒未超 `180 秒` 过期 → 自动取出群友乙的消息并生成发出！
- 若冷却被调成 300 秒，导致群友乙等到 `10:04:00`（超过 `180 秒`）还没发出去 → 群友乙的问题自动 `expired` 丢弃，不再补回。

**场景 B：AI 正在运营时，真人店主自己插了一句话**
- `10:10:00` 店主用其他设备或手机亲自发了一条「这个我马上给你们改价」；
- `10:10:03` 监控看到这条右侧新绿气泡，与过去 300 秒内 AI 发过的文案不匹配 → **立刻触发 `human_takeover`，锁定静默至 `10:20:03`**；
- `10:12:00` 即使有人继续 `@良策AI`，监控也只记录 `skipped_human_takeover`，保持完全沉默，直到 `10:20:03` 后且无人继续插话才自动恢复接管。

### 📊 时间与频控参数速查表

| 配置键名 | 代码/模板默认值 | 含义与作用 | 联调测试建议 vs 正式挂机建议 |
|---|---|---|---|
| `poll_interval_min_seconds` / `max` | `2.0` ～ `3.0` 秒 | 每次观察间隔随机休眠区间 | 测试/正式均保持 `2.0 ～ 3.0` |
| `burst_merge_window_seconds` | `5.0` 秒 | 同发信人连发合并窗口 | 保持 `5.0` |
| `pending_ttl_seconds` | `180.0` 秒（3 分钟） | 待办消息未回复的过期丢弃时间 | 保持 `180.0`（或随冷却同步略放宽） |
| `short_discontinuity_window_seconds` | `90.0` 秒 | 短暂无重叠刷屏时的漏消息捞回窗口 | 保持 `90.0` |
| `per_group_cooldown_seconds` | `60.0` 秒 | 单群两次自动回复最小间隔（可在 `target_groups[0]` 单独覆盖） | 联调连测可临时调 `15.0`；**正式挂机用 `60.0`+** |
| `per_group_max_per_hour` | `6` 次/小时 | 单群过去 3600 秒滚动窗口最大发送条数（上限 `30`） | 联调可临时调 `15`；**正式挂机用 `6`** |
| `global_min_interval_seconds` | `15.0` 秒 | 全局最小间隔（代码硬下限 `3.0s`） | 保持 `>= 15.0` |
| `global_max_per_hour` | `15` 次/小时 | 全局 3600 秒滚动窗口总上限（代码硬上限 `30`） | 保持 `<= 15` |
| `human_takeover_cooldown_seconds` | `600.0` 秒（10 分钟） | 检测到真人发新绿气泡后的让位静默时间 | 保持 `600.0` |
| `quiet_hours` | `"23:00-07:00"` | 夜间只看不发静默窗（`"off"` 关闭） | 按店铺客服班次调整或 `"off"` |
| `cli_timeout_seconds` | `90.0` 秒 | 底座单次 CLI 命令超时 | 保持 `90.0` |
| `paste_flow_timeout_seconds` | `180.0` 秒 | 底座多阶段 `paste-flow` 总超时 | 保持 `180.0` |
| `llm.timeout_seconds` | `30.0` 秒 | 单次 LLM 请求超时（失败即按当前 tick 快速熔断） | 保持 `30.0` |

> **历史数据自动保留周期**：SQLite 由监控每 100 轮自动清理一次（`seen_messages` 保留 14 天、`reply_history` 与 `send_events` 保留 30 天、已完结 `pending_queue` 保留 7 天，单表上限 `50,000` 行）。

---

## 三、目录结构与架构约束

- **零代码拷贝**：本仓库不复制、不导入底座仓库源码，仅通过受管子进程调用底座 `iphone-agent` CLI。
- **不修改底座**：不改动底座仓库代码，也不在未经确认的情况下修改底座运行配置。
- **全动态定位**：不设任何静态 `layout.json`，所有界面定位由实时图像与 OCR 算得。

```text
├── config/
│   ├── wechat-send.example.yaml     # M1 通用发消息配置模板
│   ├── group-monitor.example.yaml   # M2 群监控配置模板（默认 observe-only）
│   └── rules.example.yaml           # M2 确定性 YAML 规则模板
├── skills/                          # 通过 HIDAGENT_SKILLS 优先挂载到底座的 Skill 定义
│   ├── wechat-send/                 # M1 通用微信发消息规范与避坑边界
│   └── wechat-group-reply/          # M2 群监控与自动回复频控/停词规范
├── scripts/
│   └── check-mount.sh               # 只读检查底座 8890 状态、staged-v1 与 HIDAGENT_SKILLS 挂载
├── src/wechat_agent/
│   ├── adapters/                    # 唯一底座 CLI 适配层
│   ├── vision/                      # 独立 OCR（RapidOCR / Apple Vision 接口）与像素几何解析
│   ├── guards/                      # 目标白名单、Skill 熔断限流、M2 双层频控与违禁词守卫
│   ├── workflows/                   # M1 紧连点击链、微信导航、发消息安全状态机
│   ├── monitor/                     # M2 群提取、SQLite 持久化去重、两级决策/生成、监控循环
│   │   ├── decision/                # Layer 1: DecisionGate (rule / llm / hybrid / jev 扩展槽)
│   │   └── generators/              # Layer 2: ReplyGenerator (rule_template / llm / hybrid)
│   └── cli.py                       # `wechat-agent` CLI（doctor / verify-skills / send / monitor）
└── tests/                           # 离线单元测试与真机多帧回放测试
```

---

## 四、安装、挂载与环境体检

### 1. 安装独立虚拟环境

推荐使用 Python 3.12 + `uv`：

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
```

### 2. 挂载 `HIDAGENT_SKILLS` 到底座

确保底座服务端在启动前配置环境变量指向本仓 `skills/` 目录（并重启过底座 worker）：

```sh
export HIDAGENT_SKILLS=/Users/seven/Documents/esp32-iphone-agent-hybrid-wechat/skills
```

### 3. 一键体检

```sh
./scripts/check-mount.sh
.venv/bin/wechat-agent verify-skills
.venv/bin/wechat-agent doctor
```

确认输出中 `base_ready: true`、`staged_v1_ready: true`、`wechat_send_mounted: true` 即可继续。

---

## 五、配置准备与 LLM 密钥管理

1. **复制本地私有配置**（`config/*.local.yaml` 已被 `.gitignore` 忽略，切勿提交到仓库）：

```sh
cp config/group-monitor.example.yaml config/group-monitor.local.yaml
```

2. **配置群名称与话术边界**（编辑 `config/group-monitor.local.yaml`）：
   - 在 `target_groups` 填入唯一要监控的群名（支持自动忽略末尾群人数如 `(198)` 与中点号 `•` / `·` 差异）；
   - 填写 `self_aliases`（本号群昵称列表）与 `role_prompt`（本群业务范围、回复语气与静默红线，例如遇退款/投诉/闲聊一律保持静默）；
   - 可选 `send_method`：默认 `"enter"`（先核对绿色光标作用域再受保护下发 `Enter`，耗时更短更稳），或 `"click"`（指针动态定位点击右下角蓝色「发送」按钮）；
   - `default_gate` 支持 `hybrid`（推荐，先走 YAML 规则快通道，未命中再走 LLM 意图判断，且命中退款/售后静默类会自动跳过）、`rule`、`llm`、`observe_only`。

3. **准备 LLM API 密钥（如启用 `llm.enabled: true`）**：
   建议将密钥放入 `0600` 权限的私有文件 `.local-data/state/llm.env`（严禁写入 YAML 或提交 Git）：

```sh
mkdir -p .local-data/state && chmod 700 .local-data .local-data/state
cat <<'EOF' > .local-data/state/llm.env
LLM_BASE_URL="https://api.deepseek.com/v1"
LLM_API_KEY="sk-your-key"
LLM_MODEL="deepseek-flash"
EOF
chmod 600 .local-data/state/llm.env
```

---

## 六、常用 CLI 用法与推荐上线流程

### 1. M1 单次发消息测试（默认白名单仅 `文件传输助手`）

- **仅粘贴核对、不发送（`--dry-run` 默认模式）**：

```sh
.venv/bin/wechat-agent send --target "文件传输助手" --text "测试消息：仅粘贴不发送"
```

- **确认粘贴无误后执行单次实发（`--send`）**：

```sh
.venv/bin/wechat-agent send --target "文件传输助手" --text "测试消息：真机验收" --send
```

### 2. M2 群监控四步上线流程

```sh
# 1. 带本机监控配置执行一次完整体检（检查群配置、Skill 挂载、底座状态）
.venv/bin/wechat-agent doctor --config config/group-monitor.local.yaml

# 2. 导入 LLM 环境变量，单步只读观察一次（对齐最新聊天基线，零点击零输入）
set -a && source .local-data/state/llm.env && set +a
.venv/bin/wechat-agent monitor --config config/group-monitor.local.yaml --mode observe-only --once

# 3. 如需先看决策效果但不发消息，可保持 observe-only 持续运行并查看审计日志
.venv/bin/wechat-agent monitor --config config/group-monitor.local.yaml --mode observe-only

# 4. 确认决策与静音边界无误、手机停留在目标群且输入框空白后，加双保险开关正式运行
.venv/bin/wechat-agent monitor --config config/group-monitor.local.yaml --mode live --allow-live-send
```

> **重要安全提示**：
> - 对任何真实微信群，`--mode observe-only` 与 `--mode dry-run` **都只做只读观察和审计日志记录，绝不点击输入框或粘贴任何文字**。
> - 运行监控守护进程期间，**不要再用外部命令行单独调用 `iphone-agent observe`**，否则会顶掉当前监控线程持有的单次 `screen_id`。
> - 启动监控前如果改过 `config/group-monitor.local.yaml`，注意按需确认测试期放宽的 `per_group_cooldown_seconds` / `per_group_max_per_hour` 已恢复为正式值（`60.0` / `6`）。

---

## 七、安全红线、自动暂停与证据留存

- **已有草稿保护**：一旦聚焦输入框发现内有非空草稿，立即保留现场停机（`aborted_existing_draft`），绝不覆盖或发出。
- **风控词与违禁词双熔断**：`skill-check` 命中 `操作频繁`、`风控`、`验证码`、`账号异常`、`发送失败` 等屏幕停词立即停机；本地正/反向正则拦截手机号、身份证号、验证码、密码、转账收款引导等高危回复文本。
- **环境异常安全让位（`should_halt=false` 自动暂停，不乱点）**：遇到锁屏（`paused_lock_screen`）、被切到其他聊天或列表（`paused_left_group`）、退出微信或弹横幅（`paused_non_wechat`）会自动暂停观察，手机恢复界面后无需重启自动续跑。
- **不确定发送状态 fail-closed（绝不盲目重贴重发）**：遇到 `paste_sent_unconfirmed`、`SEND_UNCONFIRMED` 或底座剪贴板锁未关闭（`transaction_left_open=true`）时会立刻停机（退出码 `2`）并保留手机画面现场，需人工肉眼核对后清理锁与草稿再恢复。
- **隐私与文件权限**：运行证据 `.local-data/runs/<run_id>/` 与状态库 `.local-data/state/` 自动强制 `0700` 目录 / `0600` 文件权限，导出日志自动脱敏 `transaction_id`、手机号、身份证与 Token，严禁写入 `/tmp`。
