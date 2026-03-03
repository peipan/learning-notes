# 深度解析：一张图拆解OpenClaw的Agent核心设计

> OpenClaw 是一个开源的 AI Agent 框架，它把一个无状态的 LLM 变成了一个持续在线、有记忆、有主动性的长期工作伙伴。本文从架构层面拆解它的核心设计，看看它是怎么做到的。

## 1. 引言：为什么 OpenClaw 值得拆解

市面上的 Agent 框架不少，但大多数停留在"工具调用链"的层面——给 LLM 接几个 API，跑一个 ReAct 循环，就叫 Agent 了。

OpenClaw 走了一条不同的路。它不追求复杂的编排引擎或花哨的 DAG 工作流，而是用一组朴素的抽象——**文件、定时器、消息路由**——把一个 LLM 变成了一个真正意义上的"数字伙伴"：

- **有记忆**：不是靠超长 context window 硬塞，而是用 Markdown 文件做持久化，配合向量+关键词混合检索按需召回
- **有主动性**：不只是"问一句答一句"，而是通过心跳轮询和定时任务主动巡逻、主动通知
- **有人格**：通过 SOUL.md、IDENTITY.md 等文件定义性格和行为边界，每次会话启动时"恢复自我"
- **多触点**：一个 Agent 同时接入 WhatsApp、Telegram、Discord、Signal 等多个消息通道，记忆跨通道共享
- **可透视**：所有状态都是纯文本文件，用户随时可以打开查看、编辑，甚至用 Git 做版本管理

这篇文章会从架构全景开始，逐层拆解 OpenClaw 的核心设计，帮你理解它为什么能做到这些。

---

## 2. 架构全景图

![OpenClaw 架构全景图](/images/openclaw-architecture.svg)

整个系统的核心是一个单进程的 **Gateway 守护进程**，它同时扮演了消息网关、Agent 运行时、会话管理器和工具调度器的角色。所有外部通道（WhatsApp、Telegram、Discord 等）通过各自的 SDK 接入 Gateway，消息经过路由分发到对应的 Agent，Agent 在一个序列化的循环中完成推理和工具调用，最终将结果流式返回给用户。

下面我们逐层拆解每个组件。

---

## 3. Gateway：中枢神经系统

Gateway 是 OpenClaw 的心脏。它是一个长驻的 Node.js 进程，承担了三个核心职责：消息接入、协议通信、多 Agent 路由。

### 3.1 多通道统一接入

Gateway 内置了多个消息通道的 SDK：

| 通道 | 实现方式 | 说明 |
|------|---------|------|
| WhatsApp | Baileys (Web API) | 扫码登录，一个 Gateway 一个 WhatsApp 会话 |
| Telegram | grammY | 通过 BotFather 创建 Bot |
| Discord | discord.js | 需要 Bot Token + Message Content Intent |
| Signal | signal-cli | 通过 Signal CLI 桥接 |
| Slack | Slack SDK | 支持 Workspace 级别接入 |
| iMessage | BlueBubbles | 仅 macOS |
| WebChat | 内置 HTTP | Gateway 自带的 Web 聊天界面 |

所有通道的消息最终都被标准化为统一的内部格式，进入同一个处理管线。这意味着用户在 WhatsApp 上说的话，Agent 在 Telegram 上也"记得"——因为它们共享同一个 Session。

**设计亮点：Provider 抽象层**

每个通道实现一个统一的 Provider 接口，负责：
- 消息接收：将平台特定格式转为内部 envelope
- 消息发送：将 Agent 回复转为平台特定格式
- 状态管理：连接状态、认证状态、在线状态

这层抽象让新增通道变得简单——实现 Provider 接口即可，不需要改动 Agent 运行时的任何代码。

### 3.2 WebSocket 协议

Gateway 对外暴露一个 WebSocket API（默认 `127.0.0.1:18789`），所有控制面客户端（CLI、macOS App、WebChat、自动化脚本）都通过这个 WS 连接与 Gateway 通信。

协议设计简洁：

```
客户端 → Gateway:  { type: "req", id, method, params }
Gateway → 客户端:  { type: "res", id, ok, payload }
Gateway → 客户端:  { type: "event", event, payload }  // 推送
```

**连接生命周期：**

1. 客户端发送 `connect` 帧（必须是第一帧），携带设备身份和认证 token
2. Gateway 验证身份，返回 `hello-ok`（包含 presence 和 health 快照）
3. 之后进入正常的请求-响应 + 事件推送模式
4. 支持幂等键去重，防止网络抖动导致的重复请求

**安全机制：**

- 设备配对：新设备首次连接需要审批，Gateway 颁发 device token
- 本地连接（loopback）可以自动审批，保持本机使用的流畅体验
- 远程连接必须显式审批 + token 认证
- 所有连接都需要签名 challenge nonce

### 3.3 多 Agent 路由与 Bindings

OpenClaw 支持在一个 Gateway 中运行多个完全隔离的 Agent，每个 Agent 有自己的：

- **Workspace**（文件、人格、记忆）
- **Session Store**（会话历史）
- **Auth Profile**（API 密钥）
- **agentDir**（状态目录）

通过 `bindings` 配置，消息可以按通道、账号、发送者、群组等维度路由到不同的 Agent：

```json5
bindings: [
  // WhatsApp 个人号 → home agent
  { agentId: "home", match: { channel: "whatsapp", accountId: "personal" } },
  // WhatsApp 工作号 → work agent
  { agentId: "work", match: { channel: "whatsapp", accountId: "biz" } },
  // 特定群组 → 专用 agent
  { agentId: "family", match: {
    channel: "whatsapp",
    peer: { kind: "group", id: "xxx@g.us" }
  }},
]
```

**路由优先级（从高到低）：**

1. `peer` 精确匹配（特定的人或群）
2. `parentPeer` 匹配（线程继承）
3. `guildId + roles`（Discord 角色路由）
4. `guildId`（Discord 服务器）
5. `teamId`（Slack 团队）
6. `accountId` 匹配
7. `channel` 级别匹配
8. 默认 Agent（fallback）

这套路由机制非常灵活。你可以做到：一个 WhatsApp 号码，不同的人发消息过来，路由到不同的 Agent（不同的人格、不同的记忆）。也可以让日常闲聊走便宜的 Sonnet 模型，深度工作走 Opus 模型。

---

## 4. Agent Loop：一次对话的完整生命周期

当一条消息到达 Agent 时，会触发一个完整的 Agent Loop。这是 OpenClaw 最核心的执行路径。

### 4.1 完整流程

```
用户消息
    │
    ▼
┌─────────────────┐
│  消息入队        │ ← 根据队列模式决定处理方式
│  (Session Lane)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Session 加载    │ ← 从 JSONL 加载会话历史
│  + 写锁获取      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  上下文组装      │ ← System Prompt + Workspace 文件
│  (Context Build) │    + 会话历史 + 工具 Schema
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  模型推理        │ ← 调用 LLM API（流式）
│  (Inference)     │
└────────┬────────┘
         │
    ┌────┴────┐
    │ 有工具   │ 无工具调用
    │ 调用？   │────────────┐
    └────┬────┘            │
         │ 是              │
         ▼                 │
┌─────────────────┐        │
│  工具执行        │        │
│  (Tool Execute)  │        │
└────────┬────────┘        │
         │                 │
         │ 结果反馈给模型   │
         └──────┐          │
                │          │
         ┌──────▼──────┐   │
         │  继续推理    │   │
         │  (下一轮)    │   │
         └──────┬──────┘   │
                │          │
                ▼          ▼
         ┌─────────────────┐
         │  流式输出        │ ← assistant/tool/lifecycle 三条流
         │  + 持久化        │ ← 写入 JSONL 会话文件
         └─────────────────┘
```

### 4.2 队列与并发控制

每个 Session 有一个独立的执行队列（session lane），保证同一个会话内的消息严格串行处理。这避免了并发导致的状态竞争。

当 Agent 正在处理一条消息时，新消息的处理取决于队列模式：

| 模式 | 行为 | 适用场景 |
|------|------|---------|
| **steer** | 新消息注入当前运行，Agent 可以"转向" | 需要实时纠正 Agent 行为 |
| **followup** | 排队等待，当前轮结束后自动开始下一轮 | 默认模式，保证顺序 |
| **collect** | 收集多条消息，合并后一起处理 | 用户连续发多条消息 |

**steer 模式的实现细节：** 队列在每次工具调用后检查是否有新消息。如果有，当前 assistant 消息中剩余的工具调用会被跳过（返回 "Skipped due to queued user message"），然后新消息被注入，模型重新开始推理。

### 4.3 上下文组装（Context Build）

每次运行前，OpenClaw 会组装完整的上下文，这是模型"看到"的全部信息：

**System Prompt 构成：**

```
┌─────────────────────────────────────┐
│           System Prompt              │
├─────────────────────────────────────┤
│ 1. 基础指令（OpenClaw 内置）         │
│    - 工具使用规范                    │
│    - 回复格式要求                    │
│    - 安全边界                        │
├─────────────────────────────────────┤
│ 2. 工具列表 + 描述                   │
│    - 每个工具的名称和简介             │
├─────────────────────────────────────┤
│ 3. Skills 列表（仅元数据）           │
│    - 名称 + 描述 + 文件位置          │
│    - 不包含完整说明，按需读取         │
├─────────────────────────────────────┤
│ 4. 运行时元数据                      │
│    - 时间（UTC + 用户时区）          │
│    - 主机信息、模型信息              │
│    - 当前 thinking/verbose 状态      │
├─────────────────────────────────────┤
│ 5. Project Context（Workspace 文件） │
│    - AGENTS.md（操作指南）           │
│    - SOUL.md（人格定义）             │
│    - USER.md（用户信息）             │
│    - IDENTITY.md（Agent 身份）       │
│    - TOOLS.md（工具备注）            │
│    - HEARTBEAT.md（心跳清单）        │
└─────────────────────────────────────┘
```

**上下文预算管理：**

- 单文件最大注入 20,000 字符（`bootstrapMaxChars`）
- 所有文件总计最大 150,000 字符（`bootstrapTotalMaxChars`）
- 超出的文件会被截断并标记
- 工具 Schema（JSON）也占上下文，但不以文本形式展示

用户可以通过 `/context list` 和 `/context detail` 命令查看上下文的详细构成和各部分占用。

### 4.4 流式输出与事件系统

Agent Loop 的输出是完全流式的，通过三条事件流实时推送：

| 事件流 | 内容 | 用途 |
|--------|------|------|
| **assistant** | 模型的文本输出 delta | 实时显示打字效果 |
| **tool** | 工具的 start/update/end | 展示工具执行进度 |
| **lifecycle** | 运行的 start/end/error | 控制 UI 状态 |

**Block Streaming（分块发送）：** 对于长回复，OpenClaw 可以在文本块完成时就发送（而不是等整个回复结束），默认按 800-1200 字符分块，优先在段落边界切分。这在消息通道（如 WhatsApp）上体验更好。

### 4.5 Hook 系统

OpenClaw 在 Agent Loop 的关键节点提供了 Hook 拦截点：

| Hook | 时机 | 用途 |
|------|------|------|
| `before_model_resolve` | 模型选择前 | 动态切换模型/provider |
| `before_prompt_build` | 上下文组装前 | 注入额外上下文 |
| `before_tool_call` | 工具执行前 | 拦截/修改工具参数 |
| `after_tool_call` | 工具执行后 | 修改工具结果 |
| `tool_result_persist` | 结果持久化前 | 转换存储格式 |
| `agent_end` | 运行结束 | 审计/统计 |
| `before_compaction` | 压缩前 | 观察/标注 |
| `message_received` | 消息到达 | 过滤/转换 |
| `message_sending` | 消息发出前 | 最后修改机会 |

这套 Hook 系统让高级用户可以在不修改核心代码的情况下，深度定制 Agent 的行为。

---

## 5. 记忆系统：文件即记忆

这是 OpenClaw 最有特色的设计之一。它没有用向量数据库或专门的记忆服务，而是用**纯 Markdown 文件**作为记忆的存储介质。

### 5.1 双层记忆架构

```
记忆层级
├── MEMORY.md          ← 长期记忆（手动策展，类似"大脑"）
└── memory/
    ├── 2026-03-01.md  ← 每日笔记（当天发生了什么）
    ├── 2026-03-02.md
    └── 2026-03-03.md
```

- **MEMORY.md**：策展过的长期记忆，存放用户偏好、重要决定、持久事实。只在主会话（私聊）中加载，避免在群聊中泄露隐私。
- **memory/YYYY-MM-DD.md**：每日流水账，记录当天的交互和事件。每次会话启动时读取今天和昨天的文件。

这个设计的精妙之处在于：

1. **透明**：用户可以直接打开文件查看 Agent 记了什么，甚至手动编辑
2. **可版本控制**：整个记忆目录可以用 Git 管理，有完整的变更历史
3. **隐私可控**：MEMORY.md 只在私聊加载，群聊中不会泄露
4. **成本可控**：不需要额外的数据库或向量服务

### 5.2 语义搜索：向量 + BM25 混合检索

光有文件还不够——如果记忆很多，不可能每次都全部塞进上下文。OpenClaw 提供了两个记忆工具：

- **memory_search**：语义搜索，从所有记忆文件中检索相关片段
- **memory_get**：精确读取，按文件路径和行号获取特定内容

搜索引擎采用**混合检索**策略：

```
查询 → ┬→ 向量检索（语义匹配，措辞不同也能找到）
       │
       └→ BM25 检索（关键词匹配，精确 token 如 ID、代码符号）
       │
       ▼
    加权合并 → 时间衰减（可选）→ MMR 去重（可选）→ Top-K 结果
```

- **向量检索**擅长语义匹配："Mac Studio 网关主机" vs "运行网关的那台机器"
- **BM25 检索**擅长精确匹配：错误码、变量名、IP 地址等
- **时间衰减**：最近的记忆权重更高（默认半衰期 30 天），避免半年前的笔记压过昨天的更新
- **MMR 去重**：减少重复片段，确保返回的结果覆盖不同方面

### 5.3 自动记忆刷写（Pre-Compaction Flush）

当会话接近上下文窗口上限、即将触发压缩（compaction）时，OpenClaw 会自动插入一个**静默的记忆刷写轮次**，提醒模型把重要信息写入磁盘文件，避免压缩后丢失关键上下文。

这个机制是自动的，用户完全无感知。

---

## 6. Workspace：Agent 的"人格磁盘"

Workspace 是 Agent 的家目录，也是它的"人格磁盘"。每次会话启动时，OpenClaw 会把 Workspace 中的关键文件注入到上下文中，让 Agent "恢复自我"。

### 6.1 文件职责分工

| 文件 | 职责 | 加载时机 |
|------|------|---------|
| `SOUL.md` | 人格、语气、行为边界 | 每次会话 |
| `AGENTS.md` | 操作指南、工作流程、记忆规则 | 每次会话 |
| `USER.md` | 用户信息（名字、时区、偏好） | 每次会话 |
| `IDENTITY.md` | Agent 的名字、物种、风格、Emoji | 每次会话 |
| `TOOLS.md` | 本地工具备注（摄像头名、SSH 地址等） | 每次会话 |
| `HEARTBEAT.md` | 心跳检查清单 | 每次心跳 |
| `BOOTSTRAP.md` | 首次启动仪式（完成后删除） | 仅首次 |
| `MEMORY.md` | 长期记忆 | 仅主会话 |

### 6.2 Bootstrap 仪式：从零到"我是谁"

当 Workspace 是全新的（第一次使用），OpenClaw 会创建一个 `BOOTSTRAP.md` 文件，引导 Agent 和用户完成一次"认识彼此"的对话：

1. Agent 问用户：你是谁？叫我什么？
2. 一起决定 Agent 的名字、性格、风格
3. 把结果写入 IDENTITY.md、USER.md、SOUL.md
4. 删除 BOOTSTRAP.md——仪式完成，Agent "出生"了

这个设计很有仪式感，让用户从第一次交互就感受到这不是一个冷冰冰的聊天机器人。

---

## 7. 主动性引擎：Heartbeat + Cron

这是 OpenClaw 区别于大多数 Agent 框架的关键特性——它不只是被动响应，还能主动行动。

### 7.1 Heartbeat 心跳轮询

Heartbeat 是一个定时触发的 Agent 轮次（默认每 30 分钟），Gateway 会向 Agent 发送一个心跳消息：

```
Gateway 定时器
    │
    ▼ 每 30 分钟
发送心跳 Prompt → Agent 主会话
    │
    ▼
Agent 检查：
├── HEARTBEAT.md 中的待办事项
├── 邮件是否有紧急消息？
├── 日历是否有即将到来的事件？
├── 是否需要主动联系用户？
    │
    ├── 有事 → 发送通知给用户
    └── 没事 → 回复 HEARTBEAT_OK（静默丢弃）
```

关键设计点：

- **HEARTBEAT_OK 静默处理**：没事时 Agent 回复 HEARTBEAT_OK，Gateway 直接丢弃，用户无感知
- **活跃时间窗口**：可以配置只在白天运行（如 08:00-22:00），避免半夜打扰
- **成本意识**：心跳运行完整的 Agent 轮次，间隔越短 token 消耗越大

### 7.2 Heartbeat vs Cron 的分工

| 维度 | Heartbeat | Cron |
|------|-----------|------|
| 时间精度 | 大约每 30 分钟 | 精确到分钟 |
| 会话隔离 | 在主会话中运行 | 独立 Session |
| 适合场景 | 批量检查（邮件+日历+通知） | 精确调度（每周一 9:00 报告） |
| 上下文 | 有主会话的对话历史 | 无历史，干净的上下文 |

---

## 8. Session 管理：状态的持久化与隔离

### 8.1 Session Key 映射

OpenClaw 用 Session Key 来标识和隔离不同的对话上下文：

| 场景 | Session Key | 说明 |
|------|------------|------|
| 私聊（默认） | `agent:main:main` | 所有私聊共享 |
| 私聊（隔离） | `agent:main:telegram:dm:123` | 按通道+发送者隔离 |
| 群聊 | `agent:main:whatsapp:group:xxx` | 每个群独立 |
| Cron 任务 | `cron:job-id` | 每次运行独立 |

默认所有私聊共享一个主 Session（`dmScope: "main"`），跨通道保持上下文连续。多用户场景可切换到 `per-channel-peer` 模式隔离。

### 8.2 Compaction 上下文压缩

当会话历史太长、接近上下文窗口上限时，OpenClaw 自动触发 Compaction：

1. 触发记忆刷写（把重要信息写入文件）
2. 将旧的对话历史压缩为一段摘要
3. 保留最近的消息不变
4. 摘要 + 最近消息 = 新的上下文

这样 Agent 既不会"失忆"，又不会撑爆上下文窗口。

---

## 9. 工具与技能系统

OpenClaw 的工具系统分为三层：内置工具、Skills 模块、扩展能力（Sub-agents + Nodes）。

### 9.1 内置工具

这些是 Agent 始终可用的基础能力：

| 工具 | 功能 | 说明 |
|------|------|------|
| `read` | 读取文件 | 支持文本和图片 |
| `write` | 写入文件 | 自动创建父目录 |
| `edit` | 精确编辑 | 基于文本匹配的 surgical edit |
| `exec` | 执行命令 | 支持 PTY、后台运行、超时控制 |
| `process` | 管理进程 | 轮询、写入、发送按键、终止 |
| `browser` | 浏览器控制 | 截图、快照、自动化操作 |
| `message` | 发送消息 | 跨通道消息、投票、反应 |
| `memory_search` | 记忆搜索 | 语义检索记忆文件 |
| `memory_get` | 记忆读取 | 精确读取记忆片段 |
| `web_search` | 网页搜索 | Brave Search API |
| `web_fetch` | 网页抓取 | URL → Markdown |
| `tts` | 文字转语音 | 支持多种声音 |
| `sessions_spawn` | 创建子 Agent | 独立 Session 并行执行 |
| `nodes` | 设备控制 | 拍照、截屏、定位、执行命令 |

工具的可用性由 **Tool Policy** 控制，而不是 TOOLS.md（TOOLS.md 只是用户给 Agent 的使用备注）。

### 9.2 Skills 模块化加载

Skills 是 OpenClaw 的能力扩展机制。每个 Skill 是一个独立的目录：

```
skills/
├── weather/
│   ├── SKILL.md        ← 使用说明（模型按需读取）
│   └── ...             ← 可选的脚本、配置
├── clawhub/
│   ├── SKILL.md
│   └── ...
└── healthcheck/
    ├── SKILL.md
    └── ...
```

**加载优先级（高到低）：**

1. Workspace 内的 `skills/` 目录（项目级）
2. 用户级 `~/.openclaw/skills/`（全局）
3. 内置 Skills（随安装包分发）

**关键设计：延迟加载**

System Prompt 中只注入 Skills 的**名称和描述**（元数据列表），不注入完整的使用说明。模型在判断需要某个 Skill 时，才去 `read` 对应的 SKILL.md。

这个设计的好处：
- **节省上下文**：10 个 Skill 的元数据可能只占 2000 字符，但完整说明可能占 20000+
- **按需加载**：大多数对话只需要 0-1 个 Skill
- **易于扩展**：新增 Skill 只需要放一个目录，不需要改任何配置

**Skill 市场（ClawHub）：** OpenClaw 提供了一个类似 npm 的 Skill 市场（clawhub.com），可以搜索、安装、发布 Skills：

```bash
clawhub search weather
clawhub install weather
clawhub update --all
```

### 9.3 Sub-agents 子 Agent 并行

对于复杂任务，Agent 可以 spawn 子 Agent 在独立的 Session 中并行工作：

```
主 Agent
    │
    ├── spawn("分析这段代码的性能瓶颈", mode="run")
    │       → 子 Agent A（独立 Session，一次性任务）
    │       → 完成后自动汇报结果
    │
    ├── spawn("监控这个服务的日志", mode="session")
    │       → 子 Agent B（独立 Session，持久任务）
    │       → 可以通过 steer 发送后续指令
    │
    └── 主 Agent 继续处理其他事情
```

**两种模式：**

| 模式 | 行为 | 适用场景 |
|------|------|---------|
| `run` | 一次性执行，完成后自动结束 | 独立的分析/生成任务 |
| `session` | 持久运行，可以持续交互 | 需要多轮对话的复杂任务 |

子 Agent 可以指定不同的模型（比如用便宜的模型做简单任务），完成后通过 push 机制自动通知主 Agent，不需要轮询。

### 9.4 Nodes 设备配对与远程控制

OpenClaw 可以配对物理设备（手机、电脑、IoT），通过 WebSocket 连接实现远程控制：

**支持的设备能力：**

| 能力 | 说明 |
|------|------|
| `camera.snap` | 拍照（前置/后置/双摄） |
| `camera.clip` | 录制短视频 |
| `screen.record` | 屏幕录制 |
| `location.get` | 获取地理位置 |
| `canvas.*` | 在设备上展示 UI |
| 自定义命令 | 通过 `invoke` 执行设备端命令 |

**配对流程：**

1. 设备（手机/电脑）安装 OpenClaw Node 客户端
2. 连接到 Gateway 的 WebSocket，声明 `role: node`
3. Gateway 发起配对审批
4. 审批通过后，设备成为可用 Node
5. Agent 可以通过 `nodes` 工具调用设备能力

这让 Agent 的能力从纯软件扩展到了物理世界——比如"帮我拍一张前门的照片"、"我现在在哪里"。

### 9.5 Plugin 系统

除了 Skills，OpenClaw 还有一个更底层的 Plugin 系统，用于扩展 Gateway 本身的能力：

- **消息通道插件**：新增消息平台支持
- **记忆插件**：替换默认的记忆搜索引擎（如 QMD）
- **Hook 插件**：在 Agent Loop 的各个节点注入自定义逻辑
- **工具插件**：注册新的工具供 Agent 使用

Plugin 和 Skill 的区别：
- **Skill** = Agent 层面的能力扩展（模型可以读取和使用）
- **Plugin** = Gateway 层面的系统扩展（影响整个运行时行为）

---

## 10. 设计哲学总结

拆解完 OpenClaw 的核心设计，可以提炼出几个鲜明的设计哲学：

### 文件 > 数据库

记忆用 Markdown，会话用 JSONL，配置用 JSON5。没有 PostgreSQL，没有 Redis，没有 MongoDB。文件系统就是数据库，`cat` 就是查询语言，Git 就是备份方案。

这不是技术倒退，而是刻意的选择——**透明性**比性能更重要。当你的 AI 伙伴记住了什么、忘记了什么，你应该能直接看到，而不是去翻数据库。

### 心跳 > 事件驱动

没有复杂的事件总线或消息队列，一个简单的定时器就够了。每 30 分钟问一句"有什么需要关注的吗？"——简单、可靠、够用。

### 透明 > 黑盒

SOUL.md 定义人格，MEMORY.md 存储记忆，AGENTS.md 规定行为——所有影响 Agent 行为的因素都是可见的、可编辑的纯文本文件。没有隐藏的 embedding、没有不可解释的向量空间，你改一行 SOUL.md，Agent 的性格就变了。

### 简单组合 > 复杂框架

OpenClaw 没有发明新的编排语言，没有 DAG 引擎，没有状态机。它用的都是最基础的组件：

- 文件系统 → 记忆和人格
- 定时器 → 主动性
- WebSocket → 通信
- Markdown → 一切配置和内容

这些组件的组合，产生了远超各部分之和的效果。这也许是 OpenClaw 最值得学习的地方——**好的架构不是堆砌复杂度，而是找到正确的简单抽象**。

---

> 📌 项目地址：[github.com/openclaw/openclaw](https://github.com/openclaw/openclaw)
>
> 📖 官方文档：[docs.openclaw.ai](https://docs.openclaw.ai)
>
> 💬 社区：[Discord](https://discord.com/invite/clawd)
>
> 🔧 技能市场：[clawhub.com](https://clawhub.com)
