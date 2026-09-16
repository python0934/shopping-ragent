<h1 align="center">Ragent AI · Python 版</h1>

<p align="center">
  <strong>面向 Agentic RAG 的生产级 AI 应用平台 —— FastAPI 全异步重构实现</strong>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white" />&nbsp;
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-0.115+-009688?style=flat-square&logo=fastapi&logoColor=white" />&nbsp;
  <img alt="SQLAlchemy" src="https://img.shields.io/badge/SQLAlchemy-2.0%20async-D71F00?style=flat-square" />&nbsp;
  <img alt="PostgreSQL" src="https://img.shields.io/badge/PostgreSQL-pgvector-4169E1?style=flat-square&logo=postgresql&logoColor=white" />&nbsp;
  <img alt="Redis" src="https://img.shields.io/badge/Redis-5.2+-DC382D?style=flat-square&logo=redis&logoColor=white" />&nbsp;
  <img alt="Tests" src="https://img.shields.io/badge/pytest-628%20tests-blue?style=flat-square" />&nbsp;
  <img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-blue?style=flat-square" />
</p>

---

> **关于本仓库**：根目录保留了 Ragent AI 平台的原 **Java 实现**（Spring Boot 4，`framework` / `infra-ai` / `system` / `rag` / `agent` / `mcp-server` / `bootstrap` 七个 Maven 模块）。`ragent-py/` 是它的 **Python 重构版**——以 FastAPI + SQLAlchemy 2.0 全异步技术栈重建，目标是 **API 契约 100% 兼容、前端零改动**。本文档聚焦 Python 重构版。

## 🚀 什么是 Ragent AI · Python 版？

Ragent 是一个覆盖「文档入库 → 多路检索 → 智能问答 → Agent 推理」完整链路的 Agentic RAG 应用平台。Python 版在保持与原 Java 版**完全一致的 REST / SSE 契约**前提下，用 asyncio 全异步模型重写核心引擎，重点落地了 **Agent ReAct 执行引擎**、**AI 模型基础设施**与 **MCP 工具集成**。

- **Agent ReAct 引擎**：`think → act → observe` 多轮推理循环，知识检索与 MCP 工具作为可调用工具，支持上下文三级压缩、长期记忆、写操作人工确认与并发闸门。
- **AI 模型基础设施**：Chat / Embedding / Rerank / VLM 统一客户端，多供应商（Ollama / 百炼 / AIHubMix / SiliconFlow）路由、三态断路器、故障转移与首包探测。
- **MCP 工具集成**：基于官方 `mcp` Python SDK 的工具发现、参数提取与 Schema 校验，Agent 可远程调用外部工具服务。
- **知识检索门面**：pgvector 向量检索 + 可选重排 + LLM 合成 + 指代消解改写，作为 Agent `search_knowledge` 工具的唯一后端。
- **契约兼容**：所有响应（含异常）统一 HTTP 200 + `Result` 信封、Redis 随机 Token 认证（`Authorization` 无 `Bearer` 前缀）、19 位雪花 ID、SSE 分事件流式协议——与原 Java 版逐字对齐，前端无需任何改动。

## ✨ 核心特性

| 能力 | 说明 |
|:---|:---|
| **全异步架构** | FastAPI + `asyncio` + `httpx.AsyncClient` + SQLAlchemy 2.0 async，I/O 密集型链路全程无阻塞 |
| **ReAct 多轮推理** | 自实现 ReAct 循环（`max_iters` 可控），边推理边决定调用哪个工具、调用几次 |
| **上下文三级管理** | 裁剪（trim）→ 压缩（compact）→ 摘要（summary），按水位控制 Token 成本 |
| **模型容错** | 多候选模型故障转移，三态断路器（CLOSED / OPEN / HALF_OPEN）隔离故障节点，首包探测提前发现不可用 |
| **SSE 流式协议** | `meta` / `message` / `tool` / `hint` / `finish` / `done` / `cancel` / `reject` 事件，与前端 `EventSource` 监听字面量一致 |
| **MCP 工具桥接** | 将远程 MCP Server 的工具适配为 Agent 原生工具，复用参数提取与 Schema 校验 |
| **统一行为契约** | 全局异常处理器兜底 HTTP 200、幂等拦截、演示模式写保护、请求 ID 注入 |

## 🧱 技术栈映射（Java → Python）

| 层次 | 原 Java 技术 | Python 重构 |
|:---|:---|:---|
| Web 框架 | Spring Boot 4 (WebMVC) | **FastAPI** + Uvicorn |
| ORM | MyBatis-Plus | **SQLAlchemy 2.0 (async)** + asyncpg |
| 认证 | Sa-Token（Redis 随机 Token） | **Redis Token 中间件**（保持无 `Bearer` 前缀契约） |
| 缓存 / 分布式 | Redis + Redisson | **redis-py (asyncio)** |
| HTTP 客户端 | OkHttp | **httpx (async)** |
| 向量数据库 | Milvus SDK / pgvector | **pgvector**（SQLAlchemy 集成）/ pymilvus |
| MCP | MCP Java SDK | **mcp** Python SDK |
| Agent 框架 | AgentScope | **自实现 ReAct**（核心循环 + 工具注册表 + 上下文三级管理） |
| 配置管理 | Spring `@ConfigurationProperties` | **Pydantic BaseSettings**（嵌套模型复刻 `application.yaml` 层级） |
| SSE 流式 | Spring `SseEmitter` | FastAPI **StreamingResponse** |
| 序列化 | Jackson | **Pydantic v2** + orjson |

## 📁 项目结构

```
ragent-py/
├── app/
│   ├── main.py              # FastAPI 入口：中间件栈（全局异常/认证/演示模式/请求ID）+ 路由挂载
│   ├── config.py            # Pydantic BaseSettings，复刻 application.yaml 完整层级
│   ├── core/                # ← framework：Result 信封 / 三级异常 / 错误码 / SSE / 雪花 ID / Redis / 幂等 / 用户上下文 / DB
│   ├── infra/               # ← infra-ai：Chat/Embedding/Rerank/VLM 客户端 / 模型路由（断路器+故障转移）/ Token 计数
│   ├── agent/               # ← agent：ReAct 引擎 / 上下文压缩 / 事件桥 / 记忆 / provider / 并发闸门 / 状态存储 / 工具
│   ├── mcp/                 # ← MCP 客户端管理 + 工具注册表
│   ├── models/              # SQLAlchemy ORM：agent / chat / ingestion / knowledge / rag / system / vector
│   ├── schemas/             # Pydantic 请求/响应模型
│   ├── routers/             # API 路由：auth / user / agent / rag_chat / chat / knowledge / ingestion / audit / admin / sample_question
│   ├── services/            # 业务服务：agent 对话/会话、知识检索门面、prompt、意图注册表、用户/认证/审计
│   └── resources/prompt/    # Prompt 模板资源
├── tests/
│   ├── unit/                # 单元测试（14 个文件）
│   ├── integration/         # 集成测试（Agent API）
│   └── e2e/                 # 端到端测试（Agent 流式）
├── .env.example             # 环境变量样例
└── pyproject.toml           # 依赖与工具配置（pytest / ruff）
```

## 🏗️ 架构分层

```mermaid
flowchart TD
    FE["前端 React 控制台<br/>（复用原版，零改动）"]
    subgraph APP["FastAPI 应用 (app/main.py)"]
        MW["中间件栈：全局异常兜底 → 认证 → 演示模式 → 请求ID"]
    end
    R["路由层 routers/<br/>agent · rag_chat · chat · knowledge · ingestion · auth · user · admin"]
    S["服务层 services/<br/>agent 对话/会话 · 知识检索门面 · prompt · 意图注册表 · 用户/认证/审计"]
    A["Agent 引擎 agent/<br/>ReAct 循环 · 上下文压缩 · 事件桥 · 长期记忆 · 工具目录 · 并发闸门"]
    I["AI 基础设施 infra/<br/>模型路由（三态断路器 + 故障转移 + 首包探测）· Chat/Embedding/Rerank/VLM"]
    M["MCP 集成 mcp/<br/>工具注册表 · 参数提取 · Schema 校验"]
    C["核心基座 core/<br/>Result 信封 · 异常体系 · SSE · 雪花 ID · Redis · 幂等 · 用户上下文"]
    DB[("PostgreSQL<br/>+ pgvector")]
    RD[("Redis")]
    EXT["外部模型 API · MCP Server · S3/OSS"]

    FE -->|"REST / SSE（契约兼容）"| MW --> R --> S
    S --> A
    A --> I
    A --> M
    S --> C
    A --> C
    I --> C
    C --> DB
    C --> RD
    I --> EXT
    M --> EXT
```

## ⚡ 快速开始

### 环境要求

- Python **3.11+**
- PostgreSQL（含 **pgvector** 扩展）
- Redis
- 至少一个可用的模型供应商 API Key（百炼 / AIHubMix / SiliconFlow / 本地 Ollama）

### 1. 安装依赖

```bash
cd ragent-py
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS / Linux
# source .venv/bin/activate

pip install -e ".[dev]"
```

### 2. 配置环境变量

```bash
cp .env.example .env
```

按需填写 `.env`（关键项）：

| 变量 | 说明 | 默认值 |
|:---|:---|:---|
| `SERVER_PORT` | 服务端口 | `9090` |
| `API_PREFIX` | 全局 API 前缀 | `/api/ragent` |
| `DATABASE_URL` | PostgreSQL 异步连接串 | `postgresql+asyncpg://postgres:postgres@127.0.0.1:5432/ragent` |
| `REDIS_HOST` / `REDIS_PORT` / `REDIS_PASSWORD` | Redis 连接 | `127.0.0.1` / `6379` / `123456` |
| `ENGINE_TYPE` | 引擎类型：`rag` \| `agent` | `agent` |
| `DEMO_MODE` | 演示模式（拦截写操作） | `false` |
| `BAILIAN_API_KEY` / `AIHUBMIX_API_KEY` / `SILICONFLOW_API_KEY` | 各模型供应商密钥 | 空 |

> 嵌套配置支持通过 `__` 分隔的环境变量覆盖（如 `AI__CHAT__DEFAULT_TIER`），详见 `app/config.py`。

### 3. 初始化数据库

Python 版**复用原 Java 版的表结构**，直接执行仓库内的建表脚本：

```bash
psql -U postgres -d ragent -f ../resources/database/schema_pg.sql
psql -U postgres -d ragent -f ../resources/database/init_data_pg.sql
```

> 需确保目标库已安装 pgvector 扩展（`CREATE EXTENSION IF NOT EXISTS vector;`）。

### 4. 启动服务

```bash
uvicorn app.main:app --host 0.0.0.0 --port 9090 --reload
```

健康检查：

```bash
curl http://localhost:9090/health
# {"status":"ok","version":"1.0.0"}
```

## 📜 API 行为契约

Python 版严格复刻原 Java 版的运行时行为契约，任何偏离都会导致前端异常。关键约束：

| 契约 | 要求 |
|:---|:---|
| **统一响应** | 所有接口（含异常）返回 **HTTP 200**，响应体 `{code, message, data, requestId}`；`code="0"` 为成功，前端靠 `code` 判断而非 HTTP 状态码 |
| **分页结构** | 五字段 `records` / `total` / `size` / `current` / `pages`（对齐 MyBatis-Plus `IPage`） |
| **认证 Token** | 服务端随机字符串存 Redis，前端置于 `Authorization` header（**无 `Bearer` 前缀**）；未登录返回 `code="A000100"` |
| **雪花 ID** | 所有主键 / conversationId / messageId 为 **19 位数字字符串**（JSON 字符串类型，避免 JS 精度丢失） |
| **SSE 事件** | RAG：`meta`/`message`/`finish`/`done`/`cancel`/`reject`；Agent 额外增加 `tool`/`hint`。`message.type` 分 `response`（正文）与 `think`（思考） |
| **幂等 / 限流** | 命中时返回普通 JSON（**非 SSE 流**）：重复提交 `A000105`、排队满 `A000106` |
| **演示模式** | 写操作拦截返回 `A000107`；SSE 端点发 `reject` 事件 |
| **文件上传** | 单文件 ≤ 50MB，请求体 ≤ 100MB；`/rag/settings` 回传该限制 |

## 🚦 实现进度

Python 重构按「数据流关键度」分阶段推进，当前状态：

| 分层 / 模块 | 状态 | 说明 |
|:---|:---:|:---|
| `core/` 基础框架 | ✅ | Result 信封、三级异常、错误码、SSE、雪花 ID、Redis、幂等、用户上下文、异步 DB |
| `infra/` AI 基础设施 | ✅ | Chat/Embedding/Rerank/VLM 客户端、模型路由（断路器 + 故障转移 + 首包探测）、Token 计数 |
| `agent/` ReAct 引擎 | ✅ | ReAct 循环、上下文三级压缩、SSE 事件桥、长期记忆、工具目录、并发闸门、状态持久化 |
| `mcp/` 工具集成 | ✅ | MCP 客户端管理、工具注册表、参数提取与 Schema 校验 |
| 系统支撑域（user/auth/audit/sample） | ✅ | Redis Token 认证、用户管理、审计日志、示例问题 |
| Agent 对话 API | ✅ | `/agent/v1/chat`（SSE）、`/agent/v1/meta`、`/agent/v1/stop`、会话与消息管理 |
| 知识检索门面 | ✅ | 向量检索 + 重排 + LLM 合成 + 指代消解（Agent `search_knowledge` 后端） |
| `/rag/settings` | ✅ | 前端初始化所需的完整嵌套配置结构 |
| 多通道检索编排 | 🚧 | 向量/关键词/图谱/Web 通道与 RRF 融合配置已就绪，检索引擎整体移植进行中 |
| RAG v3 问答主链路 | 🚧 | `/rag/v3/chat`、Trace、eval（Phase 4） |
| 文档入库 Pipeline | 🚧 | 流水线创建与节点编排（Phase 4） |
| 意图树 / Agent 配置管理端 | 🚧 | `/intent-tree`、`/agents` CRUD（Phase 4 / 5） |
| RAG 会话 / 管理后台仪表盘 | 🚧 | `/conversations`、`/admin` 概览（Phase 3） |

> 图例：✅ 已实现并有测试覆盖 · 🚧 占位 / 进行中

## 🧪 测试

采用分阶段测试策略：单模块完成后跑单元测试，多模块 / 全链路完成后跑集成与端到端测试。

```bash
cd ragent-py

# 全部测试
pytest

# 仅单元测试
pytest tests/unit

# 集成测试（Agent API）
pytest tests/integration

# 端到端测试（Agent 流式）
pytest tests/e2e

# 覆盖率
pytest --cov=app
```

当前测试规模：**24 个测试文件、628 个测试点**，覆盖模型路由与断路器、SSE 事件协议、雪花 ID、异常体系、Agent ReAct 运行时、上下文压缩、长期记忆、MCP 工具调用等关键路径。

代码风格与静态检查使用 [ruff](https://github.com/astral-sh/ruff)：

```bash
ruff check app tests
```

## 📄 来源与许可

本项目是 [nageoffer/ragent](https://github.com/nageoffer/ragent)（Java / Spring Boot 4 生产级 Agentic RAG 平台）的 **Python 重构版**，遵循同一套 **Apache License 2.0**。原 Java 实现的完整版权归属原作者，特此致谢。

```
Copyright nageoffer

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
```
