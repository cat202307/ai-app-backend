# AI 知识库问答与智能 Agent 系统

基于 **FastAPI + LangGraph + ChromaDB + DeepSeek + Vue 3** 的知识库问答与智能 Agent 系统。
后端提供 Agent 编排、Advanced RAG 检索与 SSE 流式接口，前端采用 Vue 3 + Vite 工程化开发，
生产环境可由 FastAPI **单端口**直接服务，无需独立部署前端。

---

## ✨ 功能特性

### 核心能力
- 🤖 **智能 Agent** — 基于 LangGraph 的多工具自主决策与条件路由
- 📚 **Advanced RAG** — 混合检索（bge-m3 + BM25）+ RRF 融合 + Cross-Encoder Rerank + 引用溯源
- 💬 **真流式输出** — 后端边生成边通过 SSE 推送，首字延迟低，回答实时逐字显示
- 📁 **文件管理** — 上传自动建索引、删除、重建索引
- 📊 **系统监控** — 性能指标、接口耗时、运行时长，自动刷新

### Agent 与 RAG
- 🔧 **多工具调用** — 自动选择合适工具，记录完整 Trace（参数 / 结果 / 耗时）
- 🧠 **LangGraph 工作流** — StateGraph + 条件边，执行链路清晰可观测
- 🔍 **混合检索** — bge-m3 向量检索 + 关键词检索（jieba 分词 + BM25），RRF 倒数排名融合
- 🧩 **父子分块（Small-to-Big）** — 结构感知子块（~200 字）负责精准 embedding / rerank，命中后实时聚合父块（~4 子块）交给 LLM，兼顾「检索精准」与「上下文完整」
- ⚡ **低延迟检索** — 查询改写按问题特征条件触发（明确问题跳过、宽泛问题保留）并缓存结果，避免不必要的 LLM 往返；启动时预热本地模型，消除首次问答冷启动
- 🎯 **Cross-Encoder Rerank** — 本地 bge-reranker-v2-m3 按真实相关性重排序，零 Token、可复现
- 📎 **引用溯源** — 回答标注 `[n]` 来源，前端展示「参考来源」面板
- 🌐 **全局检索** — 用户未点名文档时自动跨全部文档做 Advanced RAG，引用各自标注来源文件；删除文档同步清理向量，避免孤儿数据
- 📑 **多文档概述全覆盖** — 无主语概述提问（“主要内容是什么？”）自动按 `## 《文件名》` 分段逐一概述每个文档，开头说明文档总数；全局概述时每文档只取前 3 个开头父块平衡上下文（可用 `GLOBAL_OVERVIEW_OPENING_CHUNKS` 调整），不会只回答排名靠前的大文档
- 💾 **多级缓存** — Embedding 缓存、检索缓存，显著降低延迟

### 前端体验（Vue 3）
- ⚡ **工程化** — Vue 3 `<script setup>` + Vite，组件化（聊天 / 文件 / 监控）
- 📚 **参考来源面板** — 展示来源文件、Chunk 与内容，悬停展开全文
- 🗂️ **多会话管理** — 服务端按 `session_id` 持久化，历史按会话分组，刷新自动恢复最近会话
- 🛠️ **消息操作栏** — 复制、工具信息、调用次数、耗时与时间戳
- 📥 **导出对话** — 一键导出
- 📱 **响应式布局** — 适配不同屏幕尺寸

---

## 🛠️ 技术栈

| 层级 | 技术 |
|------|------|
| **后端框架** | FastAPI + Uvicorn（纯 ASGI 中间件，兼容 SSE） |
| **Agent 框架** | LangGraph（StateGraph / 条件边） |
| **大模型** | DeepSeek API（`deepseek-v4-pro`） |
| **向量数据库** | ChromaDB（持久化） |
| **Embedding** | BAAI/bge-m3（多语言，1024 维稠密向量） |
| **Rerank** | BAAI/bge-reranker-v2-m3（Cross-Encoder，本地推理） |
| **关键词检索** | jieba 分词 + BM25（rank_bm25） |
| **关系数据库** | SQLite + SQLAlchemy |
| **前端** | Vue 3 + Vite（Markdown 渲染：marked） |
| **部署** | 单端口静态托管 / Docker + docker-compose |
| **测试 / 评测** | pytest / Agent 评测 / 检索评测（Recall·MRR·nDCG）/ 答案质量评测（忠实度·相关性·拒答） |

---

## 🚀 快速开始（单端口）

### 环境要求
- Python 3.10+
- Conda（推荐）或 venv
- Node.js 18+（仅构建/开发前端时需要）
- DeepSeek API Key

### 1. 创建并激活环境
```powershell
conda create -n ai-app python=3.10
conda activate ai-app
```

### 2. 安装后端依赖
```powershell
pip install -r requirements.txt
```

### 3. 配置环境变量
复制 `.env.example` 为 `.env`，填入 DeepSeek API Key：
```env
DEEPSEEK_API_KEY=your_api_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-pro
```

### 4. 构建前端（首次 / 前端有改动时）
```powershell
cd frontend
npm install --registry=https://registry.npmmirror.com
npm run build
cd ..
```
> 构建产物输出到 `frontend/dist/`，由 FastAPI 在根路径服务。

### 5. 启动后端
```powershell
python -m uvicorn main:app --reload
```
> `main.py` 已在启动时设置 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`，
> 避免 Embedding 模型联网检查导致卡顿。

### 6. 访问（仅需 8000 一个端口）
- **应用首页（Vue SPA）**：http://localhost:8000/
- **旧版 HTML（保留参考）**：http://localhost:8000/chat
- **Swagger 文档**：http://localhost:8000/docs
- **性能监控页**：前端「系统监控」或 http://localhost:8000/metrics

> 若尚未构建前端，访问 `/` 会返回后端欢迎 JSON，按第 4 步构建即可。

---

## 🧑‍💻 前端开发模式

前端支持「开发热更新」与「生产构建」两种模式：

```powershell
cd frontend

# 安装依赖（首次）
npm install --registry=https://registry.npmmirror.com

# 开发模式：启动 Vite Dev Server（http://localhost:3000）
# 已在 vite.config.js 配置 /agent /upload /history /metrics /files /static 代理到 8000
npm run dev

# 生产构建：产物输出到 frontend/dist，供 FastAPI 单端口服务
npm run build
```

| 模式 | 地址 | 说明 |
|------|------|------|
| 开发 | http://localhost:3000 | 需同时运行后端（8000），改动即时热更新 |
| 生产 | http://localhost:8000 | 仅运行后端，前端已整合，适合演示 / 部署 |

### 前端目录
```
frontend/
├── vite.config.js        # Vite 配置 + 开发代理
├── package.json
├── index.html
└── src/
    ├── main.js           # Vue 应用入口
    ├── App.vue           # 主界面：侧边栏 + 聊天 + 会话管理
    ├── style.css
    ├── assets/           # 图片等静态资源
    └── components/
        ├── FilesPage.vue     # 文件管理页（列表 / 上传 / 删除 / 重建索引）
        └── MonitorPage.vue   # 系统监控页（指标卡片 + 自动刷新）
```

---

## 📁 项目结构

```
ai-app-backend/
├── main.py                 # FastAPI 入口（中间件 / 路由注册 / SPA 挂载）
├── models.py               # SQLAlchemy ORM 模型
├── schemas.py              # Pydantic 请求/响应模型
├── database.py             # 数据库连接
├── requirements.txt
├── scripts/
│   ├── rebuild_vector_index.py   # 更换 Embedding 后删除旧 collection 并重建全部索引
│   └── rebuild_all_indexes.py    # 重建所有已上传文件的向量索引
│
├── agents/                 # Agent 模块
│   ├── graph_agent.py      # LangGraph 工作流
│   ├── loop.py             # 手写 Loop（对比验证）
│   ├── registry.py         # 工具注册中心
│   ├── runner.py           # 单工具调用
│   ├── executor.py         # 工具执行器
│   ├── base_tool.py        # 工具基类
│   └── tools/              # 具体工具
│       ├── advanced_rag_search.py   # Advanced RAG 问答（含引用）
│       ├── document_info.py         # 文档信息查询
│       ├── knowledge_search.py      # 知识库检索
│       └── list_documents.py        # 列出所有文档
│
├── routers/                # API 路由
│   ├── agent.py            # Agent 接口（含 SSE 流式）
│   ├── chat.py             # 普通聊天
│   ├── rag.py              # RAG 接口
│   ├── upload.py           # 文件上传 / 重建索引
│   └── history.py          # 历史记录
│
├── services/               # 业务服务层
│   ├── llm_service.py                  # LLM 调用
│   ├── prompt_service.py               # 各场景 Prompt 构建
│   ├── rag_service.py                  # 基础 RAG
│   ├── advanced_rag_service.py         # Advanced RAG 检索 / 融合 / 重排
│   ├── advanced_rag_answer_service.py  # RAG 问答与阈值门控
│   ├── chunking_service.py             # 父子块结构感知切分
│   ├── vector_service.py               # 向量服务 + 缓存
│   ├── citation_service.py             # 引用构建 / 裁剪
│   └── context_service.py              # 上下文管理
│
├── core/                   # 核心基础设施
│   ├── config.py           # 配置（.env 覆盖默认值）
│   ├── logging_config.py   # 结构化日志
│   ├── metrics.py          # 性能指标
│   └── exceptions.py       # 异常定义
│
├── frontend/               # Vue 3 前端工程（见上）
├── static/                 # 旧版单文件 HTML（/chat，保留参考）
│
├── evaluation/             # Agent 评测 + 检索评测
│   ├── agent_eval_dataset.json
│   ├── run_agent_evaluation.py
│   ├── retrieval_dataset.json   # 检索评测集（问题 + 人工标注 gold chunks）
│   ├── retrieval_eval.py        # Recall / Precision / MRR / nDCG
│   └── results/
│
├── tests/                  # 单元测试（pytest）
│
├── uploads/                # 上传文件目录
├── vector_db/              # ChromaDB 持久化
├── ai_app.db               # SQLite 数据库
│
├── Dockerfile
├── docker-compose.yml
└── DOCKER.md               # Docker 部署指南
```

---

## 🔌 API 接口

### Agent
| 接口 | 方法 | 说明 |
|------|------|------|
| `/agent/tools` | GET | 查看已注册工具 |
| `/agent/run` | POST | 手写 Loop 版本 |
| `/agent/graph-run` | POST | LangGraph 版本（返回完整 Trace） |
| `/agent/graph-run-stream` | POST | **流式 Agent（SSE）** |
| `/agent/compare` | POST | 对比两个版本 |

**SSE 事件顺序**（`graph-run-stream`）：
`meta` → `step`（每个工具调用）→ `answer_chunk`（正文逐块）→ `citations`（参考来源）→ `done`

**真流式实现**：LangGraph 在后台线程运行，agent 节点用 `stream=True` 调用 LLM，正文 token 与工具调用步骤经线程安全队列实时桥接到 SSE；工具调用轮通过系统提示约束保持 `content` 为空（不输出“我需要先……”独白），推给前端的只有最终答案。

**检索职责拆分（降延迟关键）**：`ask_advanced_rag` 工具只做检索与阈值门控、返回带来源编号的原文片段，**不再在工具内部生成答案**；最终回答由 Agent 的流式 LLM 基于片段一次性合成。一次问答从原先 3 次串行 LLM（改写 + 工具内答案 + 最终复述）缩减为 **1 次 LLM**。

**查询改写：条件门控 + 缓存**。早期 14 问检索集对比发现，强语义 Embedding 下对「明确问题」做 LLM 改写只引入噪声（rewrite=0 的 Recall@3=0.773 优于 rewrite=2 的 0.727），且每次改写是一次约 3.3s 的 LLM 往返。因此系统不是简单开关，而是：
- **明确问题（含技术锚点、无指代）**：跳过 LLM 改写、直接用原问题检索（实测改写阶段平均耗时 3348ms → 0.01ms，单次问答省约 3.3s）；
- **宽泛 / 口语 / 含指代问题**：保留 LLM 改写以扩展召回；
- **改写结果按问题缓存**，相同问题不重复调用 LLM。

**第一轮强制检索**：为避免模型在上下文已含相关信息时跳过工具、凭记忆直接作答（会丢失参考来源），Agent 第一轮使用 `tool_choice="required"` 强制调用工具，拿到片段后的后续轮才用 `"auto"` 自由合成；因此每条文档类回答都稳定带参考来源（需在关闭 thinking 模式下，thinking 模式不支持 required）。

**端到端延迟（CPU、模型预热后，改写门控优化后实测）**：

| 问题 | 首字延迟 | 总耗时 |
|------|---------|--------|
| 论文用了什么优化器 | 约 6.5s | 约 7.3s |

> 首字延迟中约 1.3s 为 Cross-Encoder 的 CPU 重排，其余为向量 / 关键词检索与 LLM 首 token；
> 相比改写门控优化前（首字约 9.9s）下降约 3.4s。启动时已预热本地模型。

### 文件
| 接口 | 方法 | 说明 |
|------|------|------|
| `/upload` | POST | 上传文件（自动建索引） |
| `/upload/files` | GET | 文件列表 |
| `/upload/rebuild-index` | POST | 重建指定文件索引 |
| `/upload/{id}` | DELETE | 删除文件 |

### RAG / 历史 / 监控
| 接口 | 方法 | 说明 |
|------|------|------|
| `/rag/index/{file_id}` | POST | 建立向量索引 |
| `/rag/advanced/ask` | POST | Advanced RAG 问答 |
| `/history/sessions` | GET | 会话列表（按 session 分组） |
| `/metrics` | GET | 性能指标（HTTP 统计 + 检索分段 `retrieval_stages` + LLM 用量 `llm_usage`） |

---

## 🧪 测试

```powershell
# 全部测试
pytest -q

# 单个模块
pytest tests/test_tools.py -v
```

当前 **115 个测试用例全部通过**（约 1.2s 跑完），覆盖：
父子块结构感知切分、追问改写判定与用途指代、查询改写门控 / 缓存、
Agent 工具纠偏、父块提升、metrics 指标、健康检查、Prompt、
工具系统与向量缓存。

> 补测试的过程中还顺带发现并修复了 2 个真实 bug：`main.py` 漏 import
> `FileResponse` 导致根路径 500；`build_history_text` 在未传当前问题时
> 误删最后一条用户消息。

---

## 📊 Agent 评测

```powershell
python evaluation/run_agent_evaluation.py
```

最新结果：
- **工具选择准确率**：75%（6/8）
- **完成率**：100%
- **平均耗时**：约 26s
- **平均 Token**：约 3975

> 检索门控采用「宁可告知信息不足、也不编造」的策略：当文档仅部分覆盖问题时，
> 回答会引用现有片段说明「文档提到了什么」，并明确指出「未提供所问细节」。

---

## 🔎 检索质量评测

```powershell
python evaluation/retrieval_eval.py
```

对检索四个阶段（向量 / 关键词 / RRF 融合 / Cross-Encoder 重排）计算
**Recall@k、Precision@k、MRR、nDCG@k**（k=3、5）。评测集含 14 个问题、人工标注 gold chunk，
纯本地运行、不调用 LLM，结果写入 `evaluation/results/retrieval_evaluation.json`。

最新结果（bge-m3，@5）：

| 阶段 | Recall | MRR | nDCG |
|------|--------|-----|------|
| 向量（bge-m3） | 0.821 | 0.756 | 0.721 |
| 关键词（BM25） | 0.893 | 0.756 | 0.760 |
| RRF 融合 | 0.893 | 0.827 | **0.811** |
| Cross-Encoder 重排 | 0.857 | **0.845** | 0.776 |

> 升级 bge-m3 后，向量路 MRR@3 由 0.226 提升至 0.738；RRF 融合 MRR@5 由 0.583 提升至 0.827。
> 升级前（MiniLM）历史结果见 `evaluation/results/retrieval_evaluation_minilm.json`。

---

## 🧾 端到端答案质量评测（LLM-as-a-Judge）

检索指标只衡量“找得准不准”，无法反映最终答案文本是否忠实、是否切题。
为此新增轻量级 LLM-as-a-Judge（裁判为 DeepSeek，`temperature=0`、严格 JSON 输出），
不引入 RAGAS 重依赖，量化两项指标：

- **Faithfulness（忠实度）**：把答案拆成原子事实陈述，逐条判定能否由检索上下文支撑，
  并用 claims 现算支撑比例（不直接采信模型自报值）；
- **Answer Relevancy（答案相关性）**：答案是否切题、完整、无冗余（0–1）。

```powershell
# 文档内问题：答案质量（忠实度 / 相关性）
python evaluation/run_answer_quality.py
# 文档外问题：拒答压力测试
python evaluation/run_abstention_test.py
```

最新结果：

| 评测 | 结果 |
|------|------|
| 文档内 14 问 忠实度 / 相关性 | **1.00 / 1.00**（183 条陈述全部被上下文支撑，零幻觉） |
| 拒答压力测试（6 个文档外问题） | 修复前 50% → 修复后 **100%（6/6）** |

> **拒答测试为什么必要**：全是“文档内有答案”的评测即使满分，也检验不了
> “文档没有答案时会不会乱编”。首次拒答测试发现系统虽在开头识别“文档没有”，
> 却仍用自身通用知识补出了 RRF 公式、2024 欧洲杯冠军、红烧肉做法（正确率仅 50%）。
> 两层根因与修复：
> 1. **门控阈值过低**：实测无关问题片段 rerank 分集中在 50.0–50.5、相关片段为 60–72，
>    遂将 `RAG_RERANK_MIN_SCORE` 由 30 提至 **55**（与两侧各留约 5 的 margin），
>    无关问题在检索层即判定 not_matched、citations 为空；
> 2. **提示词新增规则 11**：文档无答案时只能回复“未找到”，
>    即使自身知道答案，也严禁补充任何文档外信息（不得出现“基于通用知识……”）。
>
> **补充：概述类问题的专用路由**。阈值 55 会误杀“该论文主要内容是？”这类
> 概述问题——它们不指向某个具体 chunk，rerank 分仅约 50，与无关问题重叠，
> 单一阈值无法区分。因此检索层新增**概述路由**：识别“主要内容 / 讲了什么 /
> 摘要 / 概述 / 简介”类问题时，直接返回文档**开头的 chunk**（摘要、引言通常在最前），
> 绕过 rerank 阈值。最终形成三路各司其职：**具体问题**走 rerank 高分、
> **概述问题**走文档开头 chunk、**无关问题**被阈值拦截后拒答。
>
> **补充：多文档全局概述**。知识库存在多个文档时，模型可能把“该论文主要内容”
> 调用中的 file_id 误填为第一个文档，导致只概述第一篇。工具层做了硬处理：
> 概述类问题**未在问题中明确点名某个文档**（未出现文件名关键词或“文档N / 文件N”）
> 时，一律按全局概述处理——取**每个文档**的开头 chunk，并要求模型**逐篇分别概述**
> 每个文档（文档《A》主要讲……；文档《B》主要讲……）；只有明确点名
> （如“test02 主要内容 / 胡家浩那篇论文 / 文档2”）时才限定单个文档。
>
> 同时，概述识别覆盖“主要是什么 / 主要是啥 / 讲的是什么”等说法，并精确区分
> 三类、避免误伤：① **「文件名 + 主要是什么」**（test02.txt主要是什么？）→ 仅概述
> 该文档；② **「文档指代 + 概述词」**（该论文主要内容是？）→ 全局逐篇概述；
> ③ **「具体概念 + 是什么」**（AdamW是什么？）→ 走 rerank 正常问答，不被概述路由接管。
>
> **补充：多轮追问的指代 / 省略消解**。多轮对话中用户常做承前省略的追问
> （如先问“FastAPI是什么”，再问“有什么用”）。系统在执行检索工具前，对极短 /
> 含指代词（它、这、那、呢、有什么用……）且缺少明确主体的问题，结合对话历史
> 改写成独立完整问题（“FastAPI 有什么用、用途、应用场景”）再检索；含明确实义
> 主体（如 AdamW、RRF）的问题不改写。这样省略追问也能被正确理解，而不是被当作
> 无关问题拒答。若省略式追问**完全没有对话历史可参照**（如新对话只发"有什么用"），
> 系统不会编造答案，而是引导用户补充具体指代对象。
>
> **补充：用途追问必须指代具体对象，禁止「文档」兜底**。当历史一次性介绍了多个对象
> （如概述 FastAPI、RAG、Hybrid Search），再追问「有什么用」时，改写模型一度偷懒
> 生成「test02 文档主要讲的这些内容有什么用？」这种元层面问题，原文没有对应内容、
> 导致误拒。改写规则因此明确：补全对象**必须是历史里实际出现的技术实体**，严禁
> 「这份文档 / 文档内容 / 这些内容 / 本文」等笼统指代；多对象时改写为「FastAPI、RAG、
> Hybrid Search **分别**有什么用？」，确保每个对象的用途描述都能被检索命中。
>
> **补充：依据「对象角色」答用途，避免字面误拒**。还有一种更隐蔽的误拒：先问
> 「论文用了什么优化器」（得知 AdamW），再追问「它有什么用」。此时片段明确写了
> 「采用 AdamW」作为优化器并给出配置，却没有「作用是……」的字样，模型一度按字面
> 判定「片段不含答案」而拒答。系统提示因此区分两种情况：片段**完全没提到对象**才拒答；
> 只要片段明确了对象的角色 / 用法，就应据此说明它在本文中的作用（「AdamW 用作训练
> 优化器，更新网络参数、最小化损失」）并引用配置。对象角色固有的基本职责（优化器更新
> 参数、损失函数量化误差、调度器调整学习率）属于组织答案所必需的语义衔接、允许使用，
> 但不得扩展为片段未涉及的原理推导或优缺点对比，从而既不误拒、也不越界。

---

## 🧩 父子分块（Small-to-Big / Parent-Child）

**为什么需要它。** 检索对 chunk 大小有两个互相矛盾的要求：
- chunk **越小、主题越聚焦**，embedding 与 cross-encoder 打分越精准（大 chunk 里的无关内容会稀释语义、拉低 rerank 分）；
- 但 chunk 太小，LLM 拿到的上下文不完整（如编号列表被切断、缺少前后铺垫），答案容易断章。

固定大小切分只能二选一；父子分块让两者解耦：

| 单元 | 大小 | 职责 | 是否入库 |
|------|------|------|----------|
| **子块 child** | ~200 字，结构感知切分（段落 / 编号点边界，超长再硬切） | 生成 embedding、参与向量 / 关键词检索与 rerank 打分 | 是 |
| **父块 parent** | ~4 个子块（约 800 字） | 命中后聚合完整上下文交给 LLM | 否，按子块实时聚合 |

**检索流程：**
1. 问题对子块做向量 + 关键词检索、RRF 融合、rerank 打分；
2. 保留 rerank ≥ 阈值的子块；
3. 把子块提升到所属父块（同父多子块去重、保留最高分）；
4. 实时聚合每个父块下的子块原文，以父块为单元构建引用、交给 LLM。

**索引结构（ChromaDB collection `documents`）：**
- 子块 id：`file_{file_id}_child_{child_index}`；
- metadata：`file_id` / `filename` / `chunk_index`(=child_index) / `child_index` / `parent_id` / `parent_index`；
- 父块 id：`file_{file_id}_parent_{parent_index}`（不入库，`get_parent_content` 按 child_index 升序实时聚合）。

> 切分逻辑见 `services/chunking_service.py`，索引进 / 出见 `services/vector_service.py`，
> 父块提升见 `advanced_rag_answer_service.promote_children_to_parents`。

---

## 🔁 更换 Embedding 模型 / 重建索引

不同 Embedding 的向量维度不同（如 MiniLM 为 384 维、bge-m3 为 1024 维），
ChromaDB collection 的维度在首次写入时固定，更换模型后旧 collection 不兼容，需整体重建：

```powershell
# 1.（可选）首次使用经国内镜像预下载模型
set HF_ENDPOINT=https://hf-mirror.com

# 2. 删除旧 collection，对全部文档重新切分、编码、建立索引
python scripts/rebuild_vector_index.py

# 3. 重启后端，并重跑检索评测确认提升
python evaluation/retrieval_eval.py
```

> 当前采用**父子分块**（见下文「父子分块」小节）：子块约 200 字、结构感知切分，
> 父块约含 4 个子块、不入库、命中后实时聚合。重建后必须重启后端，
> 因为运行中的进程仍持有被删除的旧 collection 对象。

---

## ⚙️ 配置说明

主要配置项（`.env`，覆盖 `core/config.py` 默认值）：

| 配置项 | 当前值 | 说明 |
|--------|--------|------|
| `DEEPSEEK_API_KEY` | — | DeepSeek 密钥 |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | API 地址 |
| `DEEPSEEK_MODEL` | `deepseek-v4-pro` | 模型名 |
| `AGENT_MAX_STEPS` | `5` | Agent 最大执行步数 |
| `RAG_MAX_DISTANCE` | `19.0` | 向量检索最大距离（越大越宽松） |
| `RAG_RERANK_MIN_SCORE` | `50` | Rerank 通过阈值；50 兼顾召回与相关性，对比问题另有文件覆盖回填 |
| `RAG_CHILD_CHUNK_SIZE` | `200` | 子块目标字数（结构感知切分） |
| `RAG_CHILDREN_PER_PARENT` | `4` | 每个父块聚合的子块数（父块约 800 字） |
| `RAG_QUERY_REWRITE_COUNT` | `1` | 额外改写 query 数；明确问题门控自动跳过，`0`=关闭；对比/指代/宽泛问题自动提升到 2 |
| `RRF_K` | `60` | RRF 排名衰减常数 |
| `RRF_KEYWORD_WEIGHT` | `5.0` | Hybrid Search 中关键词检索的权重倍数（强化精确术语命中） |
| `LLM_PRICE_PER_MILLION_INPUT` | `2.0` | 输入单价（元 / 百万 token，用于 `/metrics` 成本估算，请按实际模型改） |
| `LLM_PRICE_PER_MILLION_OUTPUT` | `8.0` | 输出单价（元 / 百万 token，同上） |

> **修改 `.env` 后必须完全重启 uvicorn 才生效**（`--reload` 不监听 `.env`）；
> 修改 `.py` 则由 `--reload` 自动重载。

---

## 🐳 Docker 部署

详见 [DOCKER.md](./DOCKER.md)。

```bash
docker-compose up -d      # 构建并启动
docker-compose logs -f    # 查看日志
docker-compose down       # 停止
```

---

## 📝 开发指南

### 新增工具
1. 在 `agents/tools/` 新建文件，继承 `BaseTool`
2. 实现 `name`、`description`、`input_model`、`execute`
3. 在 `agents/registry.py` 注册
4. 在 `services/prompt_service.py` 的系统提示中说明使用规则

### 新增路由
1. 在 `routers/` 新建文件并定义 `APIRouter`
2. 在 `main.py` 中 `include_router`（置于 SPA 挂载之前）

---

## 🗺️ 后续 Roadmap

- [x] 未点名文档时，自动跨全部文档检索（Advanced RAG 全局模式）
- [x] Docker 镜像内整合前端构建，实现开箱即用单容器部署
- [x] 回答「重新生成」按钮
- [x] 关键词检索升级（jieba 分词 + BM25）
- [x] Rerank 换本地 Cross-Encoder（bge-reranker-v2-m3）
- [x] Embedding 升级 bge-m3，并搭建检索评测集（Recall / MRR / nDCG）
- [x] 端到端答案质量评测（LLM-as-a-Judge：忠实度 / 相关性，14 问均 100%）
- [x] 拒答压力测试与幻觉治理（门控阈值 55 + 提示词规则 11，拒答 50%→100%）
- [ ] 深色模式
- [ ] 引用片段定位到最相关句子（当前展示 Chunk 开头）

---

## 📄 License

MIT
