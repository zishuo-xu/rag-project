# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Production-grade Chinese RAG (Retrieval-Augmented Generation) system. LangChain + FastAPI + Streamlit. LLM is any OpenAI-compatible endpoint (default DeepSeek); embedding (`BAAI/bge-small-zh-v1.5`) and reranker (`BAAI/bge-reranker-base`) are local sentence-transformers models — no API key needed for those. Docs/content are Chinese; README, comments, and eval reports are in Chinese.

## 项目目标定位(2026-07-28 对齐成果)

三层目标,后续工作以此为准:
- **L1 动机**:求职作品,需要深度的工程样板。
- **L2 产物**:教学样板,借"生产级"叙事外壳讲 RAG 工程故事。核心叙事 7 条;加固/并发/限流能演示讲清即可,不追 SLO,不上云。**诚实报告 + A/B 归因是叙事方式,但评测分数本身要拿得出手——目标是优秀水平,不是"能讲清就行"。**
- **L3 成功标准**:可演示 + 可复现——陌生人 clone 后能跑通服务+前端、能复现评测数字、依赖固化。达到此状态即停止迭代、写进简历。

**硬约束:性能与效果双向均衡。** 既不为效果忍非常慢,**也不为效率牺牲质量**——两个方向都是红线。任何改动要同时报两个维度的账:质量(检索 hit/coverage、F1、faithfulness)与效率(延迟、在线 LLM 调用数)。零 LLM 化只在"质量持平或仅微降"时成立;若某 LLM 调用能带来明显质量提升,应保留并想办法压延迟(缓存、条件触发、并行),而不是直接摘除。核心 7 条叙事以质量为先;非核心特性可以效率优先精简。

**核心叙事(进简历)**:F1 Autocut、F3+F8 忠实度、F7 引用溯源、F10 answer_extraction、Graph RAG、F13 Agentic、延迟治理。
**软摘(代码留、不主动讲)**:F10 self_consistency、多模型 qwen。
**特性去留判据**:教学样板下,弱数字不是包袱而是诚实素材——只有"非高频考点 + 无数据撑 + 制造叙事噪音"三者皆占才软摘。

**复现链(图谱用零 LLM)**:`build_cmrc_eval.py`(生成 162 篇语料+300 题)→ `ingest_cmrc_full.py`(灌 chroma/summary/parent-child/BM25)→ `rebuild_graph_fast.py`(零 LLM jieba 共现图,1 秒;`rebuild_graph_typed.py` 为 LLM 类型化可选高质量路径)→ `run_e2e_eval.py --mode full`。Graph 用零 LLM 共现图而非 LLM 类型化——后者重建 3.6h 且别人复现要付同样成本,与 L3「可复现」冲突,权衡回退(诚实记录:关系无类型化、实体含 jieba 噪声,但检索仍可用、1 秒可复现)。

## Commands

```bash
uv sync                                    # install deps (uses uv, lockfile committed)
cp .env.example .env                       # then fill OPENAI_API_KEY (service refuses to start without it)

python main.py                            # FastAPI on :8000 (reload) — main entry
streamlit run frontend/app.py             # UI on :8501

# Tests (pytest, asyncio via pytest-asyncio, pythonpath=".")
uv run pytest                             # full suite (~38 files / 336+ tests)
uv run pytest tests/test_pipeline.py      # single file
uv run pytest tests/test_pipeline.py::test_name   # single test
uv run pytest -k autocut                  # by keyword

# Evaluation (each writes data/*.json) — layered: zero-LLM (seconds) → 50-q slice (~4min) → full 300-q (~20min)
uv run python run_eval.py                 # RAGAS 4-dim quality (needs LLM)
uv run python run_retrieval_eval.py       # CMRC retrieval hit/coverage (zero LLM)
uv run python run_e2e_eval.py --dataset data/eval_slice_fast.json --gate --baseline data/eval_slice_baseline.json   # daily A/B (50 q)
uv run python run_e2e_eval.py --mode full        # 3-layer e2e: retrieval+faithfulness+zh F1/EM (300 q, release/baseline)
uv run python run_e2e_eval.py --mode baseline    # A/B: F1-F4 all off (RAG1.0 baseline)
uv run python run_e2e_eval.py --only F3          # single-feature attribution
uv run python run_e2e_eval.py --smoke --gate     # quality gate (used by pre-push hook)
uv run python run_e2e_eval.py --dataset data/eval_multihop.json --only F13  # agentic eval
uv run python run_concurrency_bench.py    # needs server running; hits real /api/chat

# Indexing / rebuild (after schema or graph changes)
uv run python scripts/rebuild_index.py
uv run python scripts/rebuild_graph_typed.py
```

Pre-push gate (opt-in): `git config core.hooksPath .githooks` enables `.githooks/pre-push`, which runs the `--smoke --gate` eval and blocks on regression. It skips automatically when no LLM key is present.

## Configuration

All config is in [config.py](config.py) via pydantic `Settings` (reads `.env`). Every RAG feature is an independent boolean switch with graceful degradation on failure — defaults are tuned for the latency-optimal path (most default-path features add **zero** online LLM calls). When adding/changing behavior, add a switch here rather than hardcoding.

Notable switch families:
- RAG 2.0 (retrieval/generation depth): `use_autocut` (F1), `use_iterative_retrieval` (F2), `use_faithfulness_check` (F3), `use_query_router` (F4), `use_contextual_chunks` / `use_decomposition` (F6a/F6b).
- RAG 3.0 (production): `use_citations` (F7), `use_speculative_streaming` (F8), `use_embedding_cache`/`use_rerank_cache` (F9), `use_answer_extraction`/`use_self_consistency` (F10), `api_key`/`rate_limit_rpm`/`log_json` (F11), `use_history_rewrite` (F12).
- Agentic: `use_agentic=False` (F13 ReAct state machine; falls back to 7-stage pipeline on error/empty evidence).

LLM construction is centralized in `build_chat_llm()` / `active_llm_config()` / `get_llm_extra_body()` — **always construct the chat LLM through these**, never instantiate `ChatOpenAI` directly. `llm_provider` switches between `deepseek` (uses `openai_*` config) and `qwen` (`qwen_*` config); thinking-mode handling differs per provider. Pass per-call `timeout`/`retries`/`max_tokens` explicitly — the 2026-07-26 latency-tuning relied on per-call differences and there is no shared default.

## Architecture

The system is layered `app/{ingestion, retrieval, generation, evaluation, api, observability}` orchestrated by `RAGChain` ([app/generation/chain.py](app/generation/chain.py)).

### Retrieval pipeline (the core)
[app/retrieval/pipeline.py](app/retrieval/pipeline.py) — `RetrievalPipeline` runs a fixed 7-stage pipeline:

```
gate → query transform → multi-recall (parallel, ThreadPool) → RRF fuse
     → cross-encoder rerank → CRAG grade → remediate (HyDE re-recall on poor grade)
```

- **Five recall channels** (`ALL_CHANNELS = dense, sparse, graph, parent_child, summary`) run in parallel via `recall_max_workers` thread pool. `remediate()` reuses `recall`/`fuse`/`rerank` so补救 results also get fused + reranked.
- **RRF fusion** ([fusion.py](app/retrieval/fusion.py)) merges channels by reciprocal rank; dedup by chunk id.
- **Autocut** ([autocut.py](app/retrieval/autocut.py)) Kneedle-knee truncation replaces fixed TopK (lower bound `autocut_min_docs`, upper bound = `retrieval_top_k`).
- **CRAG** ([crg.py](app/retrieval/crg.py)) grades `correct/ambiguous/incorrect/recovered`; gate decides whether to retrieve at all.
- **Query router** ([router.py](app/retrieval/router.py)) is rule-driven (zero LLM): numeric/comparative/multi_hop/conceptual/factual → adaptive depth + denoise.
- **Agentic** ([agent.py](app/retrieval/agent.py)) F13: hand-written ReAct loop (no LangGraph dep) over `search`/`decompose`/`grade` tools — agent decides decomposition itself, not via F4. max_steps=4 hard cap, degrades to 7-stage pipeline on failure. **Hard evidence gate** (`agentic_evidence_gate`, zero LLM): after search/decompose, accumulated evidence graded by CRAG (rerank sigmoid) — `correct` force-stops the loop (`stop_reason=evidence_sufficient`). Prompt signals alone couldn't control the LLM's stop decision (finish rate was stuck <50%); the gate took it to ~93% on the multihop slice. Graph entity extraction in the decompose path is `fast_only` (`decompose_graph_fast_only`) — zero-LLM node matching, no LLM fallback, since subqueries multiply the fallback cost.

### Latency budget
[app/retrieval/deadline.py](app/retrieval/deadline.py) — `Deadline` gives each query a budget (`latency_budget_ms`, default 25s) and circuit-breaks optional stages (F2 iteration, F3 regen) on outlier overrun. The `Deadline` is carried on `RetrievalResult` so the generation layer reuses it. `answer_max_tokens` caps generation length. This is first-class — respect budget checks when adding latency-sensitive stages.

### Generation
[app/generation/chain.py](app/generation/chain.py) `RAGChain` = orchestration: semantic response cache → retrieval → generation, plus:
- **F8 speculative streaming** ([streaming.py](app/generation/streaming.py)): stream tokens immediately for low TTFT, run faithfulness check at stream end, append `correction` event if unfaithful — this is the fix for F3's streaming degradation.
- **F3 faithfulness** ([faithfulness.py](app/generation/faithfulness.py)): LLM-judge per-claim grounding; unfaithful → strict prompt bounded regen (single fact source).
- **F7 citation** ([citation.py](app/generation/citation.py)): split answer into claims, link to source chunks via embedding cosine (zero online LLM).
- **F10 answer boost** ([answer_boost.py](app/generation/answer_boost.py)): zero-LLM short-answer span extraction + adaptive self-consistency — the fix that lifted end-to-end EM from 0.
- **F12** ([conversation.py](app/retrieval/conversation.py)): history-aware query rewrite (zero-LLM heuristic by default).

### Ingestion
[app/ingestion/service.py](app/ingestion/service.py) orchestrates load → chunk → index → enhanced indexes → BM25 increment. Enhanced indexes (Parent-Child, contextual chunks) **degrade to warnings, never block the main index**. Contextual chunking runs LLM at index time only (zero online increment). Graph extraction ([graph_extractor.py](app/ingestion/graph_extractor.py)) emits typed triples (person/work/place/org/position/event/other) with chunk_id provenance; persisted to `data/knowledge_graph.json`, graph retrieval uses `graph:` prefix to avoid RRF overlap with real chunks.

### API & security
[main.py](main.py) builds the FastAPI app; routes in [app/api/routes.py](app/api/routes.py) (`/api/chat` SSE streaming, `/api/documents/upload`, `/api/evaluate`, `/api/health`). Concurrency is gated by an `asyncio.Semaphore` (`max_concurrent_requests`) with 503 on queue timeout (`request_queue_timeout`). F11 security middleware (API key + rate limit) only registers when `api_key`/`rate_limit_rpm` are set. Metrics at `/api/metrics` (Prometheus + JSON).

### Observability
[app/observability/](app/observability/) — in-process counters/histograms (`metrics.py`) and tracing (`tracing.py`). `RetrievalResult` / `RAGResponse` carry rich telemetry fields (crag_grade, iterations_used, pre_autocut_count, query_type, budget_skipped, agent_steps, etc.) — extend these dataclasses rather than adding ad-hoc logging when surfacing new signals.

## Conventions

- **Honest reporting**: eval reports in `docs/superpowers/reports/` deliberately record negative/null results as open items (e.g. EM=0, finish-rate <50%, no-significant-change). Match this tone — state what the data shows, including non-improvements, rather than rounding up.
- **Every feature degrades gracefully**: wrap new optional behavior in its switch, catch exceptions, fall back to baseline behavior, log. Never let an optional feature 500 the request.
- **Zero-online-LLM default**: prefer index-time LLM work, caches, and rule/heuristic logic over per-query LLM calls; the default path must stay latency-neutral.
- Tests are colocated in [tests/](tests/) and run offline against fakes/stubs (HF offline env vars set in main.py). Many tests assert on the telemetry fields above.
