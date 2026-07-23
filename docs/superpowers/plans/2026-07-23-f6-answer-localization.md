# F6 答案定位增强 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让「该被用到的答案真正进入送给 LLM 的 top 上下文」——通过 F6a（接线 Parent-Child + 上下文增强分块）修复「源命中但答案不在 top 块」，通过 F6b（多跳查询分解）让多跳问题被拆开分别检索再合并。

**Architecture:** F6a 在摄入侧把 Parent-Child 死特性接进上传/重建流程，并新增索引时一次性 LLM 上下文增强（写入独立 `chunks_contextual` collection，经 `HierarchicalIndexer.detail_store` 按开关切换查询目标，零在线 LLM 增量）。F6b 在 `QueryTransformer` 增加 `decompose`，在 `RetrievalPipeline.run` 的多跳分支里把问题拆成子问题（并行优先、依赖时链式），轻量召回后 RRF 合并，再走既有 rerank/autocut/CRAG。评估新增 `answer_in_top_context` 指标与多跳/细粒度切片。

**Tech Stack:** Python 3.11, FastAPI, LangChain, ChromaDB, pytest + unittest.mock, uv, DeepSeek（OpenAI 兼容）。

## Global Constraints

- 每个新特性独立配置开关，**默认开**；关闭后行为与现状完全一致（保护既有 126 个测试）。
- 任何新模块异常**优雅降级**到原行为，绝不阻断主管道。
- **F6a 零在线 LLM 增量**（上下文生成仅索引时）；**F6b 仅多跳查询**触发额外 LLM。
- 检索命中率（CMRC）必须**保持 100%** 作为回归门禁。
- 测试全离线（mock LLM / 假 embedding / 临时 Chroma 目录），`uv run pytest` 可跑。
- `.env` 绝不提交；每次 commit 前 `git ls-files --error-unmatch .env` 应报错（未跟踪）。
- 新开关在 `tests/test_pipeline.py` 的 mock 里默认关闭（沿用 F1-F5 回归保护模式）。

---

## File Structure

| 文件 | 动作 | 职责 |
|------|------|------|
| `config.py` | Modify | 新增 F6a/F6b 配置开关 |
| `app/ingestion/contextual.py` | Create | 上下文生成器（索引时一次性 LLM，失败降级裸块） |
| `app/ingestion/indexer.py` | Modify | `contextual_store` / `detail_store` 属性、`index_documents_contextual`、`search_chunks` 切换 |
| `app/retrieval/dense.py` | Modify | 用 `indexer.detail_store` 替代直接 `chunk_store` |
| `app/generation/chain.py` | Modify | `rebuild_parent_child_index()` 重建辅助方法 |
| `app/api/routes.py` | Modify | 上传接线 Parent-Child + `POST /api/documents/reindex` 重建端点 |
| `app/retrieval/query_transform.py` | Modify | `Decomposition` dataclass + `DECOMPOSE_PROMPT` + `decompose()` |
| `app/retrieval/pipeline.py` | Modify | `RetrievalResult` 新字段 + `_retrieve_subquery` + `_decompose_retrieve` + `run()` 多跳分支 |
| `app/evaluation/metrics.py` | Modify | `answer_in_top_context()` 纯函数 |
| `run_e2e_eval.py` | Modify | `FEATURE_FLAGS` 加 F6、`eval_sample`/`aggregate` 新指标、`--slice` |
| `data/eval_multihop.json` | Create | 多跳测试集（schema + 构造脚本产物） |
| `scripts/build_multihop_eval.py` | Create | 半自动构造多跳测试集的辅助脚本 |
| `tests/test_config_f6.py` | Create | 配置默认值 |
| `tests/test_contextual_chunking.py` | Create | 上下文生成 + 索引 + detail_store 切换 |
| `tests/test_parent_child_wiring.py` | Create | 上传接线 + 重建 |
| `tests/test_decomposition.py` | Create | decompose 正常/异常/裁剪/链式 |
| `tests/test_pipeline_multihop.py` | Create | 多跳分支并行/链式/退化/关闭 |
| `tests/test_e2e_metrics.py` | Modify | 追加 `answer_in_top_context` 测试 |
| `README.md` / `docs/architecture.md` / `docs/interview_guide.md` | Modify | 同步 F6 文档 |

---

# Phase A — F6a 细粒度召回 + 上下文增强

## Task 1: F6 配置开关

**Files:**
- Modify: `config.py:76`（在 F4 `use_query_router` 之后插入 F6 段）
- Test: `tests/test_config_f6.py`

**Interfaces:**
- Produces: settings 字段 `use_contextual_chunks: bool`、`contextual_max_chars: int`、`chroma_contextual_collection: str`、`use_decomposition: bool`、`decomposition_max_subquestions: int`、`decomposition_max_hops: int`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_config_f6.py
"""F6 配置开关默认值"""
from config import Settings


def test_f6a_defaults():
    s = Settings()
    assert s.use_contextual_chunks is True
    assert s.contextual_max_chars == 80
    assert s.chroma_contextual_collection == "chunks_contextual"


def test_f6b_defaults():
    s = Settings()
    assert s.use_decomposition is True
    assert s.decomposition_max_subquestions == 4
    assert s.decomposition_max_hops == 3
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/test_config_f6.py -v`
Expected: FAIL（`AttributeError: 'Settings' object has no attribute 'use_contextual_chunks'`）

- [ ] **Step 3: 实现配置**

在 `config.py` 的 `use_query_router: bool = True`（第 76 行）之后插入：

```python
    # F6 答案定位增强（每项独立开关，异常均优雅降级到原行为）
    # F6a 细粒度召回 + 上下文增强（零在线 LLM 增量）
    use_contextual_chunks: bool = True            # 查询时使用上下文增强嵌入集合
    contextual_max_chars: int = 80                # 上下文片段长度上限
    chroma_contextual_collection: str = "chunks_contextual"
    # F6b 多跳查询分解（仅多跳查询触发；并行优先，依赖时链式）
    use_decomposition: bool = True
    decomposition_max_subquestions: int = 4       # 子问题数上限
    decomposition_max_hops: int = 3               # 链式分解跳数硬上限
```

- [ ] **Step 4: 运行确认通过**

Run: `uv run pytest tests/test_config_f6.py -v`
Expected: PASS（2 passed）

- [ ] **Step 5: 提交**

```bash
git ls-files --error-unmatch .env 2>&1 | head -1   # 应报错（未跟踪）
git add config.py tests/test_config_f6.py
git commit -m "feat(F6): 新增答案定位增强配置开关(F6a上下文增强/F6b多跳分解)"
```

---

## Task 2: 上下文生成器 contextual.py

**Files:**
- Create: `app/ingestion/contextual.py`
- Test: `tests/test_contextual_chunking.py`

**Interfaces:**
- Consumes: `config.get_settings`、`config.get_llm_extra_body`
- Produces:
  - `generate_chunk_context(doc_text: str, chunk_text: str, llm=None, max_chars: int | None = None) -> str`（失败返回 `""`）
  - `build_chunk_contexts(chunks: List[Document], llm=None, max_chars: int | None = None) -> List[str]`（与 chunks 等长）

- [ ] **Step 1: 写失败测试**

```python
# tests/test_contextual_chunking.py
"""F6a 上下文增强分块：上下文生成器（mock LLM，离线）"""
from unittest.mock import MagicMock

from langchain_core.documents import Document

from app.ingestion.contextual import generate_chunk_context, build_chunk_contexts


def _llm_returning(text):
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content=text)
    return llm


def test_generate_context_normal():
    llm = _llm_returning("本文讲范廷颂生平，本段述其受封主教。")
    ctx = generate_chunk_context("文档全文...", "范廷颂于1963年受封主教。", llm=llm)
    assert ctx == "本文讲范廷颂生平，本段述其受封主教。"


def test_generate_context_truncates_to_max_chars():
    llm = _llm_returning("x" * 200)
    ctx = generate_chunk_context("d", "c", llm=llm, max_chars=10)
    assert len(ctx) == 10


def test_generate_context_takes_first_line():
    llm = _llm_returning("第一行\n第二行")
    assert generate_chunk_context("d", "c", llm=llm) == "第一行"


def test_generate_context_llm_exception_degrades_to_empty():
    llm = MagicMock()
    llm.invoke.side_effect = RuntimeError("boom")
    assert generate_chunk_context("d", "c", llm=llm) == ""


def test_build_chunk_contexts_groups_by_doc_and_aligns():
    chunks = [
        Document(page_content="a1", metadata={"doc_id": "A"}),
        Document(page_content="a2", metadata={"doc_id": "A"}),
        Document(page_content="b1", metadata={"doc_id": "B"}),
    ]
    llm = _llm_returning("CTX")
    contexts = build_chunk_contexts(chunks, llm=llm)
    assert contexts == ["CTX", "CTX", "CTX"]
    assert len(contexts) == len(chunks)
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/test_contextual_chunking.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'app.ingestion.contextual'`）

- [ ] **Step 3: 实现 contextual.py**

```python
# app/ingestion/contextual.py
"""上下文增强分块 - 索引时为每个块生成文档级上下文（Anthropic Contextual Retrieval 思路）

动机：脱离文档上下文的"裸块"embedding 无法表达"这是讲 X 的文档里关于 Y 的段落"，
排序精度受限。索引时给每块补一句文档级定位并用"上下文+原文"做 embedding，
可让真正含答案的块排序上升。上下文生成是索引时一次性 LLM，在线检索零增量。
失败一律降级为裸块（空上下文），绝不阻断索引。
"""

import logging
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from config import get_settings, get_llm_extra_body

logger = logging.getLogger(__name__)

CONTEXT_PROMPT = ChatPromptTemplate.from_template(
    """给定一篇文档和其中的一个片段，请用一句话（{max_chars}字以内）说明这个片段在文档中的定位
（文档主题 + 本片段在讲什么），用于增强该片段的检索向量。只输出这一句话，不要解释。

文档（节选）:
{doc_text}

片段:
{chunk_text}

上下文定位："""
)


def _get_llm() -> ChatOpenAI:
    settings = get_settings()
    return ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        temperature=0,
        request_timeout=30,
        max_retries=2,
        extra_body=get_llm_extra_body(),
    )


def generate_chunk_context(
    doc_text: str,
    chunk_text: str,
    llm=None,
    max_chars: Optional[int] = None,
) -> str:
    """为单个块生成文档级上下文；任何失败返回空串（降级为裸块）。"""
    settings = get_settings()
    max_chars = max_chars or settings.contextual_max_chars
    if llm is None:
        llm = _get_llm()
    chain = CONTEXT_PROMPT | llm | StrOutputParser()
    try:
        ctx = chain.invoke({
            "doc_text": (doc_text or "")[:4000],
            "chunk_text": chunk_text or "",
            "max_chars": max_chars,
        })
        ctx = ctx.strip().split("\n")[0].strip()
        return ctx[:max_chars] if ctx else ""
    except Exception as e:
        logger.warning(f"上下文生成失败，降级裸块: {e}")
        return ""


def build_chunk_contexts(
    chunks: List[Document],
    llm=None,
    max_chars: Optional[int] = None,
) -> List[str]:
    """为每个块生成上下文，返回与 chunks 等长、顺序一致的列表。

    同 doc_id 的块共享一份 doc_text（由该文档所有块拼接近似还原）。
    """
    if llm is None:
        llm = _get_llm()

    # 按 doc_id 还原 doc_text
    doc_texts: dict[str, str] = {}
    for ch in chunks:
        doc_id = ch.metadata.get("doc_id", "unknown")
        doc_texts[doc_id] = doc_texts.get(doc_id, "") + "\n" + ch.page_content

    contexts: List[str] = []
    for ch in chunks:
        doc_id = ch.metadata.get("doc_id", "unknown")
        contexts.append(
            generate_chunk_context(doc_texts[doc_id], ch.page_content, llm=llm, max_chars=max_chars)
        )
    return contexts
```

- [ ] **Step 4: 运行确认通过**

Run: `uv run pytest tests/test_contextual_chunking.py -v`
Expected: PASS（5 passed）

- [ ] **Step 5: 提交**

```bash
git ls-files --error-unmatch .env 2>&1 | head -1
git add app/ingestion/contextual.py tests/test_contextual_chunking.py
git commit -m "feat(F6a): 上下文生成器 contextual.py(索引时一次性LLM,失败降级裸块)"
```

---

## Task 3: indexer 上下文增强双 collection + detail_store 切换

**Files:**
- Modify: `app/ingestion/indexer.py:90-113`（新增 `contextual_store` / `detail_store` 属性）、`:115-150`（新增 `index_documents_contextual`）、`:162-164`（`search_chunks` 用 `detail_store`）
- Modify: `app/retrieval/dense.py:43,58`（`chunk_store` → `detail_store`）
- Test: `tests/test_contextual_chunking.py`（追加）

**Interfaces:**
- Consumes: `app.ingestion.contextual.build_chunk_contexts`（Task 2）、`config` 新字段（Task 1）
- Produces:
  - `HierarchicalIndexer.contextual_store -> Chroma`
  - `HierarchicalIndexer.detail_store -> Chroma`（`use_contextual_chunks` 且 contextual 非空时返回 contextual_store，否则 chunk_store）
  - `HierarchicalIndexer.index_documents_contextual(chunks: List[Document], contexts: List[str])`

- [ ] **Step 1: 写失败测试（追加到 test_contextual_chunking.py）**

```python
# 追加到 tests/test_contextual_chunking.py
import tempfile
import uuid

import pytest
from config import get_settings
from app.ingestion.indexer import HierarchicalIndexer


class FakeEmbeddings:
    """确定性假 embedding：向量由文本长度决定（离线、可复现）"""
    def embed_documents(self, texts):
        return [[float(len(t) % 7), float(len(t) % 3), 1.0] for t in texts]
    def embed_query(self, text):
        return [float(len(text) % 7), float(len(text) % 3), 1.0]


@pytest.fixture
def ctx_indexer(monkeypatch):
    settings = get_settings()
    tag = uuid.uuid4().hex[:8]
    monkeypatch.setattr(settings, "chroma_persist_dir", tempfile.mkdtemp())
    monkeypatch.setattr(settings, "chroma_chunk_collection", f"t_chunks_{tag}")
    monkeypatch.setattr(settings, "chroma_contextual_collection", f"t_ctx_{tag}")
    return HierarchicalIndexer(embeddings=FakeEmbeddings(), llm=MagicMock())


def test_index_documents_contextual_stores_original_text(ctx_indexer):
    chunks = [Document(page_content="原文内容", metadata={"doc_id": "A", "chunk_id": "A_0"})]
    ctx_indexer.index_documents_contextual(chunks, ["这是上下文"])
    data = ctx_indexer.contextual_store._collection.get(include=["documents", "metadatas"])
    # 存的是原文（不含上下文前缀），上下文进 metadata
    assert data["documents"][0] == "原文内容"
    assert data["metadatas"][0]["context"] == "这是上下文"


def test_detail_store_falls_back_to_chunk_store_when_contextual_empty(ctx_indexer, monkeypatch):
    monkeypatch.setattr(get_settings(), "use_contextual_chunks", True)
    # contextual 为空 → 回退 chunk_store
    assert ctx_indexer.detail_store is ctx_indexer.chunk_store


def test_detail_store_uses_contextual_when_enabled_and_built(ctx_indexer, monkeypatch):
    monkeypatch.setattr(get_settings(), "use_contextual_chunks", True)
    chunks = [Document(page_content="原文", metadata={"doc_id": "A", "chunk_id": "A_0"})]
    ctx_indexer.index_documents_contextual(chunks, ["ctx"])
    assert ctx_indexer.detail_store is ctx_indexer.contextual_store


def test_detail_store_uses_chunk_store_when_disabled(ctx_indexer, monkeypatch):
    monkeypatch.setattr(get_settings(), "use_contextual_chunks", False)
    chunks = [Document(page_content="原文", metadata={"doc_id": "A", "chunk_id": "A_0"})]
    ctx_indexer.index_documents_contextual(chunks, ["ctx"])
    assert ctx_indexer.detail_store is ctx_indexer.chunk_store
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/test_contextual_chunking.py -v`
Expected: FAIL（`AttributeError: ... no attribute 'contextual_store'`）

- [ ] **Step 3: 实现 indexer 改动**

在 `indexer.py` 的 `__init__`（第 90-91 行）追加：

```python
        self._contextual_store: Optional[Chroma] = None
```

在 `summary_store` 属性（第 113 行）之后插入：

```python
    @property
    def contextual_store(self) -> Chroma:
        """获取/创建上下文增强明细索引（F6a，与 chunk_store 并行的独立 collection）"""
        if self._contextual_store is None:
            self._contextual_store = Chroma(
                collection_name=self.settings.chroma_contextual_collection,
                embedding_function=self.embeddings,
                persist_directory=self.settings.chroma_persist_dir,
            )
        return self._contextual_store

    @property
    def detail_store(self) -> Chroma:
        """明细检索目标集合：开启上下文增强且其集合非空时用 contextual_store，否则 chunk_store。

        关闭或 contextual 未构建时回退 chunk_store，保证行为与现状一致（回归安全）。
        """
        if self.settings.use_contextual_chunks:
            try:
                if self.contextual_store._collection.count() > 0:
                    return self.contextual_store
            except Exception:
                pass
        return self.chunk_store
```

在 `index_documents`（第 150 行）之后插入：

```python
    def index_documents_contextual(self, chunks: List[Document], contexts: List[str]):
        """F6a：用「上下文+原文」做 embedding 写入 contextual_store，但存储原文 page_content。

        contexts 与 chunks 等长。embedding 用 contextualized 文本（提升排序精度），
        Chroma 中 documents 存原文（避免上下文前缀污染生成），上下文进 metadata。
        """
        if not chunks:
            return
        contextualized = [
            (f"{ctx}\n{ch.page_content}" if ctx else ch.page_content)
            for ctx, ch in zip(contexts, chunks)
        ]
        embeddings = self.embeddings.embed_documents(contextualized)
        ids = [ch.metadata.get("chunk_id", f"ctx_{i}") for i, ch in enumerate(chunks)]
        metadatas = []
        for ctx, ch in zip(contexts, chunks):
            meta = dict(ch.metadata)
            meta["context"] = ctx
            metadatas.append(meta)
        self.contextual_store._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=[ch.page_content for ch in chunks],
            metadatas=metadatas,
        )
        logger.info(f"F6a 上下文增强索引写入: {len(chunks)} 个分块")
```

把 `search_chunks`（第 162-164 行）改为：

```python
    def search_chunks(self, query: str, top_k: int = 10) -> List[Document]:
        """L2 明细检索（按 detail_store 开关选择 contextual / 裸块集合）"""
        return self.detail_store.similarity_search(query, k=top_k)
```

把 `dense.py` 第 43 行 `self.indexer.chunk_store.similarity_search_by_vector` 改为
`self.indexer.detail_store.similarity_search_by_vector`；第 58 行
`self.indexer.chunk_store.similarity_search_with_relevance_scores` 改为
`self.indexer.detail_store.similarity_search_with_relevance_scores`。

- [ ] **Step 4: 运行确认通过**

Run: `uv run pytest tests/test_contextual_chunking.py -v`
Expected: PASS（9 passed）

- [ ] **Step 5: 提交**

```bash
git ls-files --error-unmatch .env 2>&1 | head -1
git add app/ingestion/indexer.py app/retrieval/dense.py tests/test_contextual_chunking.py
git commit -m "feat(F6a): indexer 上下文增强双collection + detail_store 按开关切换"
```

---

## Task 4: Parent-Child 接线（上传 + 重建）

**Files:**
- Modify: `app/generation/chain.py`（新增 `rebuild_parent_child_index()` 方法）
- Modify: `app/api/routes.py:252`（上传接线）、新增 `POST /api/documents/reindex` 端点
- Test: `tests/test_parent_child_wiring.py`

**Interfaces:**
- Consumes: `ParentChildRetriever.index_documents(documents)`（接收**原始未分块**文档）、`HierarchicalIndexer.get_all_chunks()`
- Produces: `RAGChain.rebuild_parent_child_index() -> int`（返回重建的文档数；无 parent_child_retriever 返回 0）

- [ ] **Step 1: 写失败测试**

```python
# tests/test_parent_child_wiring.py
"""F6a Parent-Child 接线：上传索引 + 从已有分块重建（mock，离线）"""
from unittest.mock import MagicMock

from langchain_core.documents import Document

from app.generation.chain import RAGChain


def _chain_with_mocks():
    chain = RAGChain.__new__(RAGChain)  # 跳过 __init__，手动装配 mock
    chain.indexer = MagicMock()
    chain.parent_child_retriever = MagicMock()
    return chain


def test_rebuild_parent_child_groups_chunks_by_doc():
    chain = _chain_with_mocks()
    chain.indexer.get_all_chunks.return_value = [
        Document(page_content="a1", metadata={"doc_id": "A", "source": "a.txt"}),
        Document(page_content="a2", metadata={"doc_id": "A", "source": "a.txt"}),
        Document(page_content="b1", metadata={"doc_id": "B", "source": "b.txt"}),
    ]
    n = chain.rebuild_parent_child_index()
    assert n == 2  # 两个文档
    args = chain.parent_child_retriever.index_documents.call_args[0][0]
    doc_ids = sorted(d.metadata["doc_id"] for d in args)
    assert doc_ids == ["A", "B"]
    # A 的内容被拼接
    a_doc = next(d for d in args if d.metadata["doc_id"] == "A")
    assert "a1" in a_doc.page_content and "a2" in a_doc.page_content


def test_rebuild_parent_child_no_retriever_returns_zero():
    chain = _chain_with_mocks()
    chain.parent_child_retriever = None
    assert chain.rebuild_parent_child_index() == 0


def test_rebuild_parent_child_exception_returns_zero():
    chain = _chain_with_mocks()
    chain.indexer.get_all_chunks.side_effect = RuntimeError("boom")
    assert chain.rebuild_parent_child_index() == 0
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/test_parent_child_wiring.py -v`
Expected: FAIL（`AttributeError: 'RAGChain' object has no attribute 'rebuild_parent_child_index'`）

- [ ] **Step 3: 实现 chain 辅助方法**

在 `chain.py` 的 `retrieve` 方法（第 140 行）之前插入：

```python
    def rebuild_parent_child_index(self) -> int:
        """F6a：从已索引分块重建 Parent-Child 索引（一次性，用于补齐历史文档）。

        按 doc_id 把分块拼回文档级文本，再交给 ParentChildRetriever 重新切分 parent/child。
        无 parent_child_retriever 或任何异常都返回 0（优雅降级）。
        """
        if not self.parent_child_retriever:
            return 0
        try:
            all_chunks = self.indexer.get_all_chunks()
            grouped: dict[str, Document] = {}
            for ch in all_chunks:
                doc_id = ch.metadata.get("doc_id", "unknown")
                if doc_id not in grouped:
                    grouped[doc_id] = Document(
                        page_content="",
                        metadata={"doc_id": doc_id, "source": ch.metadata.get("source", "")},
                    )
                grouped[doc_id].page_content += "\n" + ch.page_content
            docs = list(grouped.values())
            if docs:
                self.parent_child_retriever.index_documents(docs)
            logger.info(f"F6a Parent-Child 重建: {len(docs)} 个文档")
            return len(docs)
        except Exception as e:
            logger.warning(f"F6a Parent-Child 重建失败: {e}")
            return 0
```

- [ ] **Step 4: 运行确认通过**

Run: `uv run pytest tests/test_parent_child_wiring.py -v`
Expected: PASS（3 passed）

- [ ] **Step 5: 接线上传端点 + 重建端点**

在 `routes.py` 上传端点的 `chain.indexer.index_documents(chunks)`（第 252 行）之后插入：

```python
        # F6a: 接线 Parent-Child（小块检索大块返回，修复"答案不在 top 块"）
        # 注意：parent_child.index_documents 接收原始文档 docs（其内部自行切分 parent/child）
        if chain.parent_child_retriever:
            try:
                chain.parent_child_retriever.index_documents(docs)
            except Exception as e:
                logger.warning(f"F6a Parent-Child 索引构建失败: {e}")

        # F6a: 上下文增强索引（索引时一次性 LLM，零在线增量）
        if get_settings().use_contextual_chunks:
            try:
                from app.ingestion.contextual import build_chunk_contexts
                contexts = build_chunk_contexts(chunks)
                chain.indexer.index_documents_contextual(chunks, contexts)
            except Exception as e:
                logger.warning(f"F6a 上下文增强索引失败: {e}")
```

> 注：`get_settings` 已在 routes.py 顶部导入则直接用；否则在文件顶部 `from config import get_settings`。

在 `list_documents` 端点（第 269 行）之前插入重建端点（同时重建 Parent-Child 与上下文增强，供历史文档做 F6a A/B）：

```python
@router.post("/api/documents/reindex")
async def reindex_f6a():
    """F6a：从已索引分块重建 Parent-Child + 上下文增强索引（补齐历史文档，一次性）。

    上下文增强需对每块调用一次 LLM，文档多时较慢，属一次性管理操作。
    """
    chain = get_rag_chain()
    num_docs = chain.rebuild_parent_child_index()
    num_ctx = 0
    if get_settings().use_contextual_chunks:
        try:
            from app.ingestion.contextual import build_chunk_contexts
            chunks = chain.indexer.get_all_chunks()
            contexts = build_chunk_contexts(chunks)
            chain.indexer.index_documents_contextual(chunks, contexts)
            num_ctx = len(chunks)
        except Exception as e:
            logger.warning(f"F6a 上下文增强重建失败: {e}")
    return {
        "message": "F6a 索引重建完成",
        "num_documents": num_docs,
        "num_contextual_chunks": num_ctx,
    }
```

- [ ] **Step 6: 冒烟验证导入无误**

Run: `uv run python -c "import app.api.routes; import app.generation.chain; print('ok')"`
Expected: `ok`

- [ ] **Step 7: 提交**

```bash
git ls-files --error-unmatch .env 2>&1 | head -1
git add app/generation/chain.py app/api/routes.py tests/test_parent_child_wiring.py
git commit -m "feat(F6a): 接线 Parent-Child(上传+重建端点) + 上传时上下文增强索引"
```

---

# Phase B — F6b 多跳查询分解

## Task 5: 分解器 decompose

**Files:**
- Modify: `app/retrieval/query_transform.py`（新增 `Decomposition` dataclass、`DECOMPOSE_PROMPT`、`decompose()`）
- Test: `tests/test_decomposition.py`

**Interfaces:**
- Consumes: `CRAGEvaluator._extract_json`（`app/retrieval/crag.py`，静态方法，返回 dict）、`config` F6b 字段
- Produces:
  - `@dataclass Decomposition: sub_questions: List[str]; chain: bool`
  - `QueryTransformer.decompose(question: str) -> Decomposition`（失败/单子问题回退 `Decomposition([question], False)`）

- [ ] **Step 1: 写失败测试**

```python
# tests/test_decomposition.py
"""F6b 多跳查询分解（mock LLM，离线）"""
import json
from unittest.mock import MagicMock

from app.retrieval.query_transform import QueryTransformer, Decomposition


def _transformer_returning(content):
    t = QueryTransformer.__new__(QueryTransformer)  # 跳过 __init__（不连真实 LLM）
    t.llm = MagicMock()
    t.llm.invoke.return_value = MagicMock(content=content)
    t._transform_cache = {}
    t._cache_ttl = 3600
    return t


def test_decompose_parallel():
    payload = json.dumps({"sub_questions": ["范廷颂担任总主教的教区是哪个？", "该教区在哪里？"], "chain": False})
    t = _transformer_returning(payload)
    d = t.decompose("范廷颂担任总主教的那个教区在哪里？")
    assert d.sub_questions == ["范廷颂担任总主教的教区是哪个？", "该教区在哪里？"]
    assert d.chain is False


def test_decompose_chain_flag():
    payload = json.dumps({"sub_questions": ["q1", "q2"], "chain": True})
    t = _transformer_returning(payload)
    assert t.decompose("Q").chain is True


def test_decompose_truncates_to_max_subquestions(monkeypatch):
    from config import get_settings
    monkeypatch.setattr(get_settings(), "decomposition_max_subquestions", 2)
    payload = json.dumps({"sub_questions": ["q1", "q2", "q3", "q4"], "chain": False})
    t = _transformer_returning(payload)
    assert t.decompose("Q").sub_questions == ["q1", "q2"]


def test_decompose_single_subquestion_falls_back_to_original():
    payload = json.dumps({"sub_questions": ["只有一个"], "chain": False})
    t = _transformer_returning(payload)
    d = t.decompose("原问题")
    assert d.sub_questions == ["原问题"]
    assert d.chain is False


def test_decompose_invalid_json_falls_back():
    t = _transformer_returning("这不是 JSON")
    d = t.decompose("原问题")
    assert d.sub_questions == ["原问题"]
    assert d.chain is False


def test_decompose_llm_exception_falls_back():
    t = QueryTransformer.__new__(QueryTransformer)
    t.llm = MagicMock()
    t.llm.invoke.side_effect = RuntimeError("boom")
    t._transform_cache = {}
    t._cache_ttl = 3600
    d = t.decompose("原问题")
    assert d.sub_questions == ["原问题"]
    assert d.chain is False
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/test_decomposition.py -v`
Expected: FAIL（`ImportError: cannot import name 'Decomposition'`）

- [ ] **Step 3: 实现 decompose**

在 `query_transform.py` 顶部 import 区追加：

```python
from dataclasses import dataclass, field
from app.retrieval.crag import CRAGEvaluator
```

在 `REFINE_PROMPT`（第 55 行）之后插入：

```python
# F6b 多跳查询分解 Prompt
DECOMPOSE_PROMPT = ChatPromptTemplate.from_template(
    """你是一个多跳问题分解器。把一个需要多步检索才能回答的复杂问题，拆成若干更简单的子问题。

要求：
1. 每个子问题应可独立检索，或构成清晰的依赖链。
2. 若后一个子问题必须依赖前一个子问题的答案（如"X 的 Y 的 Z"），把 chain 设为 true。
3. 子问题数量控制在 {max_sub} 个以内。
4. 只输出 JSON：{{"sub_questions": ["子问题1", "子问题2"], "chain": false}}

问题: {question}

JSON："""
)


@dataclass
class Decomposition:
    """多跳分解结果：sub_questions 为子问题列表；chain 表示是否存在依赖链。"""
    sub_questions: List[str] = field(default_factory=list)
    chain: bool = False
```

在 `refine` 方法（第 172 行）之后插入：

```python
    def decompose(self, question: str) -> "Decomposition":
        """F6b 多跳查询分解：把多跳问题拆成子问题。

        失败 / 解析不出 / 仅 1 个子问题时回退 Decomposition([question], False)（退化为单跳，绝不阻断）。
        """
        settings = get_settings()
        fallback = Decomposition(sub_questions=[question], chain=False)
        chain = DECOMPOSE_PROMPT | self.llm | StrOutputParser()
        try:
            raw = chain.invoke({
                "question": question,
                "max_sub": settings.decomposition_max_subquestions,
            })
            data = CRAGEvaluator._extract_json(raw)
            if not isinstance(data, dict):
                return fallback
            subs = [str(s).strip() for s in data.get("sub_questions", []) if str(s).strip()]
            subs = subs[: settings.decomposition_max_subquestions]
            if len(subs) <= 1:
                return fallback
            return Decomposition(sub_questions=subs, chain=bool(data.get("chain", False)))
        except Exception as e:
            logger.warning(f"多跳分解失败，退化为单跳: {e}")
            return fallback
```

- [ ] **Step 4: 运行确认通过**

Run: `uv run pytest tests/test_decomposition.py -v`
Expected: PASS（6 passed）

- [ ] **Step 5: 提交**

```bash
git ls-files --error-unmatch .env 2>&1 | head -1
git add app/retrieval/query_transform.py tests/test_decomposition.py
git commit -m "feat(F6b): 多跳查询分解器 decompose(JSON输出,失败退化单跳)"
```

---

## Task 6: pipeline 多跳分支（并行优先 / 依赖时链式）

**Files:**
- Modify: `app/retrieval/pipeline.py:22-41`（`RetrievalResult` 新字段）、新增 `_retrieve_subquery` / `_decompose_retrieve`、`run()` 多跳分支（第 403-425 行区域）
- Test: `tests/test_pipeline_multihop.py`

**Interfaces:**
- Consumes: `QueryTransformer.decompose(question) -> Decomposition`（Task 5）、`recall` / `fuse` / `rerank` / `evaluate`
- Produces:
  - `RetrievalResult.decomposed_subqueries: List[str]`、`decomposition_chain: bool`、`answer_localization_method: str`
  - `RetrievalPipeline._retrieve_subquery(question, subq, top_n) -> List[Document]`
  - `RetrievalPipeline._decompose_retrieve(question, decomposition, top_k, trace_id) -> List[Document]`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_pipeline_multihop.py
"""F6b pipeline 多跳分支（mock，离线；新开关默认关以保护既有测试）"""
from unittest.mock import MagicMock

from langchain_core.documents import Document

from app.retrieval.pipeline import RetrievalPipeline
from app.retrieval.query_transform import Decomposition


def _doc(cid):
    return Document(page_content=f"内容{cid}", metadata={"chunk_id": cid})


def _make_pipeline(settings_overrides=None):
    settings = MagicMock()
    settings.retrieval_top_k = 5
    settings.rerank_top_n = 20
    settings.recall_max_workers = 4
    settings.use_summary_recall = False
    settings.use_crag_gate = False          # 关门控，简化
    settings.use_query_router = True
    settings.use_autocut = False
    settings.autocut_min_docs = 2
    settings.use_iterative_retrieval = False  # 关 F2，隔离 F6b
    settings.use_decomposition = True
    settings.decomposition_max_subquestions = 4
    settings.decomposition_max_hops = 3
    for k, v in (settings_overrides or {}).items():
        setattr(settings, k, v)

    p = RetrievalPipeline.__new__(RetrievalPipeline)
    p.indexer = MagicMock()
    p.dense_retriever = MagicMock()
    p.sparse_retriever = MagicMock()
    p.reranker = None
    p.query_transformer = MagicMock()
    p.graph_retriever = None
    p.parent_child_retriever = None
    p.crag_evaluator = None               # 关 CRAG，隔离
    p.query_router = MagicMock()
    p.query_router.route.return_value = MagicMock(
        query_type="multi_hop", top_k=None, autocut_min_docs=None, reason="多跳"
    )
    p._settings = settings
    return p


def test_multihop_parallel_decomposition_merges_subqueries():
    p = _make_pipeline()
    p.query_transformer.decompose.return_value = Decomposition(
        sub_questions=["子问题1", "子问题2"], chain=False
    )
    # recall 按子问题返回不同文档
    p.recall = MagicMock(side_effect=[
        {"dense": [_doc("a")], "sparse": []},
        {"dense": [_doc("b")], "sparse": []},
    ])
    result = p.run("范廷颂担任总主教的那个教区在哪里？")
    assert result.decomposed_subqueries == ["子问题1", "子问题2"]
    assert result.decomposition_chain is False
    fused_ids = {d.metadata["chunk_id"] for d in result.fused_results}
    assert {"a", "b"} <= fused_ids


def test_multihop_single_subquery_falls_back_to_normal_recall():
    p = _make_pipeline()
    p.query_transformer.decompose.return_value = Decomposition(
        sub_questions=["原问题"], chain=False
    )
    p.recall = MagicMock(return_value={"dense": [_doc("x")], "sparse": []})
    result = p.run("简单问题")
    assert result.decomposed_subqueries == []  # 未触发分解合并
    assert p.recall.call_count == 1


def test_multihop_disabled_goes_normal():
    p = _make_pipeline({"use_decomposition": False})
    p.recall = MagicMock(return_value={"dense": [_doc("x")], "sparse": []})
    result = p.run("多跳问题")
    assert result.decomposed_subqueries == []
    p.query_transformer.decompose.assert_not_called()


def test_non_multihop_does_not_decompose():
    p = _make_pipeline()
    p.query_router.route.return_value = MagicMock(
        query_type="factual", top_k=None, autocut_min_docs=None, reason="事实"
    )
    p.recall = MagicMock(return_value={"dense": [_doc("x")], "sparse": []})
    result = p.run("事实问题")
    p.query_transformer.decompose.assert_not_called()
    assert result.decomposed_subqueries == []
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/test_pipeline_multihop.py -v`
Expected: FAIL（`AttributeError: ... 'decomposed_subqueries'` 或分解分支未实现）

- [ ] **Step 3: 实现 RetrievalResult 新字段**

在 `pipeline.py` 的 `RetrievalResult`（第 41 行 `iterative_stop_reason` 之后）追加：

```python
    # F6 答案定位增强观测字段
    decomposed_subqueries: List[str] = field(default_factory=list)  # F6b: 分解出的子问题
    decomposition_chain: bool = False                              # F6b: 是否依赖链
    answer_localization_method: str = ""                           # F6a: parent_child / contextual / ""
```

- [ ] **Step 4: 实现 _retrieve_subquery 与 _decompose_retrieve**

在 `_evidence_summary`（第 326 行）之后、`# ---- 主编排 ----` 之前插入：

```python
    # ---- 阶段 3b: F6b 多跳查询分解（并行优先，依赖时链式） ----

    def _retrieve_subquery(
        self, question: str, subq: str, top_n: int
    ) -> List[Document]:
        """子问题轻量检索：跳过门控/改写，直接 dense+sparse 召回 + fuse。"""
        recall_results = self.recall(
            question, [subq], top_n=top_n, channels=("dense", "sparse")
        )
        return self.fuse(recall_results)

    def _decompose_retrieve(
        self, question: str, decomposition, top_k: int, trace_id: str | None = None
    ) -> List[Document]:
        """按分解结果检索并合并：无依赖→并行各子问题；有依赖→链式（hop 答案构造下一跳）。"""
        settings = self._settings
        subs = decomposition.sub_questions

        if not decomposition.chain:
            # 并行：各子问题独立轻量检索，统一 RRF 合并
            per_sub: List[List[Document]] = []
            with ThreadPoolExecutor(max_workers=settings.recall_max_workers) as executor:
                futures = [
                    executor.submit(self._retrieve_subquery, question, sq, top_k)
                    for sq in subs
                ]
                for fut in as_completed(futures):
                    try:
                        per_sub.append(fut.result())
                    except Exception as e:
                        logger.warning(f"子问题检索失败: {e}")
            return reciprocal_rank_fusion(per_sub) if per_sub else []

        # 链式：逐跳检索，用上一跳的压缩证据辅助构造下一跳查询
        accumulated: List[Document] = []
        current_subs = list(subs)[: settings.decomposition_max_hops]
        for i, sq in enumerate(current_subs):
            docs = self._retrieve_subquery(question, sq, top_k)
            accumulated = self._deduplicate(accumulated + docs)
            # 若不是最后一跳，用已检索证据精化下一跳（复用 F2 精化，异常回退原子问题）
            if i < len(current_subs) - 1 and self.query_transformer:
                try:
                    current_subs[i + 1] = self.query_transformer.refine(
                        current_subs[i + 1], self._evidence_summary(accumulated), "需结合上一跳结果"
                    )
                except Exception as e:
                    logger.warning(f"链式精化失败，沿用原子问题: {e}")
        return accumulated
```

- [ ] **Step 5: 在 run() 接入多跳分支**

在 `run()` 的「③ 多路召回」之前（第 403 行 `if trace_id: tracer.start_span(trace_id, "multi_recall")` 之前）插入：

```python
        # F6b 多跳查询分解：仅 multi_hop 且开启分解时触发；分解出 >1 子问题才走分解合并
        use_decomp = (
            result.query_type == "multi_hop"
            and settings.use_decomposition
            and self.query_transformer is not None
        )
        decomposed = False
        if use_decomp:
            decomposition = self.query_transformer.decompose(question)
            if len(decomposition.sub_questions) > 1:
                if trace_id:
                    tracer.start_span(trace_id, "decomposition")
                result.decomposed_subqueries = decomposition.sub_questions
                result.decomposition_chain = decomposition.chain
                result.fused_results = self._decompose_retrieve(
                    question, decomposition, effective_top_k, trace_id=trace_id
                )
                decomposed = True
                if trace_id:
                    tracer.end_span(trace_id, "decomposition", {
                        "subquestions": decomposition.sub_questions,
                        "chain": decomposition.chain,
                        "merged": len(result.fused_results),
                    })
```

把原「③ 多路召回 + ④ RRF 融合」段（第 403-425 行）整体包进 `if not decomposed:`，即：

```python
        if not decomposed:
            # ③ 多路召回
            if trace_id:
                tracer.start_span(trace_id, "multi_recall")
            recall_results = self.recall(question, queries, trace_id=trace_id)
            result.dense_results = recall_results["dense"]
            result.sparse_results = recall_results["sparse"]
            result.graph_results = recall_results.get("graph", [])
            result.summary_results = recall_results.get("summary", [])
            if trace_id:
                tracer.end_span(trace_id, "multi_recall", {
                    "dense_hits": len(result.dense_results),
                    "sparse_hits": len(result.sparse_results),
                    "graph_hits": len(result.graph_results),
                    "pc_hits": len(recall_results.get("parent_child", [])),
                    "summary_hits": len(result.summary_results),
                })

            # ④ RRF 融合
            if trace_id:
                tracer.start_span(trace_id, "rrf_fusion")
            result.fused_results = self.fuse(recall_results)
            if trace_id:
                tracer.end_span(trace_id, "rrf_fusion", {"fused": len(result.fused_results)})
```

> ⑤ 重排及之后（第 427 行起）保持不变——分解合并出的 `fused_results` 与正常融合走同一条 rerank/autocut/CRAG 路径。

另外，在 `run()` 末尾 `result.retrieval_time_ms = (time.time() - start_time) * 1000`（第 498 行）之前插入 F6a 观测字段赋值（CRAG 可能替换 documents，故放在最末，metadata 在过滤/重排中保留）：

```python
        # F6a 观测：标注答案定位主要来源（parent_child 优先，其次 contextual）
        if any(d.metadata.get("retrieval_method") == "parent_child" for d in result.documents):
            result.answer_localization_method = "parent_child"
        elif settings.use_contextual_chunks:
            result.answer_localization_method = "contextual"
```

- [ ] **Step 6: 运行确认通过**

Run: `uv run pytest tests/test_pipeline_multihop.py -v`
Expected: PASS（4 passed）

- [ ] **Step 7: 跑全量回归确认无破坏**

Run: `uv run pytest -q`
Expected: 既有 126 + 新增测试全绿（`test_pipeline.py` 的 mock 未开 F6 开关，行为不变）

- [ ] **Step 8: 提交**

```bash
git ls-files --error-unmatch .env 2>&1 | head -1
git add app/retrieval/pipeline.py tests/test_pipeline_multihop.py
git commit -m "feat(F6b): pipeline 多跳分解分支(并行优先/依赖链式,合并后走原管道)"
```

---

# Phase C — 评估与文档

## Task 7: 评估指标 answer_in_top_context

**Files:**
- Modify: `app/evaluation/metrics.py`（在 `answer_hit` 之后追加 `answer_in_top_context`）
- Test: `tests/test_e2e_metrics.py`（追加）

**Interfaces:**
- Consumes: `normalize_answer`（`metrics.py:475`）
- Produces: `answer_in_top_context(gold: str, documents) -> bool`

- [ ] **Step 1: 写失败测试（追加到 test_e2e_metrics.py）**

```python
# 追加到 tests/test_e2e_metrics.py
from langchain_core.documents import Document
from app.evaluation.metrics import answer_in_top_context


def test_answer_in_top_context_hit():
    docs = [Document(page_content="范廷颂于1963年受封主教。")]
    assert answer_in_top_context("1963年", docs) is True


def test_answer_in_top_context_miss():
    docs = [Document(page_content="无关内容。")]
    assert answer_in_top_context("1963年", docs) is False


def test_answer_in_top_context_empty_gold():
    docs = [Document(page_content="任意内容")]
    assert answer_in_top_context("", docs) is False


def test_answer_in_top_context_empty_docs():
    assert answer_in_top_context("1963年", []) is False


def test_answer_in_top_context_normalizes_punctuation():
    docs = [Document(page_content="答案是 1963 年。")]
    assert answer_in_top_context("1963年", docs) is True  # 归一化后空格/标点被忽略
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/test_e2e_metrics.py -v -k answer_in_top_context`
Expected: FAIL（`ImportError: cannot import name 'answer_in_top_context'`）

- [ ] **Step 3: 实现指标**

在 `metrics.py` 的 `answer_hit`（第 517 行）之后追加：

```python
def answer_in_top_context(gold: str, documents) -> bool:
    """F6 细粒度指标：gold 答案（归一化后）是否出现在送入 LLM 的 top 上下文里。

    用于度量"源命中但答案段不在 top 块"是否被修复——直接看答案材料有没有进上下文。
    gold 为空或无文档返回 False。
    """
    norm_gold = normalize_answer(gold)
    if not norm_gold or not documents:
        return False
    context = normalize_answer(" ".join(getattr(d, "page_content", "") for d in documents))
    return norm_gold in context
```

- [ ] **Step 4: 运行确认通过**

Run: `uv run pytest tests/test_e2e_metrics.py -v -k answer_in_top_context`
Expected: PASS（5 passed）

- [ ] **Step 5: 提交**

```bash
git ls-files --error-unmatch .env 2>&1 | head -1
git add app/evaluation/metrics.py tests/test_e2e_metrics.py
git commit -m "feat(F6): 评估指标 answer_in_top_context(答案段是否进top上下文)"
```

---

## Task 8: 评估 harness 接入 F6 + 切片

**Files:**
- Modify: `run_e2e_eval.py:39-44`（FEATURE_FLAGS 加 F6）、`:33`（import answer_in_top_context）、`eval_sample`（新字段）、`aggregate`（新汇总）、`main`（`--slice`）

**Interfaces:**
- Consumes: `answer_in_top_context`（Task 7）、`RetrievalResult.decomposed_subqueries` / `decomposition_chain`（Task 6）
- Produces: `--slice multihop|finegrained` 过滤、汇总新增 `answer_in_top_context_rate` 与分解统计

- [ ] **Step 1: 改 FEATURE_FLAGS 与 import**

把 `run_e2e_eval.py:33` 改为：

```python
from app.evaluation.metrics import answer_f1, normalized_exact_match, answer_hit, answer_in_top_context
```

把 `FEATURE_FLAGS`（第 39-44 行）改为：

```python
FEATURE_FLAGS = {
    "F1": "use_autocut",
    "F2": "use_iterative_retrieval",
    "F3": "use_faithfulness_check",
    "F4": "use_query_router",
    "F6": "use_decomposition",
}
```

- [ ] **Step 1b: apply_feature_mode 同步 F6a 开关（使 baseline/full 干净 A/B）**

`use_contextual_chunks` / `use_parent_child` 是 F6a 的运行时开关，但不在 `FEATURE_FLAGS` 里。若 baseline/full 不切它们，则 F6a 无法做干净 A/B（baseline 仍用上下文块）。在 `FEATURE_FLAGS` 定义之后（第 44 行后）追加：

```python
# F6a 运行时开关：随 baseline/full 一并切换；--only 单特性归因时关闭以隔离
F6A_FLAGS = ["use_contextual_chunks", "use_parent_child"]
```

把 `apply_feature_mode`（第 61-81 行）的 baseline/full/only 三个分支各补一行 F6a 处理：

```python
    if mode == "baseline":
        for flag in FEATURE_FLAGS.values():
            setattr(settings, flag, False)
        for flag in F6A_FLAGS:
            setattr(settings, flag, False)
    elif mode == "full":
        for flag in FEATURE_FLAGS.values():
            setattr(settings, flag, True)
        for flag in F6A_FLAGS:
            setattr(settings, flag, True)

    if only:
        only = only.upper()
        if only not in FEATURE_FLAGS:
            raise ValueError(f"--only 须为 {list(FEATURE_FLAGS)} 之一")
        for key, flag in FEATURE_FLAGS.items():
            setattr(settings, flag, key == only)
        for flag in F6A_FLAGS:
            setattr(settings, flag, False)  # 单特性归因时关闭 F6a，避免污染
```

> 注意：`use_parent_child` 在 `RAGChain.__init__` 读取（决定是否构造 parent_child_retriever），而 `apply_feature_mode` 在 `main()` 中先于 `RAGChain()` 调用，故切换生效。`use_contextual_chunks` 在查询期由 `detail_store` 读取，切换亦生效。
> 前提：F6a 要有效果，`chunks_contextual` collection 必须已建（见 Task 4 的 `/api/documents/reindex`）。未建时 `detail_store` 自动回退 `chunk_store`，baseline/full 行为一致（安全降级）。

- [ ] **Step 2: eval_sample 增加 F6 字段**

在 `eval_sample` 的 return dict（第 122-142 行）的「观测」段追加：

```python
        # F6 答案定位
        "answer_in_top_context": answer_in_top_context(gold, docs),
        "decomposed": rr.decomposed_subqueries,
        "decomposition_chain": rr.decomposition_chain,
```

- [ ] **Step 3: aggregate 增加汇总**

在 `aggregate` 的 return dict（第 183-208 行）的 `end_to_end` 段追加：

```python
            "answer_in_top_context_rate": _mean([int(r["answer_in_top_context"]) for r in ok]),
```

并在 return dict 末尾（`iterative` 之后）追加：

```python
        "decomposition": {
            "decomposed_rate": _mean([int(bool(r["decomposed"])) for r in ok]),
            "chain_rate": _mean([int(r["decomposition_chain"]) for r in ok]),
        },
```

- [ ] **Step 4: main 增加 --slice 过滤**

在 `main()` 的 argparse（第 217 行 `--limit` 之后）追加：

```python
    parser.add_argument("--slice", default="", choices=["", "multihop", "finegrained"],
                        help="按样本 slice 字段过滤（多跳/细粒度子集）")
```

在 `samples = dataset["samples"]`（第 236 行）之后插入：

```python
    if args.slice:
        samples = [s for s in samples if s.get("slice") == args.slice]
```

- [ ] **Step 5: 冒烟验证脚本可解析**

Run: `uv run python run_e2e_eval.py --help`
Expected: 输出含 `--slice {,multihop,finegrained}` 与 `--only ... F6`

- [ ] **Step 6: 提交**

```bash
git ls-files --error-unmatch .env 2>&1 | head -1
git add run_e2e_eval.py
git commit -m "feat(F6): 评估harness接入F6开关+answer_in_top_context+多跳/细粒度切片"
```

---

## Task 9: 多跳测试集 + 细粒度子集

**Files:**
- Create: `scripts/build_multihop_eval.py`
- Create: `data/eval_multihop.json`（脚本产物，需人工核对 gold）

**Interfaces:**
- Consumes: 已索引知识库（`chain.indexer.get_all_chunks()`）
- Produces: `data/eval_multihop.json`，schema：`{"samples": [{"id","question","ground_truth","slice":"multihop","chain":bool,"metadata":{"source"}}]}`

> ⚠️ **gold 答案必须人工对照知识库核对**——脚本只辅助生成候选问题骨架，不能凭空编造答案。

- [ ] **Step 1: 写构造脚本**

```python
# scripts/build_multihop_eval.py
"""半自动构造多跳评估集：列出知识库实体候选，输出待人工填写 gold 的骨架。

用法: uv run python scripts/build_multihop_eval.py --out data/eval_multihop.json
产出后必须人工核对每条 ground_truth 与 source，再提交。
"""
import argparse
import json
from pathlib import Path


# 基于知识库真实实体（范廷颂/天主教/总主教/教区）的多跳问题模板。
# ground_truth 为占位，需人工对照 data/sample_docs 核对后填写真实答案。
SEED = [
    {"id": "mh1", "question": "范廷颂担任总主教的那个教区在哪里？",
     "ground_truth": "TODO_核对知识库", "chain": True, "slice": "multihop"},
    {"id": "mh2", "question": "范廷颂受封主教那一年的教宗是谁？",
     "ground_truth": "TODO_核对知识库", "chain": True, "slice": "multihop"},
    {"id": "mh3", "question": "总主教和主教在天主教圣统制中的区别是什么？",
     "ground_truth": "TODO_核对知识库", "chain": False, "slice": "multihop"},
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/eval_multihop.json")
    args = parser.parse_args()

    payload = {"samples": [
        {**s, "metadata": {"source": ""}} for s in SEED
    ]}
    Path(args.out).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"已写出 {args.out}（{len(SEED)} 条骨架）。")
    print("⚠️ 请人工核对每条 ground_truth 与 metadata.source 后再用于评估。")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 生成骨架并人工核对 gold**

Run: `uv run python scripts/build_multihop_eval.py`
然后打开 `data/eval_multihop.json`，对照 `data/sample_docs/` 把每条 `TODO_核对知识库` 替换为真实答案、补 `metadata.source`；按需扩充到 10-15 条（覆盖 chain=true/false 两类）。

- [ ] **Step 3: 标注 CMRC 细粒度子集**

打开 `data/eval_dataset_cmrc.json`，给 id=2、id=4 等「源命中但答案不在 top 块」的样本加字段 `"slice": "finegrained"`（其余样本不加）。

- [ ] **Step 4: 验证切片可被 harness 读取**

Run: `uv run python -c "import json; d=json.load(open('data/eval_multihop.json')); print(len(d['samples']), d['samples'][0]['slice'])"`
Expected: `3 multihop`（或扩充后的数量）

- [ ] **Step 5: 提交**

```bash
git ls-files --error-unmatch .env 2>&1 | head -1
git add scripts/build_multihop_eval.py data/eval_multihop.json data/eval_dataset_cmrc.json
git commit -m "test(F6): 多跳测试集骨架+构造脚本 + CMRC细粒度子集标注(gold需人工核对)"
```

---

## Task 10: 文档同步

**Files:**
- Modify: `README.md`（RAG 2.0 表格加 F6 + 配置开关）
- Modify: `docs/architecture.md`（第 7 章加 F6 小节、配置速查、文件清单）
- Modify: `docs/interview_guide.md`（加 F6 设计决策与面试话术）

- [ ] **Step 1: README 增加 F6**

在 README 的「RAG 2.0 深度增强」表格末尾追加一行：

```markdown
| **F6** | 答案定位增强：Parent-Child 接线 + 上下文增强分块（F6a）/ 多跳查询分解（F6b） | `use_contextual_chunks` / `use_decomposition` |
```

并在配置开关说明处补充 F6 六个开关。

- [ ] **Step 2: architecture.md 增加 F6 小节**

在 `docs/architecture.md` 第 7 章「RAG 2.0 五大特性实现」之后新增「## F6 · 答案定位增强」小节，
说明 F6a（Parent-Child 接线 + contextual chunking 双 collection + `detail_store` 切换）与
F6b（`decompose` + pipeline 多跳分支，并行优先/依赖链式），并在配置速查表与核心文件清单补 F6 条目。

- [ ] **Step 3: interview_guide.md 增加 F6 叙事**

在 `docs/interview_guide.md` 模块深潜区新增 F6 小节：问题（命中率饱和但 F1 不动 / 源命中但答案不在 top 块）、
做法（F6a/F6b）、面试话术、与「embedding 高相似但答案不在召回」深度问题的对应。

- [ ] **Step 4: 提交**

```bash
git ls-files --error-unmatch .env 2>&1 | head -1
git add README.md docs/architecture.md docs/interview_guide.md
git commit -m "docs(F6): 同步 README/架构/面试指南 答案定位增强说明"
```

---

# 验收（Phase 完成后）

- [ ] **全量测试绿**：`uv run pytest -q` → 既有 126 + 新增全部通过。
- [ ] **回归门禁**：`uv run python run_retrieval_eval.py` → CMRC 命中率保持 100%。
- [ ] **F6a A/B**：先 `curl -X POST localhost:8000/api/documents/reindex` 为历史文档建上下文索引，
      再 `uv run python run_e2e_eval.py --mode full --slice finegrained` 对比 `--mode baseline`
      （apply_feature_mode 已在 baseline 关 `use_contextual_chunks`/`use_parent_child`、full 开），
      `answer_in_top_context_rate` 提升、命中率不掉。
- [ ] **F6b A/B**：`uv run python run_e2e_eval.py --mode full --only F6 --dataset data/eval_multihop.json`
      对比 baseline，多跳子集 `avg_f1` / `hit_rate` 提升。
- [ ] **延迟核对**：多跳查询额外延迟在预期内（并行 ≈ +2 次 LLM；链式按跳数）。
- [ ] **`.env` 未跟踪**：`git ls-files --error-unmatch .env` 始终报错。
