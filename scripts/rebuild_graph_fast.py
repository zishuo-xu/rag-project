"""零 LLM 快速重建知识图谱：jieba TF-IDF 关键词 + 共现关系，秒级完成。

回退自 091b64a 的 build_fast：LLM 类型化三元组重建成本过高
（5564 块 ≈ 2782 次 LLM × 23s ≈ 3.6h）且别人复现要付同样成本，与 L3「可复现」
冲突。改零 LLM 共现图，1 秒重建、clone 即可复现。质量降（关系仅共现，无类型化），
但 Graph 检索（实体多跳）仍可用，权衡如实记录。

用法: uv run python scripts/rebuild_graph_fast.py
复现链中取代 rebuild_graph_typed.py（LLM 版，保留作可选高质量路径）。
"""
from app.generation.chain import RAGChain
from app.ingestion.graph_extractor import get_graph_builder

chain = RAGChain()
chunks = chain.indexer.get_all_chunks()
print(f"total chunks: {len(chunks)}")

builder = get_graph_builder()
builder.clear()  # 清空旧图 + meta（强制全量重建）

stats = builder.build_fast(chunks)
print("build stats:", stats)
print(f"nodes: {builder.graph.number_of_nodes()}  edges: {builder.graph.number_of_edges()}")

# 质量快照：关系分布 + chunk 溯源覆盖率
rels, with_chunk = {}, 0
for h, t, d in builder.graph.edges(data=True):
    rels[d.get("relation", "?")] = rels.get(d.get("relation", "?"), 0) + 1
    if d.get("chunk_id"):
        with_chunk += 1
print("top relations:", sorted(rels.items(), key=lambda x: -x[1]))
print(f"edges with chunk_id: {with_chunk}/{builder.graph.number_of_edges()}")
