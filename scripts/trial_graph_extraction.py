"""小批试跑：验证升级后的 LLM 类型化三元组抽取质量（抽 4 个不同文档的块，不写生产图）"""
import json

from app.generation.chain import RAGChain
from app.ingestion.graph_extractor import KnowledgeGraphBuilder

chain = RAGChain()
chunks = chain.indexer.get_all_chunks()
print(f"total chunks: {len(chunks)}")

# 选 4 个不同文档、内容充实的块
seen, picked = set(), []
for c in chunks:
    src = c.metadata.get("source", "")
    if src not in seen and len(c.page_content) > 150:
        seen.add(src)
        picked.append(c)
    if len(picked) >= 4:
        break

builder = KnowledgeGraphBuilder()  # 仅用其 extract_triples，不 save
for c in picked:
    triples = builder.extract_triples(c.page_content)
    print(f"\n=== {c.metadata.get('source')} | {c.metadata.get('chunk_id')} | {len(triples)} triples ===")
    print("文本:", c.page_content[:150].replace("\n", " "), "...")
    for t in triples:
        print(f"  ({t['head']}:{t['head_type']}) —[{t['relation']}]→ ({t['tail']}:{t['tail_type']})")
