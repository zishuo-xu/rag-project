"""Streamlit 对话界面 - RAG 系统前端"""

import json
import requests
import streamlit as st

# API 配置
API_BASE_URL = "http://localhost:8000"

# Trace 阶段配色（按出现顺序循环取用）
STAGE_COLORS = ["#38BDF8", "#F59E0B", "#A78BFA", "#34D399", "#F472B6", "#FACC15", "#818CF8"]

# 页面配置
st.set_page_config(
    page_title="RAG 智能问答系统",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============ 自定义 CSS ============
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; max-width: 1200px; }

    /* 页头状态徽章 */
    .status-pill {
        display: inline-flex; align-items: center; gap: 6px;
        padding: 4px 12px; border-radius: 999px;
        font-size: 0.75rem; font-weight: 600;
    }
    .status-pill.online { background: rgba(16,185,129,0.12); color: #34D399; border: 1px solid rgba(16,185,129,0.3); }
    .status-pill.offline { background: rgba(239,68,68,0.12); color: #F87171; border: 1px solid rgba(239,68,68,0.3); }

    /* 欢迎卡片 */
    .welcome-card {
        background: linear-gradient(135deg, #1E293B 0%, #172033 100%);
        border: 1px solid #334155; border-radius: 16px;
        padding: 2rem 2.2rem; margin: 1rem 0 1.2rem 0; text-align: center;
    }
    .welcome-card .big-icon { font-size: 2.6rem; margin-bottom: 0.5rem; }
    .welcome-card h3 { color: #E2E8F0; margin: 0.3rem 0; }
    .welcome-card p { color: #94A3B8; font-size: 0.85rem; margin: 0.2rem 0; }
    .welcome-card .feature-row {
        display: flex; justify-content: center; gap: 1.5rem; flex-wrap: wrap;
        margin-top: 1rem;
    }
    .welcome-card .feature {
        font-size: 0.75rem; color: #64748B;
        background: rgba(245,158,11,0.08); border: 1px solid rgba(245,158,11,0.2);
        border-radius: 8px; padding: 4px 12px;
    }

    /* 管道流程图 */
    .pipeline {
        display: flex; align-items: center; gap: 0;
        padding: 1rem 0; overflow-x: auto;
    }
    .pipe-node {
        background: #1E293B; border: 1px solid #334155;
        border-radius: 10px; padding: 0.6rem 1rem;
        text-align: center; min-width: 90px;
        transition: all 0.2s;
    }
    .pipe-node:hover { border-color: #F59E0B; transform: translateY(-2px); }
    .pipe-node .icon { font-size: 1.3rem; }
    .pipe-node .label { font-size: 0.72rem; color: #94A3B8; margin-top: 2px; }
    .pipe-node .value { font-size: 0.85rem; color: #E2E8F0; font-weight: 600; }
    .pipe-arrow { color: #475569; font-size: 1.1rem; padding: 0 0.3rem; }

    /* 向量条形图 */
    .vec-row { display: flex; align-items: center; gap: 8px; margin: 4px 0; }
    .vec-label { font-size: 0.72rem; color: #94A3B8; min-width: 42px; text-align: right; }
    .vec-bar-bg {
        flex: 1; height: 14px; background: #1E293B;
        border-radius: 7px; overflow: hidden; position: relative;
    }
    .vec-bar {
        height: 100%; border-radius: 7px;
        transition: width 0.4s ease;
    }
    .vec-val { font-size: 0.68rem; color: #64748B; min-width: 60px; }

    /* 词项标签 */
    .term-tag {
        display: inline-block; margin: 2px 3px;
        padding: 2px 8px; border-radius: 4px;
        font-size: 0.75rem; font-weight: 500;
        background: rgba(245,158,11,0.12); color: #FBBF24;
        border: 1px solid rgba(245,158,11,0.25);
    }
    .term-count { opacity: 0.6; font-size: 0.68rem; }

    /* 分块卡片 */
    .chunk-card {
        background: #1E293B; border: 1px solid #334155;
        border-left: 3px solid #F59E0B;
        border-radius: 8px; padding: 0.8rem 1rem; margin: 8px 0;
    }
    .chunk-header {
        display: flex; justify-content: space-between; align-items: center;
        margin-bottom: 6px;
    }
    .chunk-id { font-size: 0.75rem; color: #F59E0B; font-weight: 600; }
    .chunk-meta { font-size: 0.7rem; color: #64748B; }
    .chunk-text {
        font-size: 0.8rem; color: #CBD5E1; line-height: 1.5;
        max-height: 100px; overflow-y: auto;
        white-space: pre-wrap; word-break: break-all;
    }

    /* 摘要卡片 */
    .summary-card {
        background: rgba(16,185,129,0.08); border: 1px solid rgba(16,185,129,0.3);
        border-radius: 10px; padding: 1rem 1.2rem; margin: 8px 0;
    }
    .summary-card .title { font-size: 0.75rem; color: #34D399; font-weight: 600; margin-bottom: 6px; }
    .summary-card .text { font-size: 0.85rem; color: #D1FAE5; line-height: 1.6; }

    /* 来源卡片 */
    .source-card {
        background: #172033; border: 1px solid #334155;
        border-radius: 8px; padding: 0.6rem 0.9rem; margin: 6px 0;
    }
    .source-card .src-title { font-size: 0.8rem; color: #E2E8F0; font-weight: 600; }
    .source-card .src-score {
        float: right; font-size: 0.7rem; color: #38BDF8; font-weight: 600;
        background: rgba(56,189,248,0.1); border-radius: 4px; padding: 1px 8px;
    }
    .source-card .src-text {
        font-size: 0.75rem; color: #94A3B8; margin-top: 4px;
        display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;
        overflow: hidden;
    }

    /* Trace 图例 */
    .legend-row { display: flex; gap: 14px; flex-wrap: wrap; margin: 6px 0 10px 0; }
    .legend-item { display: inline-flex; align-items: center; gap: 5px; font-size: 0.72rem; color: #94A3B8; }
    .legend-dot { width: 10px; height: 10px; border-radius: 3px; display: inline-block; }

    /* 说明文字 */
    .explain { font-size: 0.78rem; color: #64748B; margin: 4px 0 12px 0; }

    /* 指标卡片 */
    div[data-testid="stMetric"] {
        background: #1E293B; border: 1px solid #334155;
        border-radius: 10px; padding: 0.8rem 1rem;
    }
    div[data-testid="stMetric"] label { color: #94A3B8 !important; font-size: 0.75rem; }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #E2E8F0 !important; font-size: 1.3rem; font-weight: 600;
    }

    /* 侧边栏 */
    section[data-testid="stSidebar"] { background: #111827; border-right: 1px solid #1F2937; }

    /* 聊天气泡 */
    div[data-testid="stChatMessage"] {
        background: #1E293B !important; border: 1px solid #334155;
        border-radius: 14px; padding: 0.5rem;
    }

    /* Expander */
    details[data-testid="stExpander"] {
        border: 1px solid #334155; border-radius: 10px; background: #1E293B;
    }

    /* 按钮 */
    .stButton > button {
        background: #D97706; color: #0F172A; border: none;
        border-radius: 8px; font-weight: 600; transition: all 0.2s;
    }
    .stButton > button:hover { background: #F59E0B; box-shadow: 0 2px 12px rgba(245,158,11,0.25); }

    /* 示例问题按钮（次级样式） */
    .example-btn .stButton > button {
        background: #1E293B; color: #CBD5E1;
        border: 1px solid #334155; font-weight: 500;
        text-align: left; white-space: normal; height: auto;
    }
    .example-btn .stButton > button:hover {
        background: #24334d; border-color: #F59E0B; color: #FBBF24;
        box-shadow: none;
    }

    /* 滚动条 */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
    ::-webkit-scrollbar-track { background: transparent; }

    hr { border-color: #1F2937 !important; }
    footer { visibility: hidden; }

    /* Tab 样式 */
    .stTabs [data-baseweb="tab"] { border-radius: 8px 8px 0 0; }
</style>
""", unsafe_allow_html=True)


# ============ API 工具函数 ============

def chat_with_api(question: str, chat_history: list, stream: bool = True):
    """调用后端 Chat API"""
    payload = {
        "question": question,
        "chat_history": chat_history,
        "top_k": st.session_state.get("top_k", 5),
        "use_query_transform": st.session_state.get("use_query_transform", True),
        "use_rerank": st.session_state.get("use_rerank", True),
        "query_strategy": st.session_state.get("query_strategy", "multi_query"),
        "stream": stream,
    }
    if stream:
        return requests.post(f"{API_BASE_URL}/api/chat", json=payload, stream=True)
    else:
        response = requests.post(f"{API_BASE_URL}/api/chat", json=payload)
        return response.json()


def upload_document(file):
    """上传文档到后端"""
    files = {"file": (file.name, file.getvalue(), file.type)}
    response = requests.post(f"{API_BASE_URL}/api/documents/upload", files=files)
    return response.json()


def get_documents():
    """获取已索引文档列表"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/documents")
        return response.json()
    except Exception:
        return {"documents": [], "total": 0}


def check_health():
    """检查后端服务状态"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/health", timeout=3)
        return response.json()
    except Exception:
        return None


def render_pipeline(data: dict):
    """渲染索引管道流程图"""
    stats = data.get("stats", {})
    nodes = [
        ("📄", "原始文档", f"{stats.get('total_chars', 0)} 字符"),
        ("✂️", "智能分块", f"{data['total']} 块"),
        ("🧮", "向量化", f"{stats.get('vector_dim', 0)} 维"),
        ("📝", "L1 摘要", "已生成" if stats.get("has_summary") else "—"),
        ("🔤", "BM25 索引", f"{stats.get('total_tokens', 0)} 词项"),
    ]
    html = '<div class="pipeline">'
    for i, (icon, label, value) in enumerate(nodes):
        if i > 0:
            html += '<span class="pipe-arrow">→</span>'
        html += (
            f'<div class="pipe-node">'
            f'<div class="icon">{icon}</div>'
            f'<div class="value">{value}</div>'
            f'<div class="label">{label}</div>'
            f'</div>'
        )
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def render_vector_bars(embeddings: list):
    """将向量渲染为条形图（前8维预览）"""
    for emb in embeddings:
        preview = emb["vector_preview"]
        max_abs = max(abs(v) for v in preview) or 1
        rows = ""
        for i, v in enumerate(preview):
            pct = abs(v) / max_abs * 100
            color = "#F59E0B" if v >= 0 else "#38BDF8"
            rows += (
                f'<div class="vec-row">'
                f'<span class="vec-label">d{i}</span>'
                f'<div class="vec-bar-bg"><div class="vec-bar" '
                f'style="width:{pct:.0f}%;background:{color}"></div></div>'
                f'<span class="vec-val">{v:+.4f}</span>'
                f'</div>'
            )
        st.markdown(
            f'<div style="margin:10px 0 4px 0;font-size:0.78rem;color:#94A3B8">'
            f'<b style="color:#E2E8F0">块 #{emb["position"]}</b> · '
            f'{emb["vector_dim"]} 维 · 模长 {emb["norm"]}'
            f'</div>{rows}',
            unsafe_allow_html=True,
        )


def render_sources(sources: list):
    """渲染参考来源卡片"""
    with st.expander(f"📚 参考来源 ({len(sources)} 篇)"):
        for j, src in enumerate(sources, 1):
            score_html = (
                f'<span class="src-score">相关度 {src["score"]:.3f}</span>'
                if src.get("score") else ""
            )
            content_escaped = (
                src["content"][:200]
                .replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            )
            st.markdown(
                f'<div class="source-card">'
                f'{score_html}'
                f'<div class="src-title">[{j}] {src["source"]}</div>'
                f'<div class="src-text">{content_escaped}…</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


def render_index_explorer():
    """索引透视主面板"""
    docs_info = get_documents()
    docs = docs_info.get("documents", [])

    if not docs:
        st.info("📂 暂无已索引文档 — 请先在左侧上传文档")
        return

    # 文档选择
    col_sel, col_btn = st.columns([3, 1])
    with col_sel:
        selected_doc = st.selectbox(
            "选择文档",
            options=[d["doc_id"] for d in docs],
            format_func=lambda x: next(
                (d["source"] for d in docs if d["doc_id"] == x), x
            ),
            label_visibility="collapsed",
        )
    with col_btn:
        explore = st.button("🔍 透视索引", use_container_width=True)

    if not explore:
        st.caption("选择文档后点击「透视索引」，查看从原文到索引的完整处理过程")
        return

    try:
        resp = requests.get(f"{API_BASE_URL}/api/documents/{selected_doc}/chunks")
        if resp.status_code != 200:
            st.error("获取索引详情失败")
            return
        data = resp.json()
    except Exception as e:
        st.error(f"请求失败: {e}")
        return

    stats = data.get("stats", {})

    # ── 管道流程图 ──
    st.markdown("##### 处理管道")
    st.markdown('<p class="explain">文档上传后依次经过以下 5 个阶段，每个阶段的产出都可检索</p>',
                unsafe_allow_html=True)
    render_pipeline(data)

    st.divider()

    # ── L1 摘要 ──
    st.markdown("##### 📝 文档摘要（L1 索引）")
    st.markdown('<p class="explain">由 LLM 生成的全文摘要，检索时先匹配摘要定位文档，再深入段落</p>',
                unsafe_allow_html=True)
    if data.get("summary"):
        st.markdown(
            f'<div class="summary-card"><div class="title">LLM 生成摘要</div>'
            f'<div class="text">{data["summary"]}</div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.warning("该文档暂无摘要")

    st.divider()

    # ── 向量 ──
    st.markdown("##### 🧮 Embedding 向量（L2 索引 · ChromaDB）")
    st.markdown(
        f'<p class="explain">每个分块被 <b>{stats.get("embedding_model", "")}</b> 编码为 '
        f'<b>{stats.get("vector_dim", 0)} 维</b>向量（下方展示前 8 维，'
        f'<span style="color:#F59E0B">■</span> 正值 / <span style="color:#38BDF8">■</span> 负值，'
        f'长度 = 绝对值大小）</p>',
        unsafe_allow_html=True,
    )
    render_vector_bars(data.get("embeddings", []))

    st.divider()

    # ── 分块 + BM25 ──
    st.markdown("##### ✂️ 分块内容 & BM25 关键词")
    st.markdown('<p class="explain">原文按 512 字符递归切分（重叠 64），每块提取高频词项构建倒排索引，'
                '用于关键词精确匹配</p>', unsafe_allow_html=True)

    for chunk in data["chunks"]:
        terms_html = "".join(
            f'<span class="term-tag">{t["term"]}<span class="term-count"> ×{t["count"]}</span></span>'
            for t in chunk["top_terms"][:8]
        )
        content_escaped = (
            chunk["content"]
            .replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        )
        st.markdown(
            f'<div class="chunk-card">'
            f'<div class="chunk-header">'
            f'<span class="chunk-id">块 #{chunk["position"]}</span>'
            f'<span class="chunk-meta">{chunk["char_count"]} 字符 · {chunk["token_count"]} 词项</span>'
            f'</div>'
            f'<div class="chunk-text">{content_escaped}</div>'
            f'<div style="margin-top:8px">{terms_html}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


def run_chat_turn(prompt: str):
    """执行一轮问答：渲染用户消息 + 流式输出助手回复"""
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    chat_history = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in st.session_state.messages[:-1]
    ]

    with st.chat_message("assistant", avatar="🧠"):
        try:
            response = chat_with_api(prompt, chat_history, stream=True)

            if response.status_code == 200:
                full_answer = ""
                retrieval_detail = {}
                sources = []
                event_type = ""

                placeholder = st.empty()
                for line in response.iter_lines():
                    if not line:
                        continue
                    line = line.decode("utf-8")
                    if line.startswith("event:"):
                        event_type = line[6:].strip()
                    elif line.startswith("data:"):
                        sse_data = line[5:].strip()
                        if event_type == "token":
                            full_answer += sse_data
                            placeholder.markdown(full_answer + "▌")
                        elif event_type == "retrieval":
                            retrieval_detail = json.loads(sse_data)
                        elif event_type == "done":
                            done_data = json.loads(sse_data)
                            sources = done_data.get("sources", [])

                placeholder.markdown(full_answer)

                if sources:
                    render_sources(sources)

                detail_display = {**retrieval_detail, "sources_count": len(sources)}
                st.session_state.retrieval_details.append(detail_display)
                with st.expander("🔎 检索链路详情"):
                    st.json(detail_display)

                st.session_state.messages.append(
                    {"role": "assistant", "content": full_answer, "sources": sources}
                )
            else:
                error_msg = f"请求失败 (HTTP {response.status_code})"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                st.session_state.retrieval_details.append({})

        except requests.ConnectionError:
            error_msg = "⚠️ 无法连接后端，请确认已启动: `python main.py`"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            st.session_state.retrieval_details.append({})


# ============ 侧边栏 ============
health = check_health()

with st.sidebar:
    st.markdown("### ⚙️ 控制台")

    if health:
        st.success(f"🟢 服务在线 · {health.get('indexed_documents', 0)} 篇文档")
    else:
        st.error("🔴 后端离线 — 请启动 `python main.py`")

    st.divider()
    st.markdown("#### 🎯 检索策略")
    st.session_state.top_k = st.slider("召回 Top-K", 1, 20, 5)
    col_a, col_b = st.columns(2)
    with col_a:
        st.session_state.use_query_transform = st.checkbox("查询改写", value=True)
    with col_b:
        st.session_state.use_rerank = st.checkbox("Rerank", value=True)
    st.session_state.query_strategy = st.selectbox(
        "改写策略",
        ["multi_query", "hyde", "none"],
        format_func=lambda x: {
            "multi_query": "Multi-Query（多变体）",
            "hyde": "HyDE（假设文档）",
            "none": "原始查询",
        }[x],
    )

    st.divider()
    st.markdown("#### 📄 上传文档")
    uploaded_file = st.file_uploader(
        "选择文件", type=["pdf", "txt", "md"],
        help="支持 PDF / TXT / Markdown",
    )
    if uploaded_file:
        if st.button("⬆️ 上传并索引", use_container_width=True):
            with st.spinner("正在处理文档…"):
                result = upload_document(uploaded_file)
                if "message" in result:
                    st.toast(f"✅ {result['message']}（{result.get('num_chunks', 0)} 个分块）", icon="📄")
                    st.rerun()
                else:
                    st.error(f"上传失败: {result.get('detail', '未知错误')}")

    docs_info = get_documents()
    if docs_info.get("documents"):
        st.divider()
        st.markdown(f"**已索引 ({docs_info['total']})**")
        for doc in docs_info["documents"]:
            st.caption(f"📎 {doc['source']} · {doc['num_chunks']} 块")

    st.divider()
    if st.button("🗑️ 清除对话", use_container_width=True):
        st.session_state.messages = []
        st.session_state.retrieval_details = []
        st.rerun()


# ============ 主界面：Tab 切换 ============
if health:
    status_html = (
        f'<span class="status-pill online">● 在线 · {health.get("indexed_documents", 0)} 篇文档</span>'
    )
else:
    status_html = '<span class="status-pill offline">● 后端离线</span>'

st.markdown(
    f'<div style="display:flex;align-items:center;gap:14px;flex-wrap:wrap">'
    f'<h2 style="margin:0">🧠 RAG 智能问答</h2>{status_html}</div>',
    unsafe_allow_html=True,
)
st.caption("多路召回 · RRF 融合 · Cross-Encoder 重排 · DeepSeek 生成")

tab_chat, tab_compare, tab_trace, tab_explore, tab_graph = st.tabs(
    ["💬 对话问答", "🧪 对比实验", "📊 观测台", "🔬 索引透视", "🕸️ 知识图谱"]
)

# ── Tab 1: 对话 ──
with tab_chat:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "retrieval_details" not in st.session_state:
        st.session_state.retrieval_details = []

    # 空状态：欢迎卡片 + 示例问题
    if not st.session_state.messages and not st.session_state.get("pending_question"):
        st.markdown(
            '<div class="welcome-card">'
            '<div class="big-icon">🧠</div>'
            '<h3>你好，我是 RAG 智能助手</h3>'
            '<p>基于已索引的文档回答问题，全程可观测检索链路</p>'
            '<div class="feature-row">'
            '<span class="feature">🔍 多路召回</span>'
            '<span class="feature">🔀 RRF 融合</span>'
            '<span class="feature">📊 Cross-Encoder 重排</span>'
            '<span class="feature">📝 流式生成</span>'
            '</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p class="explain" style="text-align:center">试试这些示例问题，或在下方输入你的问题：</p>',
            unsafe_allow_html=True,
        )
        examples = [
            "Docker 和 Kubernetes 有什么关系？",
            "Redis 的持久化机制有哪些？",
            "数据库索引为什么使用 B+ 树？",
        ]
        ec = st.columns(3)
        for i, q in enumerate(examples):
            with ec[i]:
                st.markdown('<div class="example-btn">', unsafe_allow_html=True)
                if st.button(q, key=f"example_{i}", use_container_width=True):
                    st.session_state.pending_question = q
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

    # 历史消息（assistant 的检索详情按独立计数器对齐）
    detail_idx = 0
    for message in st.session_state.messages:
        avatar = "👤" if message["role"] == "user" else "🧠"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                if message.get("sources"):
                    render_sources(message["sources"])
                if detail_idx < len(st.session_state.retrieval_details):
                    detail = st.session_state.retrieval_details[detail_idx]
                    if detail:
                        with st.expander("🔎 检索链路详情"):
                            st.json(detail)
                detail_idx += 1

    if prompt := st.chat_input("输入你的问题…"):
        st.session_state.pending_question = prompt

    pending = st.session_state.pop("pending_question", None)
    if pending:
        run_chat_turn(pending)

# ── Tab 2: 对比实验 ──
with tab_compare:
    st.markdown("##### 检索策略对比")
    st.markdown(
        '<p class="explain">同一问题分别用 4 种策略检索，对比命中结果和耗时，'
        '直观展示多路召回 + 重排的增益效果</p>',
        unsafe_allow_html=True,
    )
    compare_q = st.text_input("输入测试问题", placeholder="例如：什么是RAG？", key="compare_input")
    if st.button("🚀 运行对比", use_container_width=True, key="compare_btn") and compare_q:
        with st.spinner("正在运行 4 种检索策略…"):
            try:
                resp = requests.post(
                    f"{API_BASE_URL}/api/retrieval/compare",
                    json={"question": compare_q, "top_k": 5},
                )
                if resp.status_code == 200:
                    st.session_state.compare_result = resp.json()
                else:
                    st.error(f"对比失败: HTTP {resp.status_code}")
            except requests.ConnectionError:
                st.error("无法连接后端服务")

    cdata = st.session_state.get("compare_result")
    if cdata:
        strategies = cdata["strategies"]

        # 概览表格
        cols = st.columns(len(strategies))
        colors = ["#38BDF8", "#FBBF24", "#A78BFA", "#34D399"]
        for i, (key, s) in enumerate(strategies.items()):
            with cols[i]:
                st.markdown(
                    f'<div style="background:#1E293B;border:1px solid #334155;'
                    f'border-top:3px solid {colors[i % len(colors)]};border-radius:10px;'
                    f'padding:12px;text-align:center">'
                    f'<div style="font-size:0.75rem;color:#94A3B8">{s["name"]}</div>'
                    f'<div style="font-size:1.4rem;font-weight:700;color:#E2E8F0">'
                    f'{s["time_ms"]}ms</div>'
                    f'<div style="font-size:0.72rem;color:#64748B">'
                    f'命中 {s["count"]} 篇 · 重叠 {s["overlap_with_rerank"]}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        st.divider()

        # 各策略结果详情
        for key, s in strategies.items():
            with st.expander(f"{s['name']} · {s['time_ms']}ms · {s['count']} 篇"):
                for j, r in enumerate(s["results"], 1):
                    score_str = f" · score={r['score']:.3f}" if r.get("score") else ""
                    st.markdown(f"**[{j}]** {r['source']}{score_str}")
                    st.caption(r["content"][:120] + "…")
                    if j < len(s["results"]):
                        st.divider()

# ── Tab 3: 观测台 ──
with tab_trace:
    st.markdown("##### 管道追踪（Trace Waterfall）")
    st.markdown(
        '<p class="explain">每次问答自动记录完整管道耗时。'
        '在「对话问答」中提问后回到这里查看瀑布图</p>',
        unsafe_allow_html=True,
    )

    if st.button("🔄 刷新追踪数据", key="trace_refresh"):
        st.rerun()

    try:
        # 统计概览
        stats_resp = requests.get(f"{API_BASE_URL}/api/traces/stats")
        if stats_resp.status_code == 200:
            tstats = stats_resp.json()
            if tstats.get("total_traces", 0) > 0:
                sc = st.columns(3)
                sc[0].metric("总调用次数", tstats["total_traces"])
                sc[1].metric("平均总耗时", f"{tstats['avg_total_ms']}ms")
                stage_avg = tstats.get("stage_avg", {})
                slowest = max(stage_avg.items(), key=lambda x: x[1]) if stage_avg else ("-", 0)
                sc[2].metric("最慢阶段", f"{slowest[0]}", f"{slowest[1]}ms")

                # 各阶段平均耗时条形图（阶段配色）
                stage_color_map = {
                    name: STAGE_COLORS[i % len(STAGE_COLORS)]
                    for i, name in enumerate(stage_avg)
                }
                legend_html = '<div class="legend-row">' + "".join(
                    f'<span class="legend-item">'
                    f'<span class="legend-dot" style="background:{color}"></span>{name}</span>'
                    for name, color in stage_color_map.items()
                ) + '</div>'

                st.markdown("**各阶段平均耗时**")
                max_val = max(stage_avg.values()) if stage_avg else 1
                for name, val in stage_avg.items():
                    pct = val / max_val * 100
                    st.markdown(
                        f'<div class="vec-row">'
                        f'<span class="vec-label" style="min-width:110px">{name}</span>'
                        f'<div class="vec-bar-bg"><div class="vec-bar" '
                        f'style="width:{pct:.0f}%;background:{stage_color_map[name]}"></div></div>'
                        f'<span class="vec-val">{val}ms</span></div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.info("暂无追踪数据 — 请先在「对话问答」中提问")
                stage_color_map = {}
        else:
            stage_color_map = {}

        st.divider()

        # 最近 traces 列表
        traces_resp = requests.get(f"{API_BASE_URL}/api/traces?limit=10")
        if traces_resp.status_code == 200:
            traces = traces_resp.json().get("traces", [])
            if traces:
                st.markdown("**最近调用记录**")
                for t in traces:
                    with st.expander(
                        f"❓ {t['question'][:40]}{'…' if len(t['question'])>40 else ''} · {t['total_ms']}ms"
                    ):
                        # 为该 trace 构建阶段配色（沿用全局映射，新阶段追加）
                        for span in t["spans"]:
                            if span["name"] not in stage_color_map:
                                stage_color_map[span["name"]] = STAGE_COLORS[
                                    len(stage_color_map) % len(STAGE_COLORS)
                                ]
                        legend_items = "".join(
                            f'<span class="legend-item">'
                            f'<span class="legend-dot" style="background:{stage_color_map[s["name"]]}"></span>'
                            f'{s["name"]}</span>'
                            for s in t["spans"]
                        )
                        st.markdown(
                            f'<div class="legend-row">{legend_items}</div>',
                            unsafe_allow_html=True,
                        )

                        # 瀑布图
                        total = t["total_ms"] or 1
                        for span in t["spans"]:
                            left_pct = span["start_ms"] / total * 100
                            width_pct = max(span["duration_ms"] / total * 100, 2)
                            st.markdown(
                                f'<div style="display:flex;align-items:center;gap:6px;margin:3px 0">'
                                f'<span style="font-size:0.7rem;color:#94A3B8;min-width:100px;'
                                f'text-align:right">{span["name"]}</span>'
                                f'<div style="flex:1;height:16px;background:#1E293B;'
                                f'border-radius:4px;position:relative;overflow:hidden">'
                                f'<div style="position:absolute;left:{left_pct:.0f}%;'
                                f'width:{width_pct:.0f}%;height:100%;'
                                f'background:{stage_color_map[span["name"]]};'
                                f'border-radius:4px"></div></div>'
                                f'<span style="font-size:0.68rem;color:#64748B;min-width:50px">'
                                f'{span["duration_ms"]}ms</span></div>',
                                unsafe_allow_html=True,
                            )
                        if t.get("answer_preview"):
                            st.caption(f"💬 {t['answer_preview'][:80]}…")
    except requests.ConnectionError:
        st.error("无法连接后端服务")

# ── Tab 4: 索引透视 ──
with tab_explore:
    render_index_explorer()

# ── Tab 5: 知识图谱 ──
with tab_graph:
    st.markdown("##### 🕸️ 知识图谱 (Graph RAG)")
    st.markdown(
        '<p class="explain">从已索引文档中抽取实体和关系，构建知识图谱，'
        '支持关系查询和多跳推理，作为向量检索的补充</p>',
        unsafe_allow_html=True,
    )

    # 图谱操作区
    col_build, col_clear = st.columns([2, 1])
    with col_build:
        if st.button("🚀 构建知识图谱", use_container_width=True, key="graph_build"):
            with st.spinner("正在从文档中抽取实体和关系…"):
                try:
                    resp = requests.post(f"{API_BASE_URL}/api/graph/build", timeout=300)
                    if resp.status_code == 200:
                        data = resp.json()
                        st.success(f"✅ {data['message']}")
                        st.session_state.graph_stats = data.get("stats", {})
                        st.rerun()
                    else:
                        st.error(f"构建失败: {resp.json().get('detail', resp.text)}")
                except requests.ConnectionError:
                    st.error("无法连接后端服务")
                except requests.Timeout:
                    st.error("构建超时（文档较多时可能需要几分钟）")
    with col_clear:
        if st.button("🗑️ 清空图谱", use_container_width=True, key="graph_clear"):
            try:
                requests.delete(f"{API_BASE_URL}/api/graph")
                st.session_state.pop("graph_stats", None)
                st.rerun()
            except Exception:
                pass

    # 图谱统计
    try:
        stats_resp = requests.get(f"{API_BASE_URL}/api/graph/stats")
        if stats_resp.status_code == 200:
            gstats = stats_resp.json()
            if not gstats.get("is_empty", True):
                st.divider()
                sc = st.columns(3)
                sc[0].metric("实体数", gstats["num_nodes"])
                sc[1].metric("关系数", gstats["num_edges"])
                sc[2].metric("图密度", gstats.get("density", 0))

                # Top 实体
                if gstats.get("top_entities"):
                    st.markdown("**🔥 核心实体（度最高）**")
                    top_html = ""
                    for ent in gstats["top_entities"][:8]:
                        top_html += (
                            f'<span class="term-tag">{ent["entity"]}'
                            f'<span class="term-count"> ×{ent["degree"]}</span></span>'
                        )
                    st.markdown(top_html, unsafe_allow_html=True)
            else:
                st.info("📂 知识图谱为空 — 点击「构建知识图谱」从已索引文档中抽取实体和关系")
    except requests.ConnectionError:
        pass

    st.divider()

    # 图检索查询
    st.markdown("##### 🔍 图检索查询")
    graph_q = st.text_input(
        "输入问题（查看图谱中的关系）",
        placeholder="例如：Redis 的 ZSet 用了什么数据结构？",
        key="graph_query_input",
    )
    if st.button("🔎 查询图谱", use_container_width=True, key="graph_query_btn") and graph_q:
        with st.spinner("正在检索知识图谱…"):
            try:
                resp = requests.post(
                    f"{API_BASE_URL}/api/graph/query",
                    json={"question": graph_q},
                )
                if resp.status_code == 200:
                    gdata = resp.json()
                    st.session_state.graph_query_result = gdata
                else:
                    st.error(f"查询失败: {resp.json().get('detail', '')}")
            except requests.ConnectionError:
                st.error("无法连接后端服务")

    gresult = st.session_state.get("graph_query_result")
    if gresult:
        if gresult.get("entities"):
            st.markdown(f"**识别实体**: {', '.join(gresult['entities'])}")

            if gresult.get("relations"):
                st.markdown(f"**找到 {len(gresult['relations'])} 条关系**")
                for r in gresult["relations"][:15]:
                    st.markdown(
                        f'<div class="chunk-card" style="border-left-color:#A78BFA">'
                        f'<span style="color:#A78BFA;font-weight:600">{r["head"]}</span>'
                        f' <span style="color:#64748B">—[{r["relation"]}]→</span> '
                        f'<span style="color:#34D399;font-weight:600">{r["tail"]}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            if gresult.get("context_text"):
                with st.expander("📝 图上下文（送入 LLM）"):
                    st.code(gresult["context_text"], language=None)
        else:
            st.warning("未在图谱中找到相关实体")

    st.divider()

    # 路径查询
    st.markdown("##### 🛤️ 实体路径查询")
    st.markdown('<p class="explain">查找两个实体之间的关系路径（多跳推理）</p>', unsafe_allow_html=True)
    pc1, pc2 = st.columns(2)
    with pc1:
        path_source = st.text_input("起始实体", placeholder="例如：Redis", key="path_src")
    with pc2:
        path_target = st.text_input("目标实体", placeholder="例如：B+树", key="path_tgt")

    if st.button("🔗 查找路径", use_container_width=True, key="path_btn") and path_source and path_target:
        try:
            resp = requests.get(
                f"{API_BASE_URL}/api/graph/path",
                params={"source": path_source, "target": path_target},
            )
            if resp.status_code == 200:
                pdata = resp.json()
                if pdata.get("found"):
                    st.success(f"✅ 找到路径（{pdata['path_length']} 跳）")
                    path_html = f'<div style="padding:12px;background:#1E293B;border-radius:10px;border:1px solid #334155">'
                    path_html += f'<span style="color:#F59E0B;font-weight:700">{path_source}</span>'
                    for step in pdata["path"]:
                        path_html += (
                            f' <span style="color:#64748B">—[{step["relation"]}]→</span> '
                            f'<span style="color:#38BDF8;font-weight:600">{step["to"]}</span>'
                        )
                    path_html += '</div>'
                    st.markdown(path_html, unsafe_allow_html=True)
                else:
                    st.warning(f"未找到 {path_source} 到 {path_target} 的路径")
        except requests.ConnectionError:
            st.error("无法连接后端服务")

    # 三元组浏览
    st.divider()
    with st.expander("📜 浏览所有三元组"):
        try:
            resp = requests.get(f"{API_BASE_URL}/api/graph/triples?limit=50")
            if resp.status_code == 200:
                tdata = resp.json()
                triples = tdata.get("triples", [])
                if triples:
                    for t in triples:
                        st.caption(f"{t['head']} —[{t['relation']}]→ {t['tail']}  ·  {t.get('source', '')}")
                else:
                    st.info("图谱为空")
        except requests.ConnectionError:
            st.error("无法连接后端")
