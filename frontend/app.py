"""Streamlit 对话界面 - RAG 系统前端"""

import json
import requests
import streamlit as st

# API 配置
API_BASE_URL = "http://localhost:8000"

# 页面配置
st.set_page_config(
    page_title="RAG 智能问答系统",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


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
        response = requests.post(
            f"{API_BASE_URL}/api/chat",
            json=payload,
            stream=True,
        )
        return response
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


# ============ 侧边栏 ============
with st.sidebar:
    st.header("⚙️ 系统设置")

    # 服务状态
    health = check_health()
    if health:
        st.success(f"服务运行中 | 已索引 {health.get('indexed_documents', 0)} 个文档")
    else:
        st.error("后端服务未启动，请运行: python main.py")

    st.divider()

    # 检索参数
    st.subheader("检索参数")
    st.session_state.top_k = st.slider("Top-K (返回文档数)", 1, 20, 5)
    st.session_state.use_query_transform = st.checkbox("查询改写", value=True)
    st.session_state.use_rerank = st.checkbox("Rerank 重排序", value=True)
    st.session_state.query_strategy = st.selectbox(
        "查询改写策略",
        ["multi_query", "hyde", "none"],
        format_func=lambda x: {
            "multi_query": "Multi-Query (多变体)",
            "hyde": "HyDE (假设文档)",
            "none": "无 (原始查询)",
        }[x],
    )

    st.divider()

    # 文档上传
    st.subheader("📄 文档管理")
    uploaded_file = st.file_uploader(
        "上传文档",
        type=["pdf", "txt", "md"],
        help="支持 PDF、TXT、Markdown 格式",
    )
    if uploaded_file:
        if st.button("上传并索引"):
            with st.spinner("正在处理文档..."):
                result = upload_document(uploaded_file)
                if "message" in result:
                    st.success(result["message"])
                    st.info(f"生成 {result.get('num_chunks', 0)} 个分块")
                else:
                    st.error(f"上传失败: {result.get('detail', '未知错误')}")

    # 文档列表
    docs_info = get_documents()
    if docs_info.get("documents"):
        st.write(f"**已索引文档 ({docs_info['total']})**")
        for doc in docs_info["documents"]:
            st.text(f"  • {doc['source']} ({doc['num_chunks']} 块)")


# ============ 主界面 ============
st.title("🔍 RAG 智能问答系统")
st.caption("多路召回 + RRF融合 + Cross-Encoder重排序 | 基于 LangChain + ChromaDB")

# 初始化对话历史
if "messages" not in st.session_state:
    st.session_state.messages = []
if "retrieval_details" not in st.session_state:
    st.session_state.retrieval_details = []

# 显示历史消息
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # 显示检索详情（仅 assistant 消息）
        if message["role"] == "assistant" and i < len(st.session_state.retrieval_details):
            detail = st.session_state.retrieval_details[i]
            if detail:
                with st.expander("🔎 检索详情"):
                    st.json(detail)

# 用户输入
if prompt := st.chat_input("输入你的问题..."):
    # 显示用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 构建对话历史
    chat_history = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in st.session_state.messages[:-1]
    ]

    # 调用 API（流式）
    with st.chat_message("assistant"):
        try:
            response = chat_with_api(prompt, chat_history, stream=True)

            if response.status_code == 200:
                full_answer = ""
                retrieval_detail = {}
                sources = []

                # 解析 SSE 流
                placeholder = st.empty()
                for line in response.iter_lines():
                    if not line:
                        continue
                    line = line.decode("utf-8")

                    if line.startswith("event:"):
                        event_type = line[6:].strip()
                    elif line.startswith("data:"):
                        data = line[5:].strip()

                        if event_type == "token":
                            full_answer += data
                            placeholder.markdown(full_answer + "▌")
                        elif event_type == "retrieval":
                            retrieval_detail = json.loads(data)
                        elif event_type == "done":
                            done_data = json.loads(data)
                            sources = done_data.get("sources", [])

                placeholder.markdown(full_answer)

                # 显示来源
                if sources:
                    with st.expander(f"📚 参考来源 ({len(sources)} 篇)"):
                        for j, src in enumerate(sources, 1):
                            score_str = f" (分数: {src['score']:.3f})" if src.get("score") else ""
                            st.markdown(f"**[{j}] {src['source']}**{score_str}")
                            st.text(src["content"][:200] + "...")
                            st.divider()

                # 保存检索详情
                detail_display = {
                    **retrieval_detail,
                    "sources_count": len(sources),
                }
                st.session_state.retrieval_details.append(detail_display)

                with st.expander("🔎 检索详情"):
                    st.json(detail_display)

                st.session_state.messages.append(
                    {"role": "assistant", "content": full_answer}
                )
            else:
                error_msg = f"请求失败 (HTTP {response.status_code})"
                st.error(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg}
                )
                st.session_state.retrieval_details.append({})

        except requests.ConnectionError:
            error_msg = "无法连接到后端服务，请确认已启动: `python main.py`"
            st.error(error_msg)
            st.session_state.messages.append(
                {"role": "assistant", "content": error_msg}
            )
            st.session_state.retrieval_details.append({})

    # 清除对话按钮
    if st.button("🗑️ 清除对话"):
        st.session_state.messages = []
        st.session_state.retrieval_details = []
        st.rerun()
