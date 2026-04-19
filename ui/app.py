# -*- coding: utf-8 -*-
"""
敦煌文化遗产智能知识库系统 — Streamlit 主界面

页面结构：
    - 首页：项目介绍 + 引导示例
    - 智能问答：对话式 RAG 问答
    - 语义检索：关键词搜索文献片段
    - 数据看板：可视化图表
    - 关于：项目背景与技术架构
"""

import os
import sys
from pathlib import Path

import streamlit as st

# 确保项目根目录在 Python 路径中
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit.components.v1 as components
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

from config import get_config, PROJECT_ROOT
from core.pdf_parser import PDFParser
from core.chunker import TextChunker
from core.vectorizer import VectorSearchEngine
from core.rag_engine import RAGEngine
from utils.logger import setup_logger

logger = setup_logger("ui")

# ── 初始化配置 ─────────────────────────────────────────────
@st.cache_resource
def load_config():
    """加载全局配置（Streamlit 缓存，只执行一次）。"""
    return get_config()


@st.cache_resource
def init_engine(_cfg: dict) -> VectorSearchEngine:
    """初始化向量检索引擎，优先从缓存加载。

    注意：此函数被 @st.cache_resource 装饰，内部不能调用任何 st 元素
    （st.toast / st.spinner 等），否则会在 cache replay 时报错。
    初始化状态通过返回 engine.load_status 传递给调用方。
    """
    engine = VectorSearchEngine(
        model_name=_cfg["vectorizer"]["model_name"],
        embedding_dim=_cfg["vectorizer"]["embedding_dim"],
        default_top_k=_cfg["vectorizer"]["default_top_k"],
        similarity_threshold=_cfg["vectorizer"]["similarity_threshold"],
    )

    index_file = _cfg["paths"]["index_file"]
    docs_file = _cfg["paths"]["docs_file"]
    metadata_file = _cfg["paths"]["metadata_file"]

    if engine.load_index(index_file, docs_file, metadata_file):
        engine.load_status = "loaded"
    else:
        # 自动构建（纯数据处理，不调用任何 st 元素）
        parser = PDFParser(
            pdf_dir=_cfg["paths"]["pdf_dir"],
            max_chars_per_page=_cfg["pdf_parser"]["max_chars_per_page"],
        )
        documents = parser.parse_all()

        if documents:
            chunker = TextChunker(
                chunk_size=_cfg["chunker"]["chunk_size"],
                overlap=_cfg["chunker"]["overlap"],
                min_length=_cfg["chunker"]["min_chunk_length"],
            )
            chunks, metadata = chunker.chunk_documents(documents)
            engine.build_index(chunks, metadata)
            engine.save_index(index_file, docs_file, metadata_file)
            engine.load_status = "built"
        else:
            engine.load_status = "empty"

    return engine


@st.cache_resource
def init_rag(_engine: VectorSearchEngine, _cfg: dict) -> RAGEngine:
    """初始化 RAG 引擎。"""
    return RAGEngine(
        vector_engine=_engine,
        system_prompt=_cfg["rag"]["system_prompt"],
        llm_provider=_cfg["rag"]["llm_provider"],
        llm_base_url=os.environ.get("DUNHUANG_API_BASE", _cfg["rag"].get("llm_base_url", "")),
        llm_model=os.environ.get("DUNHUANG_MODEL", _cfg["rag"]["llm_model"]),
        llm_max_tokens=_cfg["rag"]["llm_max_tokens"],
        llm_temperature=_cfg["rag"]["llm_temperature"],
    )


# ── 页面配置 ───────────────────────────────────────────────
cfg = load_config()

st.set_page_config(
    page_title=cfg["ui"]["page_title"],
    page_icon=cfg["ui"]["page_icon"],
    layout=cfg["ui"]["layout"],
    initial_sidebar_state="expanded",
)

# 自定义 CSS
st.markdown("""
<style>
    /* 侧边栏 */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    [data-testid="stSidebar"] * {
        color: #e0e0e0 !important;
    }
    
    /* 卡片样式 */
    .card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1rem;
    }
    .card h3 {
        margin-top: 0;
        font-size: 1.1rem;
    }
    
    /* 来源标签 */
    .source-tag {
        display: inline-block;
        background: #f0f2f6;
        border: 1px solid #ddd;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.8rem;
        margin: 2px;
        color: #555;
    }
    
    /* 引导示例按钮 */
    .example-btn {
        display: inline-block;
        background: white;
        border: 1px solid #667eea;
        border-radius: 20px;
        padding: 6px 16px;
        font-size: 0.85rem;
        color: #667eea;
        cursor: pointer;
        margin: 4px;
        transition: all 0.2s;
    }
    .example-btn:hover {
        background: #667eea;
        color: white;
    }
    
    /* 评分条 */
    .score-bar {
        height: 8px;
        border-radius: 4px;
        background: #e0e0e0;
        overflow: hidden;
    }
    .score-fill {
        height: 100%;
        border-radius: 4px;
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
</style>
""", unsafe_allow_html=True)


# ── 侧边栏 ─────────────────────────────────────────────────
def render_sidebar():
    """渲染侧边栏内容。"""
    with st.sidebar:
        st.image(
            "https://img.icons8.com/color/96/dunhuang.png",
            width=60,
        )
        st.title("🏛️ 敦煌知识库")
        st.caption("Dunhuang Heritage AI Knowledge Base")
        st.divider()

        # 使用引导
        st.markdown("#### 📖 使用引导")
        st.markdown("""
        **新用户？按以下步骤开始：**

        1. 🏠 先看**首页**了解项目
        2. 💬 到**智能问答**提个问题
        3. 🔍 或试试**语义检索**找文献
        4. 📊 查看**数据看板**看可视化

        > 💡 不知从何下手？点击首页的示例问题快速体验！
        """)

        st.divider()
        st.markdown("#### ⚙️ API 配置")
        api_key = st.text_input(
            "API Key（可选）",
            type="password",
            help="输入 DeepSeek 或 OpenAI API Key 以启用 AI 生成模式。留空则使用检索摘要模式。",
            key="api_key_input",
        )
        if api_key:
            os.environ["DUNHUANG_API_KEY"] = api_key
            st.success("API Key 已设置 ✅")
        else:
            st.info("当前为检索摘要模式")

        st.divider()
        # 知识库状态
        with st.spinner("正在初始化知识库（首次运行需要几分钟）..."):
            engine = init_engine(cfg)

        # 根据 init_engine 的返回状态显示提示
        _status = getattr(engine, "load_status", None)
        if _status == "loaded":
            st.toast("知识库索引已加载", icon="✅")
        elif _status == "built":
            st.toast("知识库索引已构建并缓存", icon="✅")

        if engine.embeddings is not None:
            st.markdown(f"#### 📚 知识库状态")
            st.metric("文本块数量", f"{engine.embeddings.shape[0]:,}")
            st.metric("PDF 文献", f"{len(set(m['source'] for m in engine.metadata))}")
        else:
            st.warning("知识库尚未构建")


# ── 首页 ───────────────────────────────────────────────────
def render_home():
    """首页：项目介绍 + 引导示例。"""
    st.title("🏛️ 敦煌文化遗产智能知识库系统")
    st.markdown("*基于 RAG 架构的敦煌学智能问答与知识检索平台*")

    # Hero 区域
    st.markdown("""
    <div class="card">
        <h3>🎯 一句话描述</h3>
        <p>基于 35 篇学术文献、覆盖 4 大纹样类别、横跨 1000+ 年历史的敦煌文化遗产智能知识库，
        支持自然语言问答与语义化检索，每条回答均标注文献来源。</p>
    </div>
    """, unsafe_allow_html=True)

    # 核心价值
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        #### 🔍 语义检索
        不需要精确关键词，用自然语言描述你想了解的内容，系统自动理解语义并找到相关文献。
        """)
    with col2:
        st.markdown("""
        #### 📚 来源可溯
        每条回答都标注了出处，支持学术引用。不再是"AI 说的"，而是"文献说的"。
        """)
    with col3:
        st.markdown("""
        #### 🌏 多语言支持
        底层向量模型覆盖 50+ 种语言，可检索中、英、日文敦煌学研究文献。
        """)

    st.divider()

    # 引导示例
    st.markdown("#### 🚀 不知道问什么？试试这些 →")
    examples = cfg["ui"].get("example_questions", [])

    cols = st.columns(2)
    for i, q in enumerate(examples):
        col = cols[i % 2]
        if col.button(q, key=f"example_{i}", use_container_width=True):
            st.session_state["page"] = "智能问答"
            st.session_state["prefill_question"] = q
            st.rerun()

    # 技术架构简介
    st.divider()
    with st.expander("🔧 查看技术架构"):
        st.markdown("""
        ```
        用户提问（自然语言）
                ↓
        ┌─────────────────────────────────┐
        │  1. 向量化（Embedding）           │  ← paraphrase-multilingual-MiniLM-L12-v2
        │  2. 语义检索（Vector Search）     │  ← 余弦相似度 Top-K 召回
        │  3. 上下文注入（Prompt Build）    │  ← 组装 RAG Context
        │  4. 生成回答（LLM）              │  ← DeepSeek / OpenAI 兼容接口
        └─────────────────────────────────┘
                ↓
        带来源标注的准确回答
        ```
        """)

    # 数据概览
    st.divider()
    st.markdown("#### 📊 数据概览")
    metrics = [
        ("📄 文献收录", "35 篇"),
        ("🎨 纹样分类", "4 大类 15+ 子类"),
        ("⏳ 历史跨度", "北朝 → 元代（1000+ 年）"),
        ("🏛️ 洞窟覆盖", "492 个"),
        ("🖌️ 矿物颜料", "8 种"),
    ]
    cols = st.columns(len(metrics))
    for col, (label, value) in zip(cols, metrics):
        col.metric(label, value)


# ── 智能问答 ───────────────────────────────────────────────
def render_qa():
    """智能问答页面：对话式 RAG 问答。"""
    st.title("💬 智能问答")
    st.caption("基于 RAG 架构的敦煌学智能问答 — 每条回答均附来源标注")

    rag = init_rag(init_engine(cfg), cfg)

    # 预填问题（从首页示例跳转）
    prefill = st.session_state.pop("prefill_question", None)

    # 对话历史
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # 显示历史
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                st.markdown("---")
                st.markdown("**📚 参考来源：**")
                for src in msg["sources"]:
                    st.markdown(f"<span class='source-tag'>📄 {src}</span>", unsafe_allow_html=True)

    # 输入框
    question = st.chat_input(
        "请输入您关于敦煌文化的问题...",
        key="qa_input",
        default=prefill,
    )

    if question:
        # 用户消息
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # 生成回答
        with st.chat_message("assistant"):
            with st.spinner("正在检索文献并生成回答..."):
                result = rag.answer(question, top_k=5)

            st.markdown(result["answer"])

            if result.get("sources"):
                st.markdown("---")
                st.markdown("**📚 参考来源：**")
                for src in result["sources"]:
                    st.markdown(f"<span class='source-tag'>📄 {src}</span>", unsafe_allow_html=True)

                # 展开查看详细检索结果
                with st.expander("🔍 查看检索详情"):
                    for i, r in enumerate(result["search_results"], 1):
                        score = r["score"]
                        st.markdown(f"**结果 {i}** — {r['metadata'].get('source', '未知')}")
                        st.markdown(
                            f"<div class='score-bar'><div class='score-fill' style='width:{score*100:.0f}%'></div></div>"
                            f" 相关度：{score:.1%}",
                            unsafe_allow_html=True,
                        )
                        st.markdown(f"> {r['document'][:200]}{'...' if len(r['document']) > 200 else ''}")
                        st.markdown("---")

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result.get("sources", []),
        })

    # 清空对话按钮
    if st.session_state.chat_history:
        if st.button("🗑️ 清空对话", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()


# ── 语义检索 ───────────────────────────────────────────────
def render_search():
    """语义检索页面：关键词/自然语言搜索文献片段。"""
    st.title("🔍 语义检索")
    st.caption("输入任意关键词或自然语言描述，系统返回语义最相关的文献片段")

    engine = init_engine(cfg)

    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input(
            "搜索内容",
            placeholder="例如：敦煌藻井的莲花纹样有什么特征？",
            label_visibility="collapsed",
        )
    with col2:
        top_k = st.selectbox("返回数量", [3, 5, 10], index=1)

    search_clicked = st.button("🔍 搜索", type="primary", use_container_width=True)

    if query and search_clicked:
        with st.spinner("正在检索..."):
            results = engine.search(query, top_k=top_k)

        if not results:
            st.warning("未找到相关内容，请尝试换一种表述。")
        else:
            st.success(f"找到 {len(results)} 条相关结果")
            for i, r in enumerate(results, 1):
                score = r["score"]
                source = r["metadata"].get("source", "未知来源")
                chunk_id = r["metadata"].get("chunk_id", "?")

                with st.container():
                    st.markdown(f"### 结果 {i}")
                    col_a, col_b = st.columns([3, 1])
                    with col_a:
                        st.markdown(f"📄 **来源：** {source} （片段 {chunk_id}）")
                    with col_b:
                        st.markdown(f"🎯 **相关度：** {score:.1%}")

                    # 相关度进度条
                    st.markdown(
                        f"<div class='score-bar'><div class='score-fill' style='width:{score*100:.0f}%'></div></div>",
                        unsafe_allow_html=True,
                    )

                    # 文本内容（可展开）
                    doc_text = r["document"]
                    if len(doc_text) > 300:
                        with st.expander("查看完整内容"):
                            st.markdown(doc_text)
                        st.markdown(doc_text[:300] + "...")
                    else:
                        st.markdown(doc_text)

                    st.divider()

    # 搜索建议
    if not query:
        st.markdown("---")
        st.markdown("**💡 试试搜索这些话题：**")
        suggestions = [
            "飞天形象的演变历程",
            "宝相花纹样的文化含义",
            "敦煌壁画颜料制作工艺",
            "联珠纹的起源与传播",
            "三兔共耳纹样",
        ]
        cols = st.columns(3)
        for i, s in enumerate(suggestions):
            cols[i % 3].button(s, key=f"suggest_{i}")


# ── 数据看板 ───────────────────────────────────────────────
def render_dashboard():
    """数据看板页面：可视化图表。"""
    st.title("📊 数据看板")
    st.caption("敦煌文化遗产数据可视化")

    tab1, tab2, tab3, tab4 = st.tabs(["🎨 纹样分类", "⏳ 历史演变", "🖌️ 色彩分析", "📈 文献统计"])

    # ── Tab 1: 纹样分类 ──
    with tab1:
        st.markdown("#### 敦煌藻井纹样四大类别")
        
        categories = {
            "几何纹": ["方格纹", "菱形纹", "回纹", "八角星纹", "联珠纹"],
            "植物纹": ["莲花纹", "忍冬纹", "宝相花纹", "卷草纹", "葡萄纹"],
            "动物纹": ["龙纹", "凤纹", "狮子纹", "孔雀纹", "飞马纹"],
            "人物纹": ["飞天", "力士", "菩萨", "供养人", "化生童子"],
        }
        
        fig = go.Figure()
        colors = ["#636EFA", "#00CC96", "#FECB52", "#FF6692"]
        
        for i, (cat, items) in enumerate(categories.items()):
            fig.add_trace(go.Bar(
                name=cat,
                y=items,
                x=[len(items)] * len(items),
                orientation="h",
                marker_color=colors[i],
                text=[f" {cat}" if j == 0 else "" for j in range(len(items))],
                textposition="inside",
            ))
        
        fig.update_layout(
            barmode="group",
            showlegend=True,
            height=500,
            xaxis_title="子类数量",
            title="纹样分类体系",
        )
        st.plotly_chart(fig, use_container_width=True)

        # 详细说明
        for cat, items in categories.items():
            with st.expander(f"📂 {cat}（{len(items)} 个子类）"):
                for item in items:
                    st.markdown(f"- {item}")

    # ── Tab 2: 历史演变 ──
    with tab2:
        st.markdown("#### 敦煌藻井历史演变")
        
        periods = ["北朝", "隋代", "初盛唐", "中晚唐", "五代宋", "元代"]
        # 模拟各时期藻井特征数据（基于文献综述的概括性数据）
        complexity = [2, 3, 5, 5, 4, 3]
        diversity = [1, 2, 4, 5, 4, 2]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=periods, y=complexity, mode="lines+markers+text",
            name="结构复杂度", text=complexity, textposition="top center",
            line=dict(color="#636EFA", width=3), marker=dict(size=12),
        ))
        fig.add_trace(go.Scatter(
            x=periods, y=diversity, mode="lines+markers+text",
            name="纹样多样性", text=diversity, textposition="bottom center",
            line=dict(color="#FF6692", width=3), marker=dict(size=12),
        ))
        
        fig.update_layout(
            yaxis_title="评级（1-5）",
            height=450,
            title="各时期藻井艺术发展水平",
        )
        st.plotly_chart(fig, use_container_width=True)

        # 时期说明卡片
        period_desc = {
            "北朝": "藻井初创期，以简单的几何纹和莲花纹为主，结构朴素。",
            "隋代": "承上启下，忍冬纹大量出现，结构开始复杂化。",
            "初盛唐": "藻井艺术的黄金时代，宝相花纹盛行，结构最为复杂精美。",
            "中晚唐": "纹样组合达到极致，植物纹与动物纹混搭，色彩最为丰富。",
            "五代宋": "风格趋于程式化，继承唐代传统但创新减少。",
            "元代": "受藏传佛教影响，出现新的宗教元素，总体趋于简化。",
        }
        cols = st.columns(3)
        for i, (p, desc) in enumerate(period_desc.items()):
            cols[i % 3].markdown(f"**{p}：**\n{desc}")

    # ── Tab 3: 色彩分析 ──
    with tab3:
        st.markdown("#### 敦煌矿物颜料色彩分析")
        
        pigments = {
            "朱砂红": ("#E74C3C", "辰砂（HgS）", "高温还原反应，颜色鲜艳稳定"),
            "青金石蓝": ("#2980B9", "青金石（Lapis Lazuli）", "产自阿富汗，极为珍贵"),
            "孔雀石绿": ("#27AE60", "孔雀石（Cu₂CO₃(OH)₂）", "含铜矿物，易变色"),
            "雌黄": ("#F39C12", "雌黄（As₂S₃）", "即'信口雌黄'的由来"),
            "铅白": ("#ECF0F1", "铅白（2PbCO₃·Pb(OH)₂）", "覆盖力强，遇硫化物变黑"),
            "赭石": ("#D35400", "赤铁矿（Fe₂O₃）", "最常用的矿物颜料之一"),
            "炭黑": ("#2C3E50", "松烟/墨", "用途最广的黑色颜料"),
            "石黄": ("#F1C40F", "雄黄（As₄S₄）", "橘黄色，常用于袈裟描金"),
        }
        
        cols = st.columns(4)
        for i, (name, (hex_color, mineral, desc)) in enumerate(pigments.items()):
            cols[i % 4].markdown(
                f'<div style="background:{hex_color}; height:80px; border-radius:8px; '
                f'margin-bottom:8px;"></div>'
                f'**{name}**<br>'
                f'<small>矿物：{mineral}<br>{desc}</small>',
                unsafe_allow_html=True,
            )

        st.divider()

        # 各朝代色彩使用占比
        colors_data = {
            "朝代": ["北朝", "隋代", "初盛唐", "中晚唐", "五代宋", "元代"],
            "红色系": [15, 20, 25, 25, 20, 15],
            "蓝色系": [10, 15, 20, 25, 20, 10],
            "绿色系": [10, 15, 20, 20, 15, 15],
            "黄/赭色系": [25, 20, 15, 15, 25, 30],
            "黑色系": [20, 15, 10, 10, 15, 25],
            "白色系": [20, 15, 10, 5, 5, 5],
        }
        
        fig = px.bar(
            colors_data,
            x="朝代",
            y=["红色系", "蓝色系", "绿色系", "黄/赭色系", "黑色系", "白色系"],
            title="各朝代色彩使用分布（百分比）",
            barmode="stack",
            color_discrete_sequence=["#E74C3C", "#2980B9", "#27AE60", "#F39C12", "#2C3E50", "#ECF0F1"],
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Tab 4: 文献统计 ──
    with tab4:
        st.markdown("#### 知识库文献统计")
        
        engine = init_engine(cfg)
        if engine.metadata:
            # 来源分布
            source_counts = {}
            for m in engine.metadata:
                src = m.get("source", "未知")
                source_counts[src] = source_counts.get(src, 0) + 1

            # 按数量排序，取前 15
            sorted_sources = sorted(source_counts.items(), key=lambda x: x[1], reverse=True)[:15]
            names = [s[0].replace(".pdf", "") for s in sorted_sources]
            counts = [s[1] for s in sorted_sources]

            fig = go.Figure(go.Bar(
                x=counts,
                y=names,
                orientation="h",
                marker_color="#667eea",
            ))
            fig.update_layout(
                height=max(400, len(names) * 30),
                xaxis_title="文本块数量",
                title="文献文本块分布（Top 15）",
            )
            st.plotly_chart(fig, use_container_width=True)

            # 总量统计
            total = len(engine.metadata)
            st.metric("总文本块数量", f"{total:,}")
            st.metric("收录文献数量", f"{len(source_counts)} 篇")
        else:
            st.info("知识库尚未构建，请先运行 `python run.py`")


# ── 关于页面 ───────────────────────────────────────────────
def render_about():
    """关于页面：项目背景、团队信息、技术架构。"""
    st.title("ℹ️ 关于项目")

    st.markdown("""
    ## 项目简介

    **敦煌文化遗产智能知识库系统**是一个基于 RAG（检索增强生成）架构的智能问答平台，
    旨在将分散的敦煌学学术文献整合为可交互、可检索的结构化知识库，降低敦煌文化研究门槛。

    本项目收录了 **35 篇** 敦煌藻井相关学术论文，覆盖北朝至元代约 **1000 年** 的艺术演变，
    支持自然语言问答与语义检索，每条回答均标注文献来源，确保学术严谨性。

    ## 技术栈

    | 模块 | 技术 | 说明 |
    |------|------|------|
    | 向量模型 | Sentence Transformers | paraphrase-multilingual-MiniLM-L12-v2，384 维 |
    | 语义检索 | NumPy + Scikit-learn | 余弦相似度 Top-K 召回 |
    | LLM 生成 | OpenAI 兼容接口 | 支持 DeepSeek / 本地模型 |
    | PDF 处理 | pypdf | 批量提取学术文献文本 |
    | 前端界面 | Streamlit | Python 生态的快速 UI 框架 |
    | 可视化 | Plotly | 交互式图表 |
    | 配置管理 | PyYAML | 集中式配置，环境变量注入 |

    ## RAG 架构详解

    ### 为什么用 RAG？

    纯 LLM 存在以下问题：
    - ❌ **幻觉问题**：可能编造不存在的文化细节
    - ❌ **知识过时**：训练数据截止后无法更新
    - ❌ **无法溯源**：无法验证回答的准确性

    RAG 通过"先检索、后生成"的方式解决这些问题：
    - ✅ **有据可查**：每条回答都基于真实文献
    - ✅ **可追溯**：标注来源文献，支持学术引用
    - ✅ **可更新**：添加新文献即可扩展知识库

    ### 数据处理流程

    ```
    PDF 文献 → pypdf 文本提取 → 滑动窗口分块（500字/块，50字重叠）
                                       ↓
    用户提问 → 向量化 → 余弦相似度检索 Top-K → 上下文注入 Prompt → LLM 生成回答
    ```

    ## 作者

    **陈锦彤** — 独立完成系统设计、开发与实现

    ## 许可

    本项目仅供学术研究与学习交流使用。文献版权归原作者所有。
    """)


# ── 主入口 ─────────────────────────────────────────────────
def main():
    """Streamlit 应用主入口。"""
    render_sidebar()

    # 页面路由
    page_map = {
        "🏠 首页": render_home,
        "💬 智能问答": render_qa,
        "🔍 语义检索": render_search,
        "📊 数据看板": render_dashboard,
        "ℹ️ 关于": render_about,
    }

    # 从 session_state 恢复页面（示例问题跳转）
    current_page = st.session_state.get("page", None)

    with st.sidebar:
        selected = st.radio(
            "导航",
            list(page_map.keys()),
            index=list(page_map.keys()).index(current_page) if current_page and current_page in page_map else 0,
        )

    # 清除跳转状态
    if "page" in st.session_state:
        del st.session_state["page"]

    page_map[selected]()


if __name__ == "__main__":
    main()
