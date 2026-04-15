# -*- coding: utf-8 -*-
"""
Flask API 后端 — 为前端提供知识库检索与 RAG 问答接口。

提供两个核心 API：
- GET  /api/status   → 知识库状态（文献数、文本块数）
- POST /api/search   → 语义检索（返回相关文档片段）
- POST /api/ask      → RAG 问答（检索 + 可选 LLM 生成）

启动方式：
    python api/server.py
    或 python run.py --mode api
"""

import os
import sys
import logging

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# 确保项目根目录在 Python 路径中
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import load_config
from core.pdf_parser import PDFParser
from core.chunker import TextChunker
from core.vectorizer import VectorSearchEngine
from core.rag_engine import RAGEngine

# ── 日志 ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("api.server")

# ── Flask 应用 ──
app = Flask(__name__, static_folder=None)
CORS(app)  # 允许跨域（前端可能在不同端口）

# ── 全局引擎实例（模块级单例）──
_vector_engine: VectorSearchEngine = None  # type: ignore
_rag_engine: RAGEngine = None  # type: ignore
_config: dict = {}


def get_vector_engine() -> VectorSearchEngine:
    """获取或初始化向量检索引擎（惰性单例）。"""
    global _vector_engine
    if _vector_engine is None:
        _vector_engine = _build_vector_engine()
    return _vector_engine


def get_rag_engine() -> RAGEngine:
    """获取或初始化 RAG 引擎（惰性单例）。"""
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = _build_rag_engine()
    return _rag_engine


def _build_vector_engine() -> VectorSearchEngine:
    """构建并返回向量检索引擎（含索引构建/加载逻辑）。"""
    cfg = _config
    # config/__init__.py 已将相对路径转为绝对路径
    pdf_dir = cfg["paths"]["pdf_dir"]
    cache_dir = cfg["paths"]["cache_dir"]
    index_file = cfg["paths"]["index_file"]
    docs_file = cfg["paths"]["docs_file"]
    meta_file = cfg["paths"]["metadata_file"]

    vec_cfg = cfg["vectorizer"]
    engine = VectorSearchEngine(
        model_name=vec_cfg["model_name"],
        embedding_dim=vec_cfg["embedding_dim"],
        default_top_k=vec_cfg["default_top_k"],
        similarity_threshold=vec_cfg["similarity_threshold"],
    )

    # 尝试加载缓存索引
    if engine.load_index(index_file, docs_file, meta_file):
        engine.load_status = "loaded"
        logger.info("向量索引已从缓存加载")
    else:
        # 缓存不存在，从 PDF 构建
        logger.info("缓存不存在，开始从 PDF 构建知识库索引...")

        parser = PDFParser(
            pdf_dir=pdf_dir,
            max_chars_per_page=cfg["pdf_parser"]["max_chars_per_page"],
        )
        chunker = TextChunker(
            chunk_size=cfg["chunker"]["chunk_size"],
            overlap=cfg["chunker"]["overlap"],
            min_length=cfg["chunker"]["min_chunk_length"],
        )

        documents = parser.parse_all()
        if documents:
            chunks, metadata = chunker.chunk_documents(documents)
            if chunks:
                engine.build_index(chunks, metadata)
                os.makedirs(cache_dir, exist_ok=True)
                engine.save_index(index_file, docs_file, meta_file)
                engine.load_status = "built"
                logger.info("知识库索引构建完成：%d 个文本块", len(chunks))
            else:
                engine.load_status = "empty"
                logger.warning("分块后无有效文本块")
        else:
            engine.load_status = "empty"
            logger.warning("未找到有效的 PDF 文件")

    return engine


def _build_rag_engine() -> RAGEngine:
    """构建 RAG 引擎。"""
    cfg = _config["rag"]
    return RAGEngine(
        vector_engine=get_vector_engine(),
        system_prompt=cfg["system_prompt"],
        llm_provider=cfg["llm_provider"],
        llm_base_url=cfg.get("llm_base_url", ""),
        llm_model=cfg["llm_model"],
        llm_max_tokens=cfg["llm_max_tokens"],
        llm_temperature=cfg["llm_temperature"],
    )


# ── API 路由 ──────────────────────────────────────────────────


@app.route("/api/status", methods=["GET"])
def api_status():
    """知识库状态接口。

    Returns:
        JSON: 文献数量、文本块数量、模型名称、索引状态。
    """
    engine = get_vector_engine()
    pdf_count = len(set(m["source"] for m in engine.metadata)) if engine.metadata else 0
    chunk_count = len(engine.documents) if engine.documents else 0

    return jsonify({
        "status": "ready",
        "pdf_count": pdf_count,
        "chunk_count": chunk_count,
        "model_name": engine.model_name,
        "index_status": getattr(engine, "load_status", "unknown"),
    })


@app.route("/api/search", methods=["POST"])
def api_search():
    """语义检索接口。

    Request Body:
        query (str): 检索关键词或自然语言问题。
        top_k (int, optional): 返回结果数量，默认 5。

    Returns:
        JSON: results 列表，每项含 document、source、score。
    """
    data = request.get_json(silent=True) or {}
    query = data.get("query", "").strip()
    top_k = min(data.get("top_k", 5), 20)  # 上限 20

    if not query:
        return jsonify({"error": "请提供检索关键词（query）"}), 400

    engine = get_vector_engine()
    results = engine.search(query, top_k=top_k)

    return jsonify({
        "query": query,
        "results": [
            {
                "document": r["document"],
                "source": r["metadata"].get("source", "未知来源"),
                "chunk_id": r["metadata"].get("chunk_id", 0),
                "score": round(r["score"], 4),
            }
            for r in results
        ],
    })


@app.route("/api/ask", methods=["POST"])
def api_ask():
    """RAG 问答接口。

    Request Body:
        question (str): 用户问题。
        top_k (int, optional): 检索数量，默认 5。
        use_llm (bool, optional): 是否调用 LLM 生成，默认 true。

    Returns:
        JSON: question、answer、sources 列表。
    """
    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()
    top_k = min(data.get("top_k", 5), 20)
    use_llm = data.get("use_llm", True)

    if not question:
        return jsonify({"error": "请提供问题（question）"}), 400

    rag = get_rag_engine()
    result = rag.answer(question, top_k=top_k, use_llm=use_llm)

    return jsonify({
        "question": result["question"],
        "answer": result["answer"],
        "sources": result["sources"],
    })


# ── 前端静态文件服务 ─────────────────────────────────────────


@app.route("/")
def serve_index():
    """提供主页面（demo.html）。"""
    template_dir = os.path.join(PROJECT_ROOT, "ui", "templates")
    return send_from_directory(template_dir, "demo.html")


@app.route("/<path:filename>")
def serve_static(filename):
    """提供静态文件（CSS、JS、图片等）。"""
    template_dir = os.path.join(PROJECT_ROOT, "ui", "templates")
    return send_from_directory(template_dir, filename)


# ── 启动入口 ──────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="敦煌知识库 API 服务器")
    parser.add_argument("--port", type=int, default=5000, help="监听端口（默认 5000）")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="监听地址")
    args = parser.parse_args()

    # 加载配置
    _config = load_config()
    logger.info("配置加载完成")

    print(f"\n{'='*50}")
    print(f"  敦煌文化遗产智能知识库 · API 服务器")
    print(f"  访问地址：http://localhost:{args.port}")
    print(f"{'='*50}\n")

    app.run(host=args.host, port=args.port, debug=False)
