# -*- coding: utf-8 -*-
"""
向量化检索引擎 — 文档嵌入、索引构建与语义检索。

使用 Sentence Transformers 生成文本向量，基于余弦相似度进行 Top-K 召回。
支持向量索引持久化，避免重复计算。
"""

import json
import os
from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from utils.logger import setup_logger

logger = setup_logger("vectorizer")


class VectorSearchEngine:
    """语义向量检索引擎。

    Args:
        model_name: HuggingFace 模型名称或本地路径。
        embedding_dim: 向量维度（预留给模型校验）。
        default_top_k: 默认检索返回数量。
        similarity_threshold: 最低相似度阈值，低于此值的结果不返回。
    """

    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        embedding_dim: int = 384,
        default_top_k: int = 5,
        similarity_threshold: float = 0.2,
    ):
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.default_top_k = default_top_k
        self.similarity_threshold = similarity_threshold

        self.model: SentenceTransformer = None  # type: ignore
        self.embeddings: np.ndarray = None  # type: ignore
        self.documents: List[str] = []
        self.metadata: List[dict] = []

    def _load_model(self):
        """惰性加载向量化模型（首次检索或构建索引时调用）。"""
        if self.model is None:
            logger.info("正在加载向量化模型：%s ...", self.model_name)
            self.model = SentenceTransformer(self.model_name)
            logger.info("模型加载完成")

    def build_index(self, documents: List[str], metadata: List[dict]) -> None:
        """对文档列表生成向量索引。

        Args:
            documents: 文本块列表。
            metadata: 与文档一一对应的元数据列表。
        """
        self._load_model()
        self.documents = documents
        self.metadata = metadata

        logger.info("正在生成向量嵌入（共 %d 个文本块）...", len(documents))
        self.embeddings = self.model.encode(documents, show_progress_bar=True, normalize_embeddings=True)
        logger.info("向量索引构建完成，维度：%s", self.embeddings.shape)

    def search(self, query: str, top_k: int = None) -> List[dict]:
        """对查询文本进行语义检索。

        Args:
            query: 用户查询（自然语言）。
            top_k: 返回的最相似结果数，默认使用实例配置。

        Returns:
            检索结果列表，每个元素包含 document、metadata、score 字段。
        """
        if self.embeddings is None:
            logger.error("向量索引未构建，请先调用 build_index()")
            return []

        top_k = top_k or self.default_top_k
        self._load_model()

        query_vec = self.model.encode([query], normalize_embeddings=True)
        scores = cosine_similarity(query_vec, self.embeddings)[0]
        top_indices = np.argsort(scores)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score >= self.similarity_threshold:
                results.append({
                    "document": self.documents[idx],
                    "metadata": self.metadata[idx],
                    "score": score,
                })

        return results

    def save_index(self, index_path: str, docs_path: str, meta_path: str) -> None:
        """持久化向量索引与文档到磁盘。

        Args:
            index_path: numpy 索引文件路径（.npy）。
            docs_path: 文档 JSON 文件路径。
            meta_path: 元数据 JSON 文件路径。
        """
        if self.embeddings is None:
            logger.error("没有可保存的索引")
            return

        os.makedirs(os.path.dirname(index_path), exist_ok=True)

        np.save(index_path, self.embeddings)
        with open(docs_path, "w", encoding="utf-8") as f:
            json.dump(self.documents, f, ensure_ascii=False, indent=2)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

        logger.info("索引已保存 → %s", os.path.dirname(index_path))

    def load_index(self, index_path: str, docs_path: str, meta_path: str) -> bool:
        """从磁盘加载已有的向量索引。

        Args:
            index_path: numpy 索引文件路径。
            docs_path: 文档 JSON 文件路径。
            meta_path: 元数据 JSON 文件路径。

        Returns:
            bool: 是否加载成功。
        """
        if not all(os.path.exists(p) for p in (index_path, docs_path, meta_path)):
            logger.info("索引文件不完整，将重新构建")
            return False

        self.embeddings = np.load(index_path)
        with open(docs_path, "r", encoding="utf-8") as f:
            self.documents = json.load(f)
        with open(meta_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self._load_model()
        logger.info("索引已加载：%d 个文本块", len(self.documents))
        return True
