# -*- coding: utf-8 -*-
"""
文本分块模块 — 将长文档拆分为适合向量化的语义片段。

采用滑动窗口策略，支持中文文本按字符数切分。
"""

from typing import List, Tuple

from utils.logger import setup_logger

logger = setup_logger("chunker")


class TextChunker:
    """滑动窗口文本分块器。

    中文文本按字符数切分，重叠区域保证跨块语义连贯。

    Args:
        chunk_size: 每个文本块的目标字符数。
        overlap: 相邻块之间的重叠字符数。
        min_length: 低于此长度的块会被丢弃。
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50, min_length: int = 30):
        if overlap >= chunk_size:
            raise ValueError(f"overlap ({overlap}) 必须 < chunk_size ({chunk_size})")

        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_length = min_length

    def chunk_text(self, text: str) -> List[str]:
        """将单篇文本切分为多个块。

        Args:
            text: 输入文本。

        Returns:
            文本块列表。
        """
        if not text or not text.strip():
            return []

        text = text.strip()
        step = self.chunk_size - self.overlap
        chunks = []

        for start in range(0, len(text), step):
            end = start + self.chunk_size
            chunk = text[start:end].strip()
            if len(chunk) >= self.min_length:
                chunks.append(chunk)

        return chunks

    def chunk_documents(self, documents: List[Tuple[str, str]]) -> Tuple[List[str], List[dict]]:
        """批量分块，同时保留来源元数据。

        Args:
            documents: (文件名, 文本) 元组列表。

        Returns:
            (文本块列表, 元数据列表)。元数据包含 source 和 chunk_id。
        """
        all_chunks = []
        all_metadata = []

        for filename, text in documents:
            chunks = self.chunk_text(text)
            for idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadata.append({
                    "source": filename,
                    "chunk_id": idx,
                    "total_chunks": len(chunks),
                })

        logger.info(
            "分块完成：%d 篇文档 → %d 个文本块",
            len(documents), len(all_chunks),
        )
        return all_chunks, all_metadata
