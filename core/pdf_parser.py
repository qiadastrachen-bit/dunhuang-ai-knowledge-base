# -*- coding: utf-8 -*-
"""
PDF 解析模块 — 从 PDF 文献中提取文本内容。

支持批量处理指定目录下的所有 PDF 文件，返回结构化的文本与元数据。
"""

import os
from typing import List, Tuple

import pypdf

from utils.logger import setup_logger

logger = setup_logger("pdf_parser")


class PDFParser:
    """PDF 文献文本提取器。

    Args:
        pdf_dir: PDF 文件所在目录路径。
        max_chars_per_page: 单页最大提取字符数，防止异常长页导致内存问题。
    """

    def __init__(self, pdf_dir: str, max_chars_per_page: int = 10000):
        self.pdf_dir = pdf_dir
        self.max_chars_per_page = max_chars_per_page

    def list_pdfs(self) -> List[str]:
        """列出目录下所有 PDF 文件名。

        Returns:
            PDF 文件名列表（仅文件名，不含路径）。
        """
        if not os.path.isdir(self.pdf_dir):
            logger.warning("PDF 目录不存在：%s", self.pdf_dir)
            return []

        return sorted(
            f for f in os.listdir(self.pdf_dir) if f.lower().endswith(".pdf")
        )

    def extract_text(self, pdf_path: str) -> str:
        """从单个 PDF 文件提取全部文本。

        Args:
            pdf_path: PDF 文件的完整路径。

        Returns:
            提取到的文本字符串。若读取失败返回空字符串。
        """
        try:
            with open(pdf_path, "rb") as f:
                reader = pypdf.PdfReader(f)
                pages = []
                for page in reader.pages:
                    text = page.extract_text() or ""
                    if len(text) > self.max_chars_per_page:
                        text = text[: self.max_chars_per_page]
                    pages.append(text)
                return "\n".join(pages)
        except Exception as e:
            logger.error("解析 PDF 失败 [%s]：%s", pdf_path, e)
            return ""

    def parse_all(self) -> List[Tuple[str, str]]:
        """批量解析目录下所有 PDF 文件。

        Returns:
            元组列表，每个元组为 (文件名, 提取的文本)。
        """
        filenames = self.list_pdfs()
        if not filenames:
            logger.warning("未找到任何 PDF 文件")
            return []

        logger.info("找到 %d 个 PDF 文件，开始解析...", len(filenames))
        results = []

        for i, fname in enumerate(filenames, 1):
            fpath = os.path.join(self.pdf_dir, fname)
            text = self.extract_text(fpath)
            if text.strip():
                results.append((fname, text))
                logger.info("  [%d/%d] %s — %d 字符", i, len(filenames), fname, len(text))
            else:
                logger.warning("  [%d/%d] %s — 未能提取文本，已跳过", i, len(filenames), fname)

        logger.info("解析完成：成功 %d / 总计 %d", len(results), len(filenames))
        return results
