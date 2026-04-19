# -*- coding: utf-8 -*-
"""
RAG 引擎 — 检索增强生成核心模块。

整合向量检索与 LLM 生成，提供"带来源标注"的智能问答能力。
支持 OpenAI 兼容接口（DeepSeek / 本地模型等）。
"""

import os
from typing import List, Optional

from core.vectorizer import VectorSearchEngine
from utils.logger import setup_logger

logger = setup_logger("rag_engine")


class RAGEngine:
    """检索增强生成引擎。

    串联 向量检索 → 上下文组装 → LLM 生成 的完整流程。

    Args:
        vector_engine: 已构建索引的向量检索引擎。
        system_prompt: 系统提示词模板，支持 {{context}} 和 {{question}} 占位符。
        llm_provider: LLM 提供商标识（deepseek / openai / local）。
        llm_base_url: API 基础 URL（留空则使用默认端点）。
        llm_model: 模型名称。
        llm_max_tokens: 最大生成 token 数。
        llm_temperature: 生成温度（0~1，越低越确定）。
    """

    def __init__(
        self,
        vector_engine: VectorSearchEngine,
        system_prompt: str,
        llm_provider: str = "deepseek",
        llm_base_url: str = "",
        llm_model: str = "deepseek-chat",
        llm_max_tokens: int = 2048,
        llm_temperature: float = 0.3,
    ):
        self.vector_engine = vector_engine
        self.system_prompt = system_prompt
        self.llm_provider = llm_provider
        # 环境变量优先覆盖配置文件
        self.llm_base_url = os.environ.get("DUNHUANG_API_BASE", llm_base_url)
        self.llm_model = os.environ.get("DUNHUANG_MODEL", llm_model)
        self.llm_max_tokens = llm_max_tokens
        self.llm_temperature = llm_temperature

    def retrieve(self, question: str, top_k: int = 5) -> List[dict]:
        """仅执行检索步骤，返回相关文档片段。

        Args:
            question: 用户问题。
            top_k: 检索数量。

        Returns:
            检索结果列表（含 document、metadata、score）。
        """
        return self.vector_engine.search(question, top_k=top_k)

    def build_context(self, question: str, top_k: int = 5) -> str:
        """检索并组装 RAG 上下文文本。

        将检索到的文档片段格式化为 LLM 可理解的上下文。

        Args:
            question: 用户问题。
            top_k: 检索数量。

        Returns:
            格式化的上下文字符串。
        """
        results = self.retrieve(question, top_k)

        if not results:
            return ""

        context_parts = []
        for i, r in enumerate(results, 1):
            source = r["metadata"].get("source", "未知来源")
            chunk_id = r["metadata"].get("chunk_id", "?")
            context_parts.append(
                f"【参考文献 {i}】{source}（片段 {chunk_id}）\n{r['document']}"
            )

        return "\n\n".join(context_parts)

    def answer(
        self, question: str, top_k: int = 5, use_llm: bool = True
    ) -> dict:
        """完整的 RAG 问答流程：检索 → 组装 → 生成。

        Args:
            question: 用户问题。
            top_k: 检索数量。
            use_llm: 是否调用 LLM 生成回答。False 时仅返回检索结果。

        Returns:
            dict 包含:
                - question: 原始问题
                - answer: 生成的回答（或检索摘要）
                - sources: 引用的来源文献列表
                - search_results: 原始检索结果
        """
        results = self.retrieve(question, top_k)
        sources = [r["metadata"].get("source", "未知来源") for r in results]
        unique_sources = list(dict.fromkeys(sources))  # 去重保序

        if not results:
            return {
                "question": question,
                "answer": "抱歉，知识库中未找到与您的问题相关的内容。您可以尝试换一种表述，或参考其他研究方向。",
                "sources": [],
                "search_results": [],
            }

        context = self.build_context(question, top_k)

        if not use_llm:
            # 无 LLM 时，返回检索摘要
            answer = self._summarize_without_llm(results)
        else:
            answer = self._call_llm(question, context)

        return {
            "question": question,
            "answer": answer,
            "sources": unique_sources,
            "search_results": results,
        }

    def _call_llm(self, question: str, context: str) -> str:
        """调用 LLM 生成回答。

        尝试使用 OpenAI 兼容接口。若未配置 API Key，回退到基于检索的摘要。

        Args:
            question: 用户问题。
            context: RAG 检索上下文。

        Returns:
            LLM 生成的回答，或回退摘要。
        """
        api_key = os.environ.get("DUNHUANG_API_KEY") or os.environ.get("OPENAI_API_KEY", "")

        if not api_key:
            logger.warning("未检测到 API Key（DUNHUANG_API_KEY / OPENAI_API_KEY），回退到检索摘要模式")
            return self._summarize_with_context(question, context)

        try:
            from openai import OpenAI

            client = OpenAI(
                api_key=api_key,
                base_url=self.llm_base_url or None,
            )

            prompt = self.system_prompt.replace("{{context}}", context).replace("{{question}}", question)

            response = client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": question},
                ],
                max_tokens=self.llm_max_tokens,
                temperature=self.llm_temperature,
            )

            return response.choices[0].message.content or ""

        except ImportError:
            logger.warning("openai 库未安装，回退到检索摘要模式。安装：pip install openai")
            return self._summarize_with_context(question, context)
        except Exception as e:
            logger.error("LLM 调用失败：%s，回退到检索摘要模式", e)
            return self._summarize_with_context(question, context)

    def _summarize_without_llm(self, results: List[dict]) -> str:
        """无 LLM 时的简单摘要：拼接检索结果。

        输出 Markdown 格式（Streamlit 的 st.markdown 原生支持渲染）。
        Flask 前端 demo.html 会通过正则将 Markdown 转为 HTML。

        Args:
            results: 检索结果列表。

        Returns:
            格式化的检索摘要文本。
        """
        lines = ["以下是与您的问题最相关的文献片段：\n"]
        for i, r in enumerate(results, 1):
            source = r["metadata"].get("source", "未知来源")
            score = r["score"]
            lines.append(f"**{i}. 来源：{source}** （相关度：{score:.1%}）\n")
            lines.append(f"> {r['document'][:300]}{'...' if len(r['document']) > 300 else ''}\n")

        lines.append("\n> 💡 *提示：配置 API Key 后可获得 AI 生成的完整回答。*")
        return "\n".join(lines)

    def _summarize_with_context(self, question: str, context: str) -> str:
        """有上下文但无 LLM 时的增强摘要。

        输出 Markdown 格式。Flask 前端会通过正则将 Markdown 转为 HTML。

        Args:
            question: 用户问题。
            context: RAG 上下文。

        Returns:
            包含上下文信息的摘要文本。
        """
        # 从上下文中提取文献编号，构建来源索引
        import re
        ref_pattern = re.compile(r'【参考文献 (\d+)】([^\n（（]+)')
        refs_found = ref_pattern.findall(context)
        ref_index = {}
        for num, name in refs_found:
            ref_index[num] = name.strip().rstrip('_').strip()

        # 构建来源说明
        source_notes = ""
        if ref_index:
            source_lines = []
            for num in sorted(ref_index.keys(), key=int):
                source_lines.append(f"  【{num}】{ref_index[num]}")
            source_notes = "\n\n**📎 参考文献说明：**\n" + "\n".join(source_lines)

        lines = [
            f"**问题：** {question}\n",
            "**以下内容来自知识库检索结果：**\n",
            context[:2000],
            source_notes,
            "\n\n> 💡 *提示：安装 openai 库并配置 DUNHUANG_API_KEY 环境变量后，"
            "系统将自动切换为 AI 生成模式。当前为检索摘要模式。*",
        ]
        return "\n".join(lines)
