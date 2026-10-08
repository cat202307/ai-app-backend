from typing import Any
import re

def build_citations(
    reranked_results: list[dict[str, Any]],
    file_id: int,
    filename: str,
) -> list[dict[str, Any]]:
    """
    根据 Rerank 最终结果创建 Citation（引用来源）。

    参数：

        reranked_results
            Rerank 后的最终 Chunk。

        file_id
            文件数据库 ID。

        filename
            文件名称。

    返回：

        [
            {
                "citation_id": 1,
                "file_id": 1,
                "filename": "test01.txt",
                "chunk_id": "file_1_chunk_0",
                "content": "...",
                "rerank_score": 95
            }
        ]
    """

    citations = []

    for citation_id, item in enumerate(
        reranked_results,
        start=1,
    ):

        citations.append({
            # citation_id
            # = 引用编号
            "citation_id": citation_id,

            # 全局检索时以 chunk 自带归属为准，单文件回退入参
            "file_id": item.get(
                "file_id",
                file_id,
            ),

            "filename": item.get(
                "filename",
                filename,
            ),

            "chunk_id": item.get(
                "chunk_id"
            ),

            "content": item.get(
                "content",
                ""
            ),

            "rerank_score": item.get(
                "rerank_score"
            ),
        })

    return citations


def build_citation_context(
    citations: list[dict[str, Any]],
) -> str:
    """
    将 Citation 转换成提供给 LLM 的 Context。

    Context
        = 上下文

    示例：

        [1]
        文件：test01.txt
        Chunk：file_1_chunk_0
        内容：xxxx

        [2]
        ...
    """

    context_parts = []

    for citation in citations:

        citation_id = citation[
            "citation_id"
        ]

        filename = citation.get(
            "filename",
            ""
        )

        chunk_id = citation.get(
            "chunk_id",
            ""
        )

        content = citation.get(
            "content",
            ""
        )

        context_parts.append(
            (
                f"[{citation_id}]\n"
                f"文件：{filename}\n"
                f"Chunk：{chunk_id}\n"
                f"内容：{content}"
            )
        )

    return "\n\n".join(
        context_parts
    )

def extract_citation_ids(
    answer: str,
) -> list[int]:
    """
    从最终回答中提取实际使用的 Citation ID。

    Citation
        = 引用

    示例：

        FastAPI 是 Web 框架 [1]。
        RAG 是检索增强生成 [2]。

    返回：

        [1, 2]
    """

    if not answer:
        return []

    matches = re.findall(
        r"\[(\d+)\]",
        answer,
    )

    citation_ids = []

    for item in matches:

        citation_id = int(item)

        if citation_id not in citation_ids:
            citation_ids.append(
                citation_id
            )

    return citation_ids

def filter_used_citations(
    answer: str,
    citations: list[dict],
) -> list[dict]:
    """
    Citation Pruning（引用裁剪）。

    只保留最终 Answer 中真正使用到的 Citation。

    例如：

        answer：
            AI Agent 是……[1]

        citations：
            [1] chunk_2
            [2] chunk_1

        返回：
            [1] chunk_2
    """

    used_citation_ids = set(
        extract_citation_ids(
            answer
        )
    )

    if not used_citation_ids:
        # 模型未在回答中插入 [n] 引用标记（常见于“信息不足/
        # 否定”类回答）。此时 citations 均已通过 rerank 相关性
        # 阈值，是回答结论的依据，回退保留全部，避免支撑来源
        # 因缺少标记而被全部裁剪。
        return citations

    return [
        citation
        for citation in citations
        if citation.get(
            "citation_id"
        )
        in used_citation_ids
    ]