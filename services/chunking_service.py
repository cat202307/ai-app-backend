"""
结构感知的父子块切分服务（Parent-Child Chunking）。

背景：
    固定字符硬切（800 字一刀）会把一个具体答案（如优化器
    AdamW 的描述）埋在大块边缘，周围是大量无关内容，导致：
      1. Embedding 向量被无关内容主导，召回不到；
      2. Cross-Encoder 对「草堆寻针」整体匹配被稀释，
         即使召回也只给基线分（~51）。

方案（Small-to-Big / Parent-Child）：
    - 子块（Child，~200 字）：按段落 / 句子边界切，主题聚焦，
      负责精准的 Embedding + Rerank；
    - 父块（Parent，~4 个子块）：子块命中后聚合同组父块，
      给 LLM 提供完整上下文生成答案。

索引只存子块（metadata 带 parent_id）；父块不单独存储，
命中后按 parent_id 实时聚合同组子块得到。
"""

from __future__ import annotations

import re
from typing import Any

# 编号点开头：（1） / (1) / ① / 1. / 1、（后不跟数字，避免误判 1.5 章节号）
_ENUMERATION_RE = re.compile(
    r"^\s*(?:[（(]\s*\d+\s*[）)]|[①-⑳]|\d+\s*[.、](?!\d))"
)

# 超长段落的切分边界：句末标点或编号点
_BOUNDARY_RE = re.compile(
    r"[。！？；]|\s*[（(]\s*\d+\s*[）)]|[①-⑳]|(?<!\d)\d+\s*[.、](?!\d)"
)


def is_enumeration_start(text: str) -> bool:
    """段落是否以编号点（1）（2）开头。"""
    return bool(_ENUMERATION_RE.match(text or ""))


def split_long_paragraph(text: str, max_len: int) -> list[str]:
    """
    把超长段落按句子 / 编号边界切成不超过 max_len 的片段。
    无法在边界内切分时硬切，保证长度上限。
    """

    if len(text) <= max_len:
        return [text.strip()]

    # 找所有边界位置
    cuts = [0]
    for match in _BOUNDARY_RE.finditer(text):
        cuts.append(match.end())

    segments = []
    for i in range(len(cuts) - 1):
        segments.append(text[cuts[i]:cuts[i + 1]])
    segments.append(text[cuts[-1]:])

    pieces: list[str] = []
    current = ""

    for segment in segments:
        if len(current) + len(segment) <= max_len:
            current += segment
            continue

        if current.strip():
            pieces.append(current.strip())

        # 单个片段仍超长：硬切
        while len(segment) > max_len:
            pieces.append(segment[:max_len].strip())
            segment = segment[max_len:]
        current = segment

    if current.strip():
        pieces.append(current.strip())

    return pieces


def build_parent_child_chunks(
    text: str,
    file_id: int,
    child_max: int = 200,
    children_per_parent: int = 4,
) -> dict[str, Any]:
    """
    把完整文档切分为父子块。

    返回：
        {
            "children": [
                {
                    "child_id": "file_1_child_52",
                    "parent_id": "file_1_parent_13",
                    "content": "...",
                    "child_index": 52,
                    "parent_index": 13,
                }, ...
            ],
            "parents": [
                {
                    "parent_id": "file_1_parent_13",
                    "content": "child...\nchild...",
                    "parent_index": 13,
                }, ...
            ],
        }
    """

    paragraphs = [
        line.strip()
        for line in (text or "").split("\n")
        if line.strip()
    ]

    # 1. 拆成最小语义单元 (text, is_enumeration)
    units: list[tuple[str, bool]] = []
    for paragraph in paragraphs:
        is_enum = is_enumeration_start(paragraph)
        if len(paragraph) <= child_max:
            units.append((paragraph, is_enum))
        else:
            for piece in split_long_paragraph(paragraph, child_max):
                units.append((piece, is_enum or is_enumeration_start(piece)))

    # 2. 贪心聚合成子块；编号点保持独立，不与其他编号点合并
    child_texts: list[str] = []
    current = ""
    current_is_enum = False

    def flush() -> None:
        nonlocal current
        if current.strip():
            child_texts.append(current.strip())
        current = ""

    for unit_text, is_enum in units:
        if not current:
            current = unit_text
            current_is_enum = is_enum
            continue

        # 编号点不与编号点合并；新编号点也不追加到非编号块
        if is_enum:
            flush()
            current = unit_text
            current_is_enum = True
            continue

        if len(current) + 1 + len(unit_text) <= child_max:
            current = current + "\n" + unit_text
        else:
            flush()
            current = unit_text
            current_is_enum = is_enum

    flush()

    # 3. 每 children_per_parent 个子块聚合成父块
    children: list[dict[str, Any]] = []
    parents: list[dict[str, Any]] = []

    for child_index, child_text in enumerate(child_texts):
        parent_index = child_index // children_per_parent
        child_id = f"file_{file_id}_child_{child_index}"
        parent_id = f"file_{file_id}_parent_{parent_index}"
        children.append(
            {
                "child_id": child_id,
                "parent_id": parent_id,
                "content": child_text,
                "child_index": child_index,
                "parent_index": parent_index,
            }
        )

    for parent_index in range(
        (len(child_texts) + children_per_parent - 1)
        // children_per_parent
    ):
        group = child_texts[
            parent_index
            * children_per_parent : (parent_index + 1)
            * children_per_parent
        ]
        parents.append(
            {
                "parent_id": f"file_{file_id}_parent_{parent_index}",
                "content": "\n".join(group),
                "parent_index": parent_index,
            }
        )

    return {"children": children, "parents": parents}
