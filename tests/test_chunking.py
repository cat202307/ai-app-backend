"""
父子块切分服务测试（services.chunking_service）。

纯文本逻辑，不加载模型：
  - 编号点识别（含章节号 1.5 排除）
  - 超长段落切分
  - 父子块结构、长度上限、分组、内容守恒
"""

import re

import pytest

from services.chunking_service import (
    build_parent_child_chunks,
    is_enumeration_start,
    split_long_paragraph,
)


class TestEnumerationStart:
    """编号点开头识别。"""

    @pytest.mark.parametrize(
        "text",
        [
            "（1）优化器采用 AdamW",
            "(2) 数据处理流程",
            "① 初始化参数",
            "1. 设置学习率",
            "1、加载数据",
            "  （3）带前导空格",
        ],
    )
    def test_matches_enumeration(self, text):
        assert is_enumeration_start(text) is True

    @pytest.mark.parametrize(
        "text",
        [
            "1.5 算法总体流程",
            "普通段落没有编号",
            "版本 2.0 说明",
            "",
        ],
    )
    def test_rejects_non_enumeration(self, text):
        assert is_enumeration_start(text) is False


class TestSplitLongParagraph:
    """超长段落切分。"""

    def test_short_text_unchanged(self):
        text = "这是一个短段落。"
        assert split_long_paragraph(text, 200) == [text]

    def test_no_punctuation_hard_cut(self):
        text = "字" * 500
        pieces = split_long_paragraph(text, 100)
        assert len(pieces) == 5
        for piece in pieces:
            assert len(piece) <= 100

    def test_sentence_boundary_cut(self):
        text = "第一句内容。" * 30  # 180 字
        pieces = split_long_paragraph(text, 50)
        for piece in pieces:
            assert len(piece) <= 50
        # 内容守恒（去空白）
        joined = re.sub(r"\s", "", "".join(pieces))
        assert joined == re.sub(r"\s", "", text)

    def test_content_preserved(self):
        text = (
            "模型采用 AdamW 优化器。学习率设为 1e-3。"
            "损失函数使用 MSE。"
        ) * 10
        pieces = split_long_paragraph(text, 80)
        joined = re.sub(r"\s", "", "".join(pieces))
        assert joined == re.sub(r"\s", "", text)


class TestBuildParentChildChunks:
    """父子块整体结构。"""

    def test_empty_text(self):
        result = build_parent_child_chunks("", file_id=1)
        assert result["children"] == []
        assert result["parents"] == []

    def test_short_single_chunk(self):
        text = "这是一段很短的文档内容。"
        result = build_parent_child_chunks(text, file_id=7)
        assert len(result["children"]) == 1
        assert len(result["parents"]) == 1

        child = result["children"][0]
        assert child["child_id"] == "file_7_child_0"
        assert child["parent_id"] == "file_7_parent_0"
        assert child["child_index"] == 0
        assert child["parent_index"] == 0

    def test_child_indices_contiguous(self):
        text = "\n".join(
            "这是第 %d 段的内容，用于填充。" % i
            for i in range(20)
        )
        result = build_parent_child_chunks(text, file_id=1)
        indices = [c["child_index"] for c in result["children"]]
        assert indices == list(range(len(indices)))

    def test_children_within_size_limit(self):
        text = "\n".join(
            "段落 %d：%s" % (i, "内容" * 30)
            for i in range(15)
        )
        result = build_parent_child_chunks(
            text, file_id=1, child_max=200
        )
        for child in result["children"]:
            assert len(child["content"]) <= 200
            assert child["content"].strip() != ""

    def test_parent_grouping_size(self):
        text = "\n".join(
            "这是编号 %d 的独立段落文本。" % i
            for i in range(18)
        )
        result = build_parent_child_chunks(
            text,
            file_id=1,
            children_per_parent=4,
        )
        parent_ids = [
            c["parent_id"] for c in result["children"]
        ]
        # 每个父块最多 4 个子块
        for parent in result["parents"]:
            count = parent_ids.count(parent["parent_id"])
            assert count <= 4

    def test_parent_id_references_exist(self):
        text = "\n".join(
            "段落 %d 的填充内容文字。" % i
            for i in range(20)
        )
        result = build_parent_child_chunks(text, file_id=3)
        parent_ids = {p["parent_id"] for p in result["parents"]}
        for child in result["children"]:
            assert child["parent_id"] in parent_ids

    def test_parent_content_equals_children(self):
        text = "\n".join(
            "段落 %d 的内容文字，用于测试聚合。" % i
            for i in range(15)
        )
        result = build_parent_child_chunks(
            text, file_id=1, children_per_parent=4
        )
        for parent in result["parents"]:
            group = [
                c["content"]
                for c in result["children"]
                if c["parent_id"] == parent["parent_id"]
            ]
            assert parent["content"] == "\n".join(group)

    def test_enumeration_points_kept_separate(self):
        text = (
            "（1）第一个编号点的具体内容。\n"
            "（2）第二个编号点的具体内容。\n"
            "（3）第三个编号点的具体内容。"
        )
        result = build_parent_child_chunks(text, file_id=1)
        # 三个编号点不应合并进同一个子块
        merged = [
            c["content"] for c in result["children"]
        ]
        for marker in ("（1）", "（2）", "（3）"):
            containing = [
                c for c in merged if marker in c
            ]
            assert len(containing) == 1
        # 没有任何一个子块同时含两个编号点
        for content in merged:
            hits = sum(
                marker in content
                for marker in ("（1）", "（2）", "（3）")
            )
            assert hits <= 1

    def test_full_content_conservation(self):
        text = (
            "开篇介绍系统总体目标与背景。\n"
            "（1）优化器与调度：采用 AdamW，lr=1e-3。\n"
            "（2）数据迭代：批大小为 512。\n"
            "（3）监控指标：跟踪损失与准确率。\n"
            "（4）早停机制：patience 设为 30。\n"
            "最后给出实验结论与未来工作方向。"
        )
        result = build_parent_child_chunks(
            text,
            file_id=1,
            child_max=200,
            children_per_parent=4,
        )
        joined = re.sub(
            r"\s",
            "",
            "".join(c["content"] for c in result["children"]),
        )
        assert joined == re.sub(r"\s", "", text)
