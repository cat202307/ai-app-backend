"""
父块提升逻辑测试（promote_children_to_parents）。

纯逻辑 + monkeypatch get_parent_content（不访问向量库）：
  - 同父块多子块去重、保留最高 rerank 分
  - 不同父块按最高分降序
  - 缺 parent_id 时以子块兜底
  - 空输入、None 分数处理
"""

import pytest

from services import advanced_rag_answer_service as ars


def make_child(
    child_index,
    parent_index,
    score,
    parent_id=None,
    file_id=1,
    filename="paper.docx",
):
    """构造一个命中子块。"""
    pid = (
        parent_id
        if parent_id is not None
        else "file_%d_parent_%d" % (file_id, parent_index)
    )
    return {
        "chunk_id": "file_%d_child_%d" % (file_id, child_index),
        "parent_id": pid,
        "content": "child content %d" % child_index,
        "file_id": file_id,
        "filename": filename,
        "rerank_score": score,
    }


@pytest.fixture
def fake_parent_content(monkeypatch):
    """用确定性内容替换实时父块聚合。"""
    def _get(parent_id):
        return "AGGREGATED::" + parent_id

    monkeypatch.setattr(ars, "get_parent_content", _get)
    return _get


class TestPromoteChildrenToParents:
    def test_empty_input(self, fake_parent_content):
        assert ars.promote_children_to_parents([]) == []

    def test_same_parent_dedup_keeps_highest(
        self, fake_parent_content
    ):
        children = [
            make_child(4, 1, 60.0),
            make_child(5, 1, 70.0),
            make_child(6, 1, 65.0),
        ]
        result = ars.promote_children_to_parents(children)

        assert len(result) == 1
        parent = result[0]
        assert parent["chunk_id"] == "file_1_parent_1"
        assert parent["rerank_score"] == 70.0
        assert parent["content"] == (
            "AGGREGATED::file_1_parent_1"
        )

    def test_parents_sorted_by_score_desc(
        self, fake_parent_content
    ):
        children = [
            make_child(0, 0, 55.0),
            make_child(8, 2, 80.0),
            make_child(4, 1, 63.0),
        ]
        result = ars.promote_children_to_parents(children)

        assert len(result) == 3
        scores = [p["rerank_score"] for p in result]
        assert scores == [80.0, 63.0, 55.0]
        assert result[0]["chunk_id"] == "file_1_parent_2"

    def test_missing_parent_id_fallback(
        self, fake_parent_content
    ):
        children = [
            make_child(
                9, 9, 58.0, parent_id=""
            )
        ]
        result = ars.promote_children_to_parents(children)

        assert len(result) == 1
        unit = result[0]
        assert unit["chunk_id"] == "file_1_child_9"
        assert unit["content"] == "child content 9"
        assert unit["rerank_score"] == 58.0

    def test_none_score_then_real_score(
        self, fake_parent_content
    ):
        children = [
            make_child(4, 1, None),
            make_child(5, 1, 60.0),
        ]
        result = ars.promote_children_to_parents(children)
        assert result[0]["rerank_score"] == 60.0

    def test_metadata_propagated(self, fake_parent_content):
        children = [
            make_child(
                4, 1, 70.0, file_id=3, filename="doc.txt"
            )
        ]
        result = ars.promote_children_to_parents(children)
        parent = result[0]
        assert parent["file_id"] == 3
        assert parent["filename"] == "doc.txt"
