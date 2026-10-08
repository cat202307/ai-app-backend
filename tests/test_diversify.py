"""
检索文档多样性与输入有效性测试。

  - diversify_by_file：按文件分组轮转，保证小文档进入候选
  - routers.agent._is_empty_message：纯标点 / 空白判定
"""

from services import advanced_rag_service as ars
from routers import agent as agent_router


def _make_cid(file_id: int, rank: int) -> dict:
    return {
        "chunk_id": f"file_{file_id}_child_{rank}",
        "file_id": file_id,
        "filename": f"doc_{file_id}.txt",
        "rrf_score": 1.0 / (60 + rank),
        "content": f"content {file_id}-{rank}",
    }


class TestDiversifyByFile:
    def test_small_file_kept_in_candidates(self):
        # 前 6 条全是 file_1，第 7、8 条是 file_2
        fused = [_make_cid(1, i) for i in range(1, 7)]
        fused += [_make_cid(2, i) for i in range(1, 3)]

        result = ars.diversify_by_file(
            fused, final_top_k=6, per_file_limit=3
        )

        file_ids = {item["file_id"] for item in result}
        # file_2 必须出现在最终候选中
        assert 2 in file_ids
        assert 1 in file_ids

    def test_respects_final_top_k(self):
        fused = [_make_cid(1, i) for i in range(1, 10)]
        result = ars.diversify_by_file(
            fused, final_top_k=5, per_file_limit=9
        )
        assert len(result) == 5

    def test_round_robin_interleaves_files(self):
        fused = [_make_cid(1, i) for i in range(1, 5)]
        fused += [_make_cid(2, i) for i in range(1, 5)]
        fused += [_make_cid(3, i) for i in range(1, 5)]

        result = ars.diversify_by_file(
            fused, final_top_k=9, per_file_limit=3
        )
        seq = [item["file_id"] for item in result]
        # 轮转：1,2,3,1,2,3,1,2,3
        assert seq == [1, 2, 3, 1, 2, 3, 1, 2, 3]

    def test_empty_input(self):
        assert ars.diversify_by_file([], final_top_k=5) == []


class TestIsEmptyMessage:
    def test_punctuation_only(self):
        assert agent_router._is_empty_message("？？？") is True
        assert agent_router._is_empty_message("...") is True
        assert agent_router._is_empty_message("！！!!") is True
        assert agent_router._is_empty_message("   ") is True
        assert agent_router._is_empty_message(None) is True

    def test_meaningful_content(self):
        assert agent_router._is_empty_message("你好") is False
        assert agent_router._is_empty_message("R²是多少") is False
        assert agent_router._is_empty_message("test02") is False
        assert agent_router._is_empty_message("？为什么") is False
