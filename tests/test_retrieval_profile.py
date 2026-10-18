"""
验证评测与生产共用同一套检索参数画像（避免参数漂移）。

通过 monkeypatch 拦截底层检索函数，断言评测在
hybrid_rerank 变体里使用的召回侧参数与生产配置一致。
"""

from core.config import settings
from services import rag_evaluation_service as ev


def test_hybrid_rerank_variant_uses_production_profile(monkeypatch):
    captured = {}

    def fake_hybrid_rerank_retrieve(**kwargs):
        captured.update(kwargs)
        return {"reranked_results": []}

    monkeypatch.setattr(
        ev,
        "hybrid_rerank_retrieve",
        fake_hybrid_rerank_retrieve,
    )

    ev.run_retrieval_variant(
        variant="hybrid_rerank",
        question="测试问题",
        file_id=1,
        top_k=3,
    )

    # 召回侧参数必须与生产配置一致
    assert (
        captured["rewrite_count"]
        == settings.rag_global_rewrite_count
    )
    assert (
        captured["per_retriever_top_k"]
        == settings.rag_per_retriever_top_k
    )
    assert (
        captured["fusion_top_k"]
        == settings.rag_fusion_top_k
    )
    # 最终截断是评测指标口径（top_k），与生产 final_top_k 不同，
    # 这里显式锁定为评测传入的 top_k
    assert captured["final_top_k"] == 3


def test_basic_vector_variant_returns_results(monkeypatch):
    def fake_search(question, file_id, top_k):
        return [{"chunk_id": "a"}]

    monkeypatch.setattr(
        ev, "search_similar_chunks", fake_search
    )
    out = ev.run_retrieval_variant(
        variant="basic_vector",
        question="x",
        file_id=1,
        top_k=3,
    )
    assert out == [{"chunk_id": "a"}]
