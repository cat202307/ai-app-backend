"""
Metrics 收集器测试（core.metrics）。

用局部 MetricsCollector 实例，避免污染全局单例：
  - 检索阶段计时（record_retrieval_stage / stage_timer）
  - LLM token 与成本累计（record_llm_usage）
  - get_summary 结构
"""

import pytest

from core.metrics import MetricsCollector, metrics


class TestRetrievalStages:
    def test_record_stage_accumulates(self):
        mc = MetricsCollector()
        mc.record_retrieval_stage("rerank", 100.0)
        mc.record_retrieval_stage("rerank", 50.0)

        summary = mc.get_summary()
        stages = {
            s["stage"]: s for s in summary["retrieval_stages"]
        }
        rerank = stages["rerank"]
        assert rerank["count"] == 2
        assert rerank["total_ms"] == 150.0
        assert rerank["max_ms"] == 100.0
        assert rerank["avg_ms"] == 75.0

    def test_stage_timer_records(self):
        mc = MetricsCollector()
        with mc.stage_timer("vector_search"):
            pass  # 极快也应记录一次

        stages = {
            s["stage"]: s
            for s in mc.get_summary()["retrieval_stages"]
        }
        assert stages["vector_search"]["count"] == 1


class TestLLMUsage:
    def test_tokens_accumulate(self):
        mc = MetricsCollector()
        mc.record_llm_usage(
            prompt_tokens=100, completion_tokens=50
        )
        mc.record_llm_usage(
            prompt_tokens=200, completion_tokens=80
        )

        usage = mc.get_summary()["llm_usage"]
        assert usage["calls"] == 2
        assert usage["prompt_tokens"] == 300
        assert usage["completion_tokens"] == 130
        assert usage["total_tokens"] == 430

    def test_cost_estimated_from_settings(self):
        # 默认单价 input 2 元 / M、output 8 元 / M
        mc = MetricsCollector()
        mc.record_llm_usage(
            prompt_tokens=1_000_000,
            completion_tokens=1_000_000,
        )
        usage = mc.get_summary()["llm_usage"]
        assert usage["estimated_cost"] == pytest.approx(
            10.0, abs=0.01
        )

    def test_zero_usage_no_cost(self):
        mc = MetricsCollector()
        mc.record_llm_usage(0, 0)
        usage = mc.get_summary()["llm_usage"]
        assert usage["estimated_cost"] == 0
        assert usage["total_tokens"] == 0


class TestSummaryShape:
    def test_summary_contains_new_sections(self):
        mc = MetricsCollector()
        summary = mc.get_summary()
        assert "retrieval_stages" in summary
        assert "llm_usage" in summary

    def test_global_singleton_exists(self):
        assert metrics is not None
        summary = metrics.get_summary()
        assert "llm_usage" in summary


class TestRetrievalRuns:
    """生产检索运行记录（命中状态 / 查询数 / 耗时 / rerank 分）。"""

    def test_match_rate_and_aggregates(self):
        mc = MetricsCollector()
        mc.record_retrieval_run(
            status="matched",
            query_count=3,
            result_count=8,
            best_rerank_score=70.0,
            latency_ms=100.0,
        )
        mc.record_retrieval_run(
            status="not_matched",
            query_count=1,
            result_count=8,
            best_rerank_score=45.0,
            latency_ms=300.0,
        )
        mc.record_retrieval_run(
            status="matched",
            query_count=3,
            result_count=6,
            best_rerank_score=65.0,
            latency_ms=200.0,
        )

        summary = mc.get_summary()["retrieval"]
        assert summary["runs"] == 3
        assert summary["matched"] == 2
        assert summary["not_matched"] == 1
        assert summary["match_rate"] == round(2 / 3, 4)
        assert summary["avg_query_count"] == round(7 / 3, 2)
        assert summary["avg_latency_ms"] == 200.0
        assert summary["max_latency_ms"] == 300.0
        assert summary["rerank_score_min"] == 45.0
        assert summary["rerank_score_max"] == 70.0
        assert summary["rerank_score_avg"] == 60.0

    def test_empty_retrieval_summary(self):
        mc = MetricsCollector()
        summary = mc.get_summary()["retrieval"]
        assert summary["runs"] == 0
        assert summary["match_rate"] == 0.0
        assert summary["rerank_score_min"] is None
        assert summary["recent_runs"] == []


class TestCacheRegistration:
    def test_registered_caches_in_summary(self):
        from core.cache import TTLCache

        mc = MetricsCollector()
        c = TTLCache(maxsize=4)
        c["k"] = "v"
        mc.register_cache("demo", c)
        caches = mc.get_summary()["caches"]
        assert "demo" in caches
        assert caches["demo"]["size"] == 1
        assert caches["demo"]["maxsize"] == 4
