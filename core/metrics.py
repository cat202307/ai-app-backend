"""
性能与用量监控统计（内存级别）。

统计：
  - HTTP 请求数、成功 / 失败、各接口耗时
  - RAG 检索各阶段耗时（query rewrite / vector /
    keyword / RRF / rerank / 父块聚合）
  - LLM token 用量与估算成本
"""

import time
from collections import defaultdict
from contextlib import contextmanager
from threading import Lock


class MetricsCollector:
    """
    指标收集器（线程安全，内存级单例）。
    """

    def __init__(self):
        self._lock = Lock()

        # ---- HTTP 请求 ----
        self.total_requests: int = 0
        self.success_requests: int = 0
        self.error_requests: int = 0

        # key: method + path,
        # value: [count, total_ms, min_ms, max_ms]
        self._path_stats: dict[str, list[float]] = defaultdict(
            lambda: [0, 0.0, float("inf"), 0.0]
        )

        # ---- RAG 检索阶段 ----
        # key: stage, value: [count, total_ms, max_ms]
        self._retrieval_stats: dict[str, list[float]] = (
            defaultdict(lambda: [0, 0.0, 0.0])
        )

        # ---- LLM 用量 ----
        self.llm_calls: int = 0
        self.llm_prompt_tokens: int = 0
        self.llm_completion_tokens: int = 0
        self.llm_estimated_cost: float = 0.0

        # ---- Advanced RAG 检索运行 ----
        # 命中 / 未命中计数、查询数、耗时、rerank 分聚合，
        # 用于阈值与检索参数的迭代。
        self._retrieval_runs: int = 0
        self._retrieval_matched: int = 0
        self._retrieval_not_matched: int = 0
        self._retrieval_query_count: int = 0
        self._retrieval_latency_total_ms: float = 0.0
        self._retrieval_latency_max_ms: float = 0.0
        # 每次 top1 rerank 分（用于观察分数分布与阈值余量），
        # 有界保留，避免内存只增不减。
        self._rerank_scores: list[float] = []
        self._recent_retrieval_runs: list[dict] = []
        self._recent_runs_limit = 100

        # 启动时间
        self.start_time: float = time.time()

        # 已注册的进程内缓存（name -> 缓存对象，提供 stats()）
        self._caches: dict = {}

    # --------------------------------------------------------
    # 缓存
    # --------------------------------------------------------

    def register_cache(self, name: str, cache) -> None:
        """注册一个本地缓存，供 /metrics 汇总命中率与淘汰量。"""
        self._caches[name] = cache

    def cache_stats(self) -> dict:
        """返回所有已注册缓存的统计。"""
        return {
            name: (c.stats() if hasattr(c, "stats") else {})
            for name, c in self._caches.items()
        }

    # --------------------------------------------------------
    # HTTP
    # --------------------------------------------------------

    def record_request(
        self,
        method: str,
        path: str,
        duration_ms: float,
        status_code: int,
    ) -> None:
        """记录一次 HTTP 请求。"""

        with self._lock:
            self.total_requests += 1

            if 200 <= status_code < 400:
                self.success_requests += 1
            else:
                self.error_requests += 1

            key = f"{method} {path}"
            stats = self._path_stats[key]

            stats[0] += 1
            stats[1] += duration_ms

            if duration_ms < stats[2]:
                stats[2] = duration_ms
            if duration_ms > stats[3]:
                stats[3] = duration_ms

    # --------------------------------------------------------
    # RAG 检索阶段
    # --------------------------------------------------------

    def record_retrieval_stage(
        self, stage: str, duration_ms: float
    ) -> None:
        """记录一个检索阶段的耗时。"""

        with self._lock:
            stats = self._retrieval_stats[stage]
            stats[0] += 1
            stats[1] += duration_ms
            if duration_ms > stats[2]:
                stats[2] = duration_ms

    @contextmanager
    def stage_timer(self, stage: str):
        """阶段计时上下文管理器。"""

        started = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (
                time.perf_counter() - started
            ) * 1000.0
            self.record_retrieval_stage(stage, elapsed_ms)

    # --------------------------------------------------------
    # Advanced RAG 检索运行
    # --------------------------------------------------------

    def record_retrieval_run(
        self,
        *,
        status: str,
        query_count: int = 0,
        result_count: int = 0,
        best_rerank_score: float | None = None,
        latency_ms: float = 0.0,
        mode: str = "advanced_rag",
    ) -> None:
        """
        记录一次生产检索运行的结果，供阈值迭代：
        - status：matched / not_matched / overview_opening ...
        - query_count：实际使用的查询数（含改写）
        - result_count：最终保留片段数
        - best_rerank_score：top1 rerank 分
        - latency_ms：端到端检索耗时
        """
        with self._lock:
            self._retrieval_runs += 1
            if status == "matched":
                self._retrieval_matched += 1
            else:
                self._retrieval_not_matched += 1

            self._retrieval_query_count += int(query_count or 0)
            self._retrieval_latency_total_ms += latency_ms
            if latency_ms > self._retrieval_latency_max_ms:
                self._retrieval_latency_max_ms = latency_ms

            if best_rerank_score is not None:
                self._rerank_scores.append(
                    float(best_rerank_score)
                )
                if len(self._rerank_scores) > 1000:
                    self._rerank_scores = self._rerank_scores[-1000:]

            self._recent_retrieval_runs.append({
                "mode": mode,
                "status": status,
                "query_count": int(query_count or 0),
                "result_count": int(result_count or 0),
                "best_rerank_score": best_rerank_score,
                "latency_ms": round(latency_ms, 2),
                "ts": round(time.time(), 1),
            })
            if len(self._recent_retrieval_runs) > (
                self._recent_runs_limit
            ):
                self._recent_retrieval_runs = (
                    self._recent_retrieval_runs[
                        -self._recent_runs_limit:
                    ]
                )

    # --------------------------------------------------------
    # LLM 用量
    # --------------------------------------------------------

    def record_llm_usage(
        self,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> None:
        """
        记录一次 LLM 调用的 token 用量，并按配置单价
        累计估算成本。
        """

        prompt_tokens = int(prompt_tokens or 0)
        completion_tokens = int(completion_tokens or 0)

        # 延迟 import，避免 metrics <-> config 循环依赖
        try:
            from core.config import settings

            input_price = getattr(
                settings,
                "llm_price_per_million_input",
                0.0,
            )
            output_price = getattr(
                settings,
                "llm_price_per_million_output",
                0.0,
            )
        except Exception:
            input_price = 0.0
            output_price = 0.0

        cost = (
            prompt_tokens / 1_000_000.0 * input_price
            + completion_tokens / 1_000_000.0 * output_price
        )

        with self._lock:
            self.llm_calls += 1
            self.llm_prompt_tokens += prompt_tokens
            self.llm_completion_tokens += completion_tokens
            self.llm_estimated_cost += cost

    # --------------------------------------------------------
    # 汇总
    # --------------------------------------------------------

    def get_summary(self) -> dict:
        """获取监控摘要。"""

        with self._lock:
            uptime_seconds = time.time() - self.start_time

            # HTTP 路径统计
            top_paths = []
            for key, stats in self._path_stats.items():
                count, total_ms, min_ms, max_ms = stats
                avg_ms = total_ms / count if count > 0 else 0
                top_paths.append({
                    "path": key,
                    "count": int(count),
                    "avg_ms": round(avg_ms, 2),
                    "min_ms": (
                        round(min_ms, 2)
                        if min_ms != float("inf")
                        else 0
                    ),
                    "max_ms": round(max_ms, 2),
                })
            top_paths.sort(
                key=lambda x: x["avg_ms"], reverse=True
            )

            # 检索阶段统计（按总耗时降序）
            retrieval_stages = []
            for stage, stats in self._retrieval_stats.items():
                count, total_ms, max_ms = stats
                avg_ms = total_ms / count if count > 0 else 0
                retrieval_stages.append({
                    "stage": stage,
                    "count": int(count),
                    "avg_ms": round(avg_ms, 2),
                    "max_ms": round(max_ms, 2),
                    "total_ms": round(total_ms, 2),
                })
            retrieval_stages.sort(
                key=lambda x: x["total_ms"], reverse=True
            )

            total_tokens = (
                self.llm_prompt_tokens
                + self.llm_completion_tokens
            )

            # 检索运行汇总
            runs = self._retrieval_runs
            scores = list(self._rerank_scores)
            scores_sorted = sorted(scores)
            retrieval_summary = {
                "runs": runs,
                "matched": self._retrieval_matched,
                "not_matched": self._retrieval_not_matched,
                "match_rate": round(
                    self._retrieval_matched / runs, 4
                ) if runs else 0.0,
                "avg_query_count": round(
                    self._retrieval_query_count / runs, 2
                ) if runs else 0.0,
                "avg_latency_ms": round(
                    self._retrieval_latency_total_ms / runs, 2
                ) if runs else 0.0,
                "max_latency_ms": round(
                    self._retrieval_latency_max_ms, 2
                ),
                "rerank_score_min": round(
                    scores_sorted[0], 2
                ) if scores_sorted else None,
                "rerank_score_avg": round(
                    sum(scores) / len(scores), 2
                ) if scores else None,
                "rerank_score_max": round(
                    scores_sorted[-1], 2
                ) if scores_sorted else None,
                "recent_runs": list(self._recent_retrieval_runs[-10:]),
            }

            return {
                "uptime_seconds": round(uptime_seconds, 1),
                "total_requests": self.total_requests,
                "success_requests": self.success_requests,
                "error_requests": self.error_requests,
                "avg_request_per_second": round(
                    self.total_requests / uptime_seconds, 2
                ) if uptime_seconds > 0 else 0,
                "top_slow_paths": top_paths[:10],
                "retrieval_stages": retrieval_stages,
                "llm_usage": {
                    "calls": self.llm_calls,
                    "prompt_tokens": self.llm_prompt_tokens,
                    "completion_tokens": (
                        self.llm_completion_tokens
                    ),
                    "total_tokens": total_tokens,
                    "estimated_cost": round(
                        self.llm_estimated_cost, 6
                    ),
                },
                "retrieval": retrieval_summary,
                "caches": self.cache_stats(),
            }


# 全局单例
metrics = MetricsCollector()
