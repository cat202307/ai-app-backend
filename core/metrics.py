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

        # 启动时间
        self.start_time: float = time.time()

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
            }


# 全局单例
metrics = MetricsCollector()
