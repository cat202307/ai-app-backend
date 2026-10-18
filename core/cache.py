"""
进程内通用缓存：线程安全的 LRU + TTL。

解决裸 dict 缓存只增不减、内存随使用量无限上涨的问题：
- maxsize：容量上限，超出时淘汰最久未访问的条目（LRU）；
- ttl：过期时间（秒），访问时惰性清理；
- 内部用 threading.Lock 保护，FastAPI 线程池下也安全。

单机定位下用于 embedding、查询改写等结果缓存；
多实例部署时这类缓存应下沉到 Redis（见项目二）。
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Any, Hashable


_MISS = object()


class TTLCache:
    """
    带容量上限与过期时间的本地缓存。

    用法与 dict 基本一致：cache[k] = v、cache.get(k)、k in cache、
    len(cache)、cache.pop(k, None)、cache.clear()。
    """

    def __init__(
        self,
        maxsize: int = 1024,
        ttl: float | None = None,
        *,
        name: str = "",
    ) -> None:
        if maxsize <= 0:
            raise ValueError("maxsize 必须为正整数")
        self.maxsize = maxsize
        # ttl=None 或 ttl<=0 表示不过期
        self.ttl = ttl if ttl and ttl > 0 else None
        self.name = name
        self._data: OrderedDict[Hashable, tuple[Any, float | None]] = (
            OrderedDict()
        )
        self._lock = threading.Lock()

        # 命中 / 未命中 / 淘汰计数，便于观测与排查
        self.hits = 0
        self.misses = 0
        self.expired = 0
        self.evictions = 0

    def _expire_at(self) -> float | None:
        if self.ttl is None:
            return None
        return time.monotonic() + self.ttl

    def _is_alive(self, expire_at: float | None) -> bool:
        return expire_at is None or expire_at >= time.monotonic()

    def get(self, key: Hashable, default: Any = None) -> Any:
        with self._lock:
            item = self._data.get(key)
            if item is None:
                self.misses += 1
                return default
            value, expire_at = item
            if not self._is_alive(expire_at):
                self._data.pop(key, None)
                self.expired += 1
                self.misses += 1
                return default
            self._data.move_to_end(key)
            self.hits += 1
            return value

    def __getitem__(self, key: Hashable) -> Any:
        with self._lock:
            item = self._data.get(key)
            if item is None:
                raise KeyError(key)
            value, expire_at = item
            if not self._is_alive(expire_at):
                self._data.pop(key, None)
                self.expired += 1
                raise KeyError(key)
            self._data.move_to_end(key)
            self.hits += 1
            return value

    def __setitem__(self, key: Hashable, value: Any) -> None:
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
            self._data[key] = (value, self._expire_at())
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)
                self.evictions += 1

    def __contains__(self, key: Hashable) -> bool:
        with self._lock:
            item = self._data.get(key)
            if item is None:
                return False
            if not self._is_alive(item[1]):
                self._data.pop(key, None)
                self.expired += 1
                return False
            return True

    def pop(self, key: Hashable, default: Any = _MISS) -> Any:
        with self._lock:
            item = self._data.pop(key, None)
        if item is None:
            if default is _MISS:
                raise KeyError(key)
            return default
        value, expire_at = item
        if not self._is_alive(expire_at):
            return default if default is not _MISS else None
        return value

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self.hits = 0
            self.misses = 0
            self.expired = 0
            self.evictions = 0

    def __len__(self) -> int:
        with self._lock:
            self._purge_locked()
            return len(self._data)

    def _purge_locked(self) -> None:
        now = time.monotonic()
        dead = [
            k
            for k, (_v, exp) in self._data.items()
            if exp is not None and exp < now
        ]
        for k in dead:
            self._data.pop(k, None)
            self.expired += 1

    def stats(self) -> dict[str, int]:
        """返回命中/未命中/过期/淘汰统计。"""
        with self._lock:
            return {
                "size": len(self._data),
                "maxsize": self.maxsize,
                "hits": self.hits,
                "misses": self.misses,
                "expired": self.expired,
                "evictions": self.evictions,
            }
