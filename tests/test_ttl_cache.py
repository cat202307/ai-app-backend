"""
TTLCache（LRU + TTL 本地缓存）单元测试。
"""

import time

import pytest

from core.cache import TTLCache


class TestTTLCacheBasics:
    def test_set_get(self):
        c = TTLCache(maxsize=10)
        c["a"] = 1
        assert c.get("a") == 1
        assert "a" in c

    def test_missing_default(self):
        c = TTLCache(maxsize=10)
        assert c.get("x") is None
        assert c.get("x", 42) == 42
        with pytest.raises(KeyError):
            _ = c["x"]

    def test_overwrite(self):
        c = TTLCache(maxsize=10)
        c["a"] = 1
        c["a"] = 2
        assert c["a"] == 2
        assert len(c) == 1

    def test_pop(self):
        c = TTLCache(maxsize=10)
        c["a"] = 1
        assert c.pop("a") == 1
        assert c.pop("a", None) is None
        with pytest.raises(KeyError):
            c.pop("missing")

    def test_clear(self):
        c = TTLCache(maxsize=10)
        c["a"] = 1
        c["b"] = 2
        c.clear()
        assert len(c) == 0


class TestLRUEviction:
    def test_maxsize_evicts_oldest(self):
        c = TTLCache(maxsize=2)
        c["a"] = 1
        c["b"] = 2
        # 插入第三个，应淘汰最久未访问的 a
        c["c"] = 3
        assert "a" not in c
        assert "b" in c
        assert "c" in c
        assert len(c) == 2
        assert c.evictions == 1

    def test_get_refreshes_lru_order(self):
        c = TTLCache(maxsize=2)
        c["a"] = 1
        c["b"] = 2
        # 访问 a，让它成为最近使用，淘汰顺序应为 b
        assert c["a"] == 1
        c["c"] = 3
        assert "a" in c
        assert "b" not in c
        assert c["c"] == 3


class TestTTLExpiry:
    def test_expired_entry_returns_default(self):
        c = TTLCache(maxsize=10, ttl=0.05)
        c["a"] = 1
        assert c["a"] == 1
        time.sleep(0.08)
        assert c.get("a") is None
        assert "a" not in c
        assert c.expired >= 1

    def test_no_ttl_when_zero(self):
        c = TTLCache(maxsize=10, ttl=0)
        c["a"] = 1
        time.sleep(0.02)
        assert c["a"] == 1


class TestStats:
    def test_hit_miss_counters(self):
        c = TTLCache(maxsize=10)
        c["a"] = 1
        c["a"]
        c["a"]
        c.get("nope")
        s = c.stats()
        assert s["hits"] == 2
        assert s["misses"] == 1
        assert s["maxsize"] == 10

    def test_invalid_maxsize(self):
        with pytest.raises(ValueError):
            TTLCache(maxsize=0)
