"""
向量服务与缓存测试。
"""

import pytest

from services.vector_service import (
    get_embedding,
    clear_cache,
    _embedding_cache,
    _bm25_state,
    invalidate_keyword_index,
    keyword_search_chunks,
    extract_lexical_terms,
)


class TestEmbeddingCache:
    """
    Embedding 缓存测试。
    """

    def setup_method(self):
        """
        每个测试前清空缓存。
        """
        clear_cache()

    def test_get_embedding_returns_list(self, embedding_model):
        """
        测试 get_embedding 返回 list 类型。
        """
        result = get_embedding("测试文本")
        assert isinstance(result, list)
        assert len(result) > 0
        assert isinstance(result[0], float)

    def test_embedding_cache_hit(self, embedding_model):
        """
        测试相同文本第二次调用命中缓存。
        """
        text = "缓存测试文本"

        # 第一次调用
        result1 = get_embedding(text)
        assert text in _embedding_cache

        # 第二次调用
        result2 = get_embedding(text)

        # 结果应该完全一致
        assert result1 == result2
        assert len(_embedding_cache) == 1

    def test_embedding_cache_miss(self, embedding_model):
        """
        测试不同文本缓存不命中。
        """
        text1 = "第一个文本"
        text2 = "第二个文本"

        get_embedding(text1)
        get_embedding(text2)

        assert len(_embedding_cache) == 2

    def test_clear_cache(self, embedding_model):
        """
        测试清空缓存：embedding 清空，BM25 索引失效。
        """
        get_embedding("测试")
        assert len(_embedding_cache) > 0

        # 先让 BM25 缓存处于有效状态
        _bm25_state["valid"] = True

        clear_cache()

        assert len(_embedding_cache) == 0
        assert _bm25_state["valid"] is False


class TestBm25IndexCache:
    """
    BM25 关键词索引缓存复用与失效测试。

    用假的 collection 替换真实 Chroma，只验证缓存行为：
    命中缓存时不再全量拉取语料，失效后才重建。
    """

    def setup_method(self):
        invalidate_keyword_index()

    def teardown_method(self):
        # 防止 monkeypatch 的假语料残留在缓存里影响其它测试
        invalidate_keyword_index()

    def _patch_collection(self, monkeypatch, documents, metadatas):
        import services.vector_service as vs

        calls = {"get": 0}

        class FakeCollection:
            def get(self, **kwargs):
                calls["get"] += 1
                return {
                    "ids": [
                        "chunk_%d" % i
                        for i in range(len(documents))
                    ],
                    "documents": documents,
                    "metadatas": metadatas,
                }

        monkeypatch.setattr(vs, "collection", FakeCollection())
        return calls

    def test_index_built_once_and_reused(self, monkeypatch):
        """两次查询只拉取一次全量语料、结果一致。"""
        # 语料需足够大，保证查询词只出现在少数文档中
        # （BM25 的 IDF 对“小语料里的常见词”可能给 0 分）。
        docs = [
            "FastAPI 性能优化方法",
            "向量检索原理详解",
            "数据库连接池配置",
            "缓存策略与失效",
            "日志采集方案",
            "容器化部署实践",
        ]
        metas = [
            {"file_id": 1, "filename": "a.txt"}
            for _ in docs
        ]
        calls = self._patch_collection(monkeypatch, docs, metas)

        r1 = keyword_search_chunks(
            "FastAPI", file_id=None, top_k=5
        )
        r2 = keyword_search_chunks(
            "FastAPI", file_id=None, top_k=5
        )

        # 缓存命中：两次查询只触发一次全量语料拉取
        assert calls["get"] == 1
        assert r1 and r1[0]["chunk_id"] == "chunk_0"
        assert r1 == r2

    def test_invalidate_forces_rebuild(self, monkeypatch):
        """失效后下一次查询会重新构建索引。"""
        docs = ["测试内容说明"]
        metas = [{"file_id": 1}]
        calls = self._patch_collection(monkeypatch, docs, metas)

        keyword_search_chunks("测试")
        keyword_search_chunks("测试")
        assert calls["get"] == 1

        invalidate_keyword_index()
        keyword_search_chunks("测试")
        assert calls["get"] == 2

    def test_file_id_filter_on_cached_corpus(self, monkeypatch):
        """全量缓存上按 file_id 过滤，仍只返回目标文件片段。"""
        docs = ["FastAPI 性能", "无关内容文档", "FastAPI 部署"]
        metas = [
            {"file_id": 1, "filename": "a.txt"},
            {"file_id": 2, "filename": "b.txt"},
            {"file_id": 3, "filename": "c.txt"},
        ]
        self._patch_collection(monkeypatch, docs, metas)

        result = keyword_search_chunks(
            "FastAPI", file_id=1, top_k=5
        )

        assert result
        assert all(
            item["file_id"] == 1 for item in result
        )
        assert len(result) == 1
        assert result[0]["chunk_id"] == "chunk_0"


class TestLexicalTerms:
    """
    关键词提取测试。
    """

    def test_extract_english_terms(self):
        """
        测试提取英文关键词。
        """
        terms = extract_lexical_terms("FastAPI is a web framework")
        assert "fastapi" in terms
        assert "web" in terms

    def test_extract_chinese_bigrams(self):
        """
        测试提取中文二元词组。
        """
        terms = extract_lexical_terms("人工智能")
        assert "人工" in terms
        assert "智能" in terms

    def test_extract_mixed_text(self):
        """
        测试混合中英文文本。
        """
        terms = extract_lexical_terms("FastAPI 是一个 Web 框架")
        assert "fastapi" in terms
        assert "web" in terms
        assert "是一" in terms or "个" in terms

    def test_empty_text(self):
        """
        测试空文本。
        """
        terms = extract_lexical_terms("")
        assert terms == []

    def test_single_chinese_char(self):
        """
        测试单个汉字。
        """
        terms = extract_lexical_terms("好")
        assert "好" in terms
