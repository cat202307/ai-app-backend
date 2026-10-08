"""
向量服务和缓存测试。
"""

import pytest

from services.vector_service import (
    get_embedding,
    clear_cache,
    _embedding_cache,
    _vector_search_cache,
    _keyword_search_cache,
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
        测试清空缓存。
        """
        get_embedding("测试")
        assert len(_embedding_cache) > 0

        clear_cache()

        assert len(_embedding_cache) == 0
        assert len(_vector_search_cache) == 0
        assert len(_keyword_search_cache) == 0


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
