import re
import threading
from typing import Any

import chromadb
import jieba
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from core.cache import TTLCache
from core.config import settings
from core.metrics import metrics


# ============================================================
# 缓存
# ============================================================

# Embedding 向量缓存：带 LRU 容量上限和 TTL 过期，
# 避免文档/查询累积时内存无限增长。
_embedding_cache = TTLCache(
    maxsize=settings.embedding_cache_maxsize,
    ttl=settings.embedding_cache_ttl or None,
    name="embedding",
)
metrics.register_cache("embedding", _embedding_cache)

# BM25 关键词索引缓存：
# 语料（全部 chunk）只在「建索引 / 删除文件」时发生变化。
# 若每次查询都重新全量分词 + new BM25Okapi，延迟会随文档量
# 线性增长。这里缓存构建好的全量 BM25 索引，索引发生变更时
# 通过 invalidate_keyword_index() 显式失效，下一次查询惰性重建。
_bm25_lock = threading.Lock()
_bm25_state: dict[str, Any] = {
    "valid": False,   # 缓存是否有效
    "bm25": None,     # BM25Okapi 实例（语料为空时为 None）
    "entries": [],    # 与 BM25 语料对齐的 chunk 条目（含分词）
}


def get_embedding(text: str) -> list[float]:
    cached = _embedding_cache.get(text)
    if cached is not None:
        return cached
    embedding = embedding_model.encode(text).tolist()
    _embedding_cache[text] = embedding
    return embedding


def invalidate_keyword_index() -> None:
    """
    使 BM25 关键词索引缓存失效。

    在向量索引发生变化（新建索引、删除文档）时调用，
    下一次关键词检索会重新构建一次 BM25；
    其余查询直接复用缓存，避免每请求全量重建。
    """

    with _bm25_lock:
        _bm25_state["valid"] = False
        _bm25_state["bm25"] = None


def clear_cache() -> None:
    """
    清空缓存：embedding 缓存，并使 BM25 索引缓存失效。
    """
    _embedding_cache.clear()
    invalidate_keyword_index()


# ChromaDB 本地向量数据库保存目录
CHROMA_DIR = "vector_db"

# collection 名称
# 可以理解成向量数据库中的一张表
COLLECTION_NAME = "documents"


# 加载 Embedding 模型
# bge-m3：多语言、中文语义表征强，输出 1024 维稠密向量
embedding_model = SentenceTransformer(
    "BAAI/bge-m3"
)


# 创建 ChromaDB 持久化客户端
# 数据会保存到 vector_db 文件夹中
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)


# 获取或创建 collection
# collection 用来存放文档 chunk、向量、metadata
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME
)


def build_vector_index(
    file_id: int,
    filename: str,
    children: list[dict[str, Any]],
) -> int:
    """
    为指定文件建立向量索引（父子块架构：只存子块）。

    参数：
        file_id：上传文件 ID
        filename：文件名
        children：build_parent_child_chunks 返回的子块列表，每个含
            child_id / parent_id / content / child_index / parent_index

    返回：
        写入向量数据库的子块数量
    """

    # 索引即将变更，先使 BM25 关键词索引缓存失效，
    # 避免在删除旧 chunk 与写入新 chunk 之间命中过期缓存。
    invalidate_keyword_index()

    # 先删除当前 file_id 对应的旧索引
    # 避免同一个文件多次建立索引后出现重复数据
    try:
        collection.delete(
            where={
                "file_id": file_id
            }
        )
    except Exception:
        # 如果之前没有索引，delete 可能会失败
        # 这里忽略即可，不影响后续新建索引
        pass

    ids = []
    embeddings = []
    documents = []
    metadatas = []

    for child in children:
        content = child["content"]
        child_index = child["child_index"]

        # 子块唯一 ID
        ids.append(child["child_id"])

        # 子块文本转向量（子块主题聚焦，向量更精准）。
        # 走 get_embedding 复用带 TTL 的 LRU 缓存，
        # 重建索引 / 重复子块时避免重复向量化。
        embeddings.append(get_embedding(content))

        documents.append(content)

        # metadata：chunk_index 保留为 child_index 以兼容检索，
        # 额外记录 parent_id / parent_index，用于命中后聚合父块
        metadatas.append({
            "file_id": file_id,
            "filename": filename,
            "chunk_index": child_index,
            "child_index": child_index,
            "parent_id": child["parent_id"],
            "parent_index": child["parent_index"],
        })

    # 写入 ChromaDB
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )

    return len(children)


def get_parent_content(parent_id: str) -> str:
    """
    按 parent_id 聚合同组所有子块，得到父块完整内容。

    子块按 child_index 升序拼接；父块不单独存储，
    检索命中子块后实时聚合。
    """

    result = collection.get(
        where={"parent_id": parent_id},
        include=["documents", "metadatas"],
    )

    pairs = list(
        zip(
            result.get("documents", []),
            result.get("metadatas", []),
        )
    )
    pairs.sort(
        key=lambda pair: pair[1].get("child_index", 0)
    )

    return "\n".join(doc for doc, _ in pairs)



def delete_vector_index(file_id: int) -> int:
    """
    删除指定文件的全部向量索引。

    在删除文档时调用，避免向量库中残留孤儿 chunk
    （全局检索会检索全部文档，残留数据会造成错误引用）。

    返回删除的 chunk 数量；无法精确获取时返回 0。
    """

    # 删除会改变语料，使 BM25 关键词索引缓存失效
    invalidate_keyword_index()

    try:
        existing = collection.get(
            where={
                "file_id": file_id
            },
            include=[],
        )

        chunk_count = len(
            existing.get("ids", [])
        )

        collection.delete(
            where={
                "file_id": file_id
            }
        )

        return chunk_count
    except Exception:
        return 0


def get_opening_chunks(
    file_id: int | None = None,
    chunks_per_file: int = 1,
) -> list[dict[str, Any]]:
    """
    取文档开头的父块（父子块架构）。

    概述类问题（“主要内容 / 讲了什么 / 摘要”）不指向某个具体子块，
    Cross-Encoder 对这类问题给分接近基线（约 50），无法与无关问题区分。
    文档开头的父块通常包含标题、摘要和引言，正好用于概述。

    参数：
        file_id：指定文档时只取该文档；为空时每个文档各取前 N 个父块。
        chunks_per_file：每个文档取多少个开头父块。

    返回的每条以父块为单元（chunk_id 用 parent_id，content 为父块内容）。
    """

    # 取全部子块的 metadata，按文档分组找开头父块
    if file_id is not None:
        result = collection.get(
            where={"file_id": file_id},
            include=["metadatas"],
        )
    else:
        result = collection.get(include=["metadatas"])

    # 每个文档需要取的 parent_index 集合
    wanted: dict[Any, set[int]] = {}
    filenames: dict[Any, str] = {}

    for meta in result.get("metadatas", []):
        f_id = meta.get("file_id")
        p_idx = meta.get("parent_index", 0)
        filenames[f_id] = meta.get("filename")
        wanted.setdefault(f_id, set())
        if p_idx < chunks_per_file:
            wanted[f_id].add(p_idx)

    opening: list[dict[str, Any]] = []

    for f_id in sorted(
        wanted.keys(),
        key=lambda value: (value is None, value),
    ):
        for p_idx in sorted(wanted[f_id]):
            parent_id = f"file_{f_id}_parent_{p_idx}"
            opening.append({
                "chunk_id": parent_id,
                "content": get_parent_content(parent_id),
                "file_id": f_id,
                "filename": filenames.get(f_id),
                "chunk_index": p_idx,
                "rerank_score": None,
            })

    return opening



def search_similar_chunks(
    question: str,
    file_id: int | None = None,
    top_k: int = 3
) -> list[dict]:
    """
    根据用户问题，从向量数据库中检索相似文档片段。

    参数：
        question：用户问题
        file_id：指定在哪个文件中检索
        top_k：返回最相似的几个 chunk

    返回：
        [
            {
                "chunk_id": "file_1_chunk_0",
                "content": "文档片段内容",
                "distance": 0.38
            }
        ]

    说明：
        distance 越小，通常表示越相似。
    """

    # ============================================================
    # 1. 把用户问题转换成 Embedding
    # ============================================================

    # 走带 TTL 的 LRU 缓存：相同问题 / 改写后的重复子查询
    # 不再重复调用 embedding 模型。
    query_embedding = get_embedding(question)

    # ============================================================
    # 2. 在 ChromaDB 中进行向量检索
    # ============================================================

    query_kwargs = dict(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    # file_id 为 None 时不做文件过滤，检索全部文档
    if file_id is not None:
        query_kwargs["where"] = {
            "file_id": file_id
        }

    result = collection.query(**query_kwargs)

    # ============================================================
    # 3. 获取查询结果
    # ============================================================

    # ChromaDB 中保存的 chunk ID
    ids = result.get("ids", [])

    # 文本内容
    documents = result.get("documents", [])

    # 向量距离
    distances = result.get("distances", [])

    # 元数据（含 file_id / filename）
    metadatas = result.get("metadatas", [])

    # 如果没有结果
    if not documents or not documents[0]:
        return []

    # ============================================================
    # 4. 整理结果
    # ============================================================

    data = []

    for index, document in enumerate(documents[0]):

        chunk_id = None
        distance = None

        # 获取 chunk ID
        if (
            ids
            and ids[0]
            and index < len(ids[0])
        ):
            chunk_id = ids[0][index]

        # 获取 distance
        if (
            distances
            and distances[0]
            and index < len(distances[0])
        ):
            distance = distances[0][index]

        item_meta = {}

        if (
            metadatas
            and metadatas[0]
            and index < len(metadatas[0])
        ):
            item_meta = metadatas[0][index] or {}

        data.append({
            "chunk_id": chunk_id,
            "content": document,
            "distance": distance,
            "file_id": item_meta.get(
                "file_id",
                file_id,
            ),
            "filename": item_meta.get(
                "filename"
            ),
            "parent_id": item_meta.get("parent_id"),
            "parent_index": item_meta.get("parent_index"),
        })

    return data

# ============================================================
# BM25 关键词检索：停用词 / jieba 分词
# ============================================================

# 停用词：这些词信息量低，不应主导关键词得分
STOPWORDS = frozenset({
    # 英文
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "of", "to", "in", "on", "at", "and", "or", "for", "with",
    "as", "by", "this", "that", "these", "those", "it", "its",
    "what", "which", "how", "do", "does", "did",
    # 中文
    "的", "了", "是", "在", "和", "与", "或", "也", "都", "就",
    "我", "你", "他", "她", "它", "们", "这", "那", "些", "个",
    "吗", "呢", "吧", "啊", "什么", "怎么", "怎样", "如何",
    "请问", "一下", "有", "中", "为", "对", "把", "被", "让",
    "可以", "需要", "关于", "以及", "等",
})

# 向 jieba 注册专业术语，避免被错误切散
for _custom_word in (
    "kubernetes",
    "docker",
    "深度学习",
    "多任务学习",
    "异构资源",
    "数据预处理",
    "特征工程",
    "交叉编码器",
    "重排序",
):
    jieba.add_word(_custom_word)


def tokenize_text(text: str) -> list[str]:
    """
    用 jieba 对中英文混合文本分词，供 BM25 使用。

    - 英文/数字/技术词整体保留
    - 中文用 jieba 分词
    - 去除停用词、空白和标点

    返回保留重复的 token 列表（BM25 需要词频信息）。
    """

    if not text:
        return []

    normalized = text.strip().lower()
    tokens: list[str] = []

    # 英文 / 数字 / 技术词
    for english_word in re.findall(
        r"[a-z0-9_+#.\-]+",
        normalized,
    ):
        if english_word and english_word not in STOPWORDS:
            tokens.append(english_word)

    # 中文片段用 jieba 分词
    for chinese_part in re.findall(
        r"[\u4e00-\u9fff]+",
        normalized,
    ):
        for word in jieba.cut(chinese_part):
            word = word.strip()
            if (
                word
                and word not in STOPWORDS
                and not re.fullmatch(r"[\s\W_]+", word)
            ):
                tokens.append(word)

    return tokens


def extract_lexical_terms(
    text: str,
) -> list[str]:
    """
    提取 Keyword Search（关键词检索）需要使用的词项。

    lexical
        = 词汇的、字面的

    term
        = 词项、关键词

    例如：

        fastAI 是什么？

    可能提取：

        [
            "fastai",
            "是什",
            "什么"
        ]

    中文当前使用简单二元词组，
    先用于理解 Hybrid Search 原理。

    后面可以继续升级：
        - jieba 分词
        - BM25
        - Elasticsearch
    """

    # 去掉首尾空格，并统一转成小写
    normalized_text = (
        text
        .strip()
        .lower()
    )

    if not normalized_text:
        return []

    terms: list[str] = []

    # ============================================================
    # 1. 英文 / 数字 / 技术词提取
    # ============================================================

    english_terms = re.findall(
        r"[a-z0-9_+#.\-]+",
        normalized_text,
    )

    for term in english_terms:

        if term not in terms:
            terms.append(term)

    # ============================================================
    # 2. 中文连续文本提取
    # ============================================================

    chinese_parts = re.findall(
        r"[\u4e00-\u9fff]+",
        normalized_text,
    )

    for part in chinese_parts:

        # 一个汉字时直接保留
        if len(part) == 1:

            if part not in terms:
                terms.append(part)

            continue

        # ========================================================
        # 中文二元词组
        #
        # 例如：
        #
        # 文档主要内容
        #
        # ↓
        #
        # 文档
        # 档主
        # 主要
        # 要内
        # 内容
        # ========================================================

        for index in range(
            len(part) - 1
        ):

            bigram = part[
                index:index + 2
            ]

            if bigram not in terms:
                terms.append(bigram)

    return terms

def _get_bm25_corpus() -> tuple[Any, list[dict[str, Any]]]:
    """
    获取（必要时惰性构建）全量 BM25 索引。

    语料只在文件索引/删除时变化，因此 BM25 只构建一次并缓存复用，
    不再每次查询都全量分词 + new BM25Okapi。

    返回：
        (bm25, entries)
        entries 与 BM25 语料下标对齐，每项含 chunk_id/content/
        tokens/file_id/filename 等。语料为空时 bm25 为 None。
    """

    with _bm25_lock:
        if _bm25_state["valid"]:
            return _bm25_state["bm25"], _bm25_state["entries"]

        # 取全量 chunk（不按 file_id 过滤，缓存一份全量索引，
        # 单文件过滤在打分阶段做，避免每个文件单独建索引）。
        result = collection.get(
            include=["documents", "metadatas"],
        )

        ids = result.get("ids", []) or []
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []

        entries: list[dict[str, Any]] = []

        for index, document in enumerate(documents):
            if not document:
                continue

            doc_tokens = tokenize_text(document)

            # 全为停用词/无有效 token 的 chunk 不进语料，
            # 与 BM25Okapi 语料下标保持一一对应。
            if not doc_tokens:
                continue

            item_meta = (
                metadatas[index]
                if index < len(metadatas)
                else {}
            ) or {}

            entries.append({
                "chunk_id": (
                    ids[index]
                    if index < len(ids)
                    else None
                ),
                "content": document,
                "tokens": doc_tokens,
                "file_id": item_meta.get("file_id"),
                "filename": item_meta.get("filename"),
                "parent_id": item_meta.get("parent_id"),
                "parent_index": item_meta.get("parent_index"),
            })

        bm25 = (
            BM25Okapi([entry["tokens"] for entry in entries])
            if entries
            else None
        )

        _bm25_state.update(
            bm25=bm25,
            entries=entries,
            valid=True,
        )

        return bm25, entries


def keyword_search_chunks(
    question: str,
    file_id: int | None = None,
    top_k: int = 3,
) -> list[dict]:
    """
    Keyword Search（关键词检索，BM25 索引缓存复用）。

    作用：
        不使用 Embedding，
        而是根据用户问题中的关键词，
        在指定文件的 Chunk 中进行字面匹配。

    参数：

        question
            用户问题。

        file_id
            文件 ID。

        top_k
            Top K，
            即最终返回排名最高的前 K 个 Chunk。

    返回示例：

        [
            {
                "chunk_id": "file_1_chunk_0",
                "content": "关于 fastAI 的应用说明。",
                "keyword_score": 1.0,
                "matched_terms": [
                    "fastai"
                ]
            }
        ]

    keyword_score：
        关键词匹配得分。

        与 Vector Search 的 distance 不一样：

        distance：
            越小通常越相关。

        keyword_score：
            越大表示字面匹配越强。
    """

    if not question or not question.strip():
        return []

    # 1. 查询分词（jieba + 停用词）
    query_tokens = tokenize_text(question)
    if not query_tokens:
        return []

    # 查询词项（去重、保序）
    query_terms: list[str] = []
    for token in query_tokens:
        if token not in query_terms:
            query_terms.append(token)

    # 2. 复用缓存的全量 BM25 索引（命中缓存时不再全量分词、重建索引）
    bm25, entries = _get_bm25_corpus()

    if bm25 is None or not entries:
        return []

    # BM25 在全量语料上打分；IDF 基于全语料统计，更稳定。
    # 单文件检索在打分后按 file_id 过滤，无需为每个文件单独建索引。
    scores = bm25.get_scores(query_tokens)

    # 3. 组装结果（按需过滤 file_id）
    scored_results = []

    for index, entry in enumerate(entries):

        # 单文件检索：跳过不属于目标文件的 chunk
        if (
            file_id is not None
            and entry.get("file_id") != file_id
        ):
            continue

        score = scores[index]

        # score <= 0 表示没有有效命中
        if score <= 0:
            continue

        doc_token_set = set(entry["tokens"])

        # 该 chunk 实际命中的查询词项（token 级精确匹配）
        matched_terms = [
            term
            for term in query_terms
            if term in doc_token_set
        ]

        scored_results.append({
            "chunk_id": entry.get("chunk_id"),
            "content": entry["content"],
            "keyword_score": float(score),
            "matched_terms": matched_terms,
            "file_id": entry.get(
                "file_id",
                file_id,
            ),
            "filename": entry.get("filename"),
            "parent_id": entry.get("parent_id"),
            "parent_index": entry.get("parent_index"),
        })

    # 4. 按 BM25 得分从高到低排序，返回 Top K
    scored_results.sort(
        key=lambda item: item["keyword_score"],
        reverse=True,
    )

    return scored_results[:top_k]



