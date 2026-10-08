import re
from typing import Any

import chromadb
import jieba
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


# ============================================================
# 缓存
# ============================================================

_embedding_cache: dict[str, list[float]] = {}
_vector_search_cache: dict[tuple, list[dict]] = {}
_keyword_search_cache: dict[tuple, list[dict]] = {}


def get_embedding(text: str) -> list[float]:
    if text in _embedding_cache:
        return _embedding_cache[text]
    embedding = embedding_model.encode(text).tolist()
    _embedding_cache[text] = embedding
    return embedding


def clear_cache() -> None:
    _embedding_cache.clear()
    _vector_search_cache.clear()
    _keyword_search_cache.clear()


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

        # 子块文本转向量（子块主题聚焦，向量更精准）
        embeddings.append(
            embedding_model.encode(content).tolist()
        )

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

    query_embedding = embedding_model.encode(
        question
    ).tolist()

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

def keyword_search_chunks(
    question: str,
    file_id: int | None = None,
    top_k: int = 3,
) -> list[dict]:
    """
    Keyword Search（关键词检索）。

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

    # ============================================================
    # 1. 从 ChromaDB 获取当前文件全部 Chunk
    # ============================================================

    get_kwargs = dict(
        include=[
            "documents",
            "metadatas",
        ],
    )

    # file_id 为 None 时获取全部文档的 Chunk
    if file_id is not None:
        get_kwargs["where"] = {
            "file_id": file_id
        }

    result = collection.get(**get_kwargs)

    # Chunk ID
    ids = result.get(
        "ids",
        []
    )

    # Chunk 文本
    documents = (
        result.get(
            "documents"
        )
        or []
    )

    # Chunk 元数据
    metadatas = (
        result.get(
            "metadatas"
        )
        or []
    )

    if not documents:
        return []

    # ============================================================
    # 2. 查询分词（jieba）
    # ============================================================

    query_tokens = tokenize_text(question)

    if not query_tokens:
        return []

    # 查询词项（去重、保序）
    query_terms: list[str] = []
    for token in query_tokens:
        if token not in query_terms:
            query_terms.append(token)

    # ============================================================
    # 3. 对每个 Chunk 分词并构建 BM25 索引
    # ============================================================

    # (原始 documents 下标, 该 chunk 的 token 列表)
    corpus_entries: list[tuple[int, list[str]]] = []

    for index, document in enumerate(documents):

        if not document:
            continue

        doc_tokens = tokenize_text(document)

        if doc_tokens:
            corpus_entries.append((index, doc_tokens))

    if not corpus_entries:
        return []

    bm25 = BM25Okapi(
        [tokens for _, tokens in corpus_entries]
    )

    # BM25 相关性得分
    scores = bm25.get_scores(query_tokens)

    # ============================================================
    # 4. 组装结果
    # ============================================================

    scored_results = []

    for (index, doc_tokens), score in zip(
        corpus_entries,
        scores,
    ):

        # score <= 0 表示没有有效命中
        if score <= 0:
            continue

        doc_token_set = set(doc_tokens)

        # 该 chunk 实际命中的查询词项（token 级精确匹配）
        matched_terms = [
            term
            for term in query_terms
            if term in doc_token_set
        ]

        chunk_id = None

        if index < len(ids):
            chunk_id = ids[index]

        item_meta = {}

        if index < len(metadatas):
            item_meta = metadatas[index] or {}

        scored_results.append({
            "chunk_id": chunk_id,
            "content": documents[index],
            "keyword_score": float(score),
            "matched_terms": matched_terms,
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

    # ============================================================
    # 5. 按 BM25 得分从高到低排序
    # ============================================================

    scored_results.sort(
        key=lambda item: item[
            "keyword_score"
        ],
        reverse=True,
    )

    # 返回 Top K
    return scored_results[
        :top_k
    ]



