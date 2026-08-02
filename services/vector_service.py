import chromadb
from sentence_transformers import SentenceTransformer


# ChromaDB 本地向量数据库保存目录
CHROMA_DIR = "vector_db"

# collection 名称
# 可以理解成向量数据库中的一张表
COLLECTION_NAME = "documents"


# 加载 Embedding 模型
# 这个模型支持中文和英文语义向量化
embedding_model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


# 创建 ChromaDB 持久化客户端
# 数据会保存到 vector_db 文件夹中
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)


# 获取或创建 collection
# collection 用来存放文档 chunk、向量、metadata
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME
)


def build_vector_index(file_id: int, filename: str, chunks: list[str]) -> int:
    """
    为指定文件建立向量索引。

    参数：
        file_id：上传文件 ID
        filename：文件名
        chunks：文档切分后的文本片段列表

    返回：
        写入向量数据库的 chunk 数量
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

    for index, chunk in enumerate(chunks):
        # 为每个 chunk 创建唯一 ID
        chunk_id = f"file_{file_id}_chunk_{index}"

        # 把文本 chunk 转成向量
        embedding = embedding_model.encode(chunk).tolist()

        # 保存 chunk ID
        ids.append(chunk_id)

        # 保存向量
        embeddings.append(embedding)

        # 保存原始文本内容
        documents.append(chunk)

        # 保存 metadata
        # 方便后续通过 file_id 限定只在某个文件里检索
        metadatas.append({
            "file_id": file_id,
            "filename": filename,
            "chunk_index": index
        })

    # 写入 ChromaDB
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )

    return len(chunks)


def search_similar_chunks(question: str, file_id: int, top_k: int = 3) -> list[dict]:
    """
    根据用户问题，从向量数据库中检索相似文档片段。

    参数：
        question：用户问题
        file_id：指定在哪个文件中检索
        top_k：返回最相似的几个 chunk

    返回：
        [
            {
                "content": "文档片段内容",
                "distance": 0.38
            }
        ]

    说明：
        distance 越小，通常表示越相似。
    """

    # 把用户问题转成向量
    query_embedding = embedding_model.encode(question).tolist()

    # 在 ChromaDB 中查询相似 chunk
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where={
            "file_id": file_id
        }
    )

    # documents 是检索到的文本片段
    documents = result.get("documents", [])

    # distances 是距离分数
    # 一般 distance 越小，表示越相似
    distances = result.get("distances", [])

    # 如果没有检索到内容，返回空列表
    if not documents:
        return []

    data = []

    # ChromaDB 返回的是二维列表
    # documents[0] 才是真正的 chunk 列表
    for index, document in enumerate(documents[0]):
        distance = None

        # 如果返回了 distance，就取对应位置的距离值
        if distances and distances[0]:
            distance = distances[0][index]

        data.append({
            "content": document,
            "distance": distance
        })

    return data