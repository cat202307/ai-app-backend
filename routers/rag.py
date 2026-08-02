import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from database import get_db
from models import UploadedFile, RagQueryLog
from schemas import RagAskRequest
from services.llm_service import call_llm, stream_llm
from services.rag_service import (
    read_document_file,
    split_text,
    simple_search,
    clean_text
)
from services.vector_service import (
    build_vector_index,
    search_similar_chunks
)


router = APIRouter(
    prefix="/rag",
    tags=["rag"]
)


@router.post("/ask")
def rag_ask(req: RagAskRequest, db: Session = Depends(get_db)):
    """
    简单版 RAG 问答接口。

    这个接口不使用向量数据库，
    只是通过简单关键词匹配查找相关文档片段。

    流程：
    1. 根据 file_id 查询上传文件
    2. 读取文档内容
    3. 文本清洗
    4. 文本切分
    5. 简单关键词检索
    6. 拼接上下文
    7. 调用大模型回答
    """

    # 根据 file_id 查询上传文件
    uploaded_file = (
        db.query(UploadedFile)
        .filter(UploadedFile.id == req.file_id)
        .first()
    )

    # 如果文件不存在，返回 404
    if not uploaded_file:
        raise HTTPException(
            status_code=404,
            detail="文件不存在"
        )

    # 判断文件类型是否支持
    if uploaded_file.file_type not in [".txt", ".md", ".pdf", ".docx"]:
        raise HTTPException(
            status_code=400,
            detail="当前 RAG 支持 txt、md、pdf、docx 文件"
        )

    # 根据文件类型读取文档内容
    text = read_document_file(
        uploaded_file.filepath,
        uploaded_file.file_type
    )

    # 如果内容为空，说明读取失败或文件本身为空
    if not text.strip():
        raise HTTPException(
            status_code=400,
            detail="文件内容为空或读取失败"
        )

    # 将长文本切分成多个 chunk
    chunks = split_text(text)

    # 简单关键词检索相关片段
    related_chunks = simple_search(
        question=req.question,
        chunks=chunks,
        top_k=3
    )

    # 把检索到的片段拼接成上下文
    context = "\n\n".join(related_chunks)

    # 构造大模型 messages
    messages = [
        {
            "role": "system",
            "content": (
                "你是一个知识库问答助手。"
                "请严格根据给定的文档内容回答问题。"
                "如果文档中没有相关信息，请回答：文档中没有找到相关内容。"
            )
        },
        {
            "role": "user",
            "content": f"文档内容如下：\n{context}\n\n用户问题：{req.question}"
        }
    ]

    # 调用大模型
    answer = call_llm(messages)

    return {
        "code": 200,
        "message": "success",
        "data": {
            "file_id": uploaded_file.id,
            "filename": uploaded_file.filename,
            "question": req.question,
            "answer": answer,
            "related_chunks": [
                clean_text(chunk)[:500]
                for chunk in related_chunks
            ]
        }
    }


@router.post("/index/{file_id}")
def rag_index(file_id: int, db: Session = Depends(get_db)):
    """
    建立向量索引接口。

    这个接口用于把文档内容写入 ChromaDB 向量数据库。

    流程：
    1. 根据 file_id 查询文件
    2. 读取 PDF / Word / TXT / Markdown 内容
    3. 清洗文本
    4. 切分 chunk
    5. 使用 Embedding 模型向量化
    6. 写入 ChromaDB
    """

    # 根据 file_id 查询上传文件
    uploaded_file = (
        db.query(UploadedFile)
        .filter(UploadedFile.id == file_id)
        .first()
    )

    # 文件不存在
    if not uploaded_file:
        raise HTTPException(
            status_code=404,
            detail="文件不存在"
        )

    # 判断文件类型是否支持
    if uploaded_file.file_type not in [".txt", ".md", ".pdf", ".docx"]:
        raise HTTPException(
            status_code=400,
            detail="当前向量索引支持 txt、md、pdf、docx 文件"
        )

    # 读取文档内容
    text = read_document_file(
        uploaded_file.filepath,
        uploaded_file.file_type
    )

    # 判断文档内容是否为空
    if not text.strip():
        raise HTTPException(
            status_code=400,
            detail="文件内容为空或读取失败"
        )

    # 文本切分
    # chunk_size：每个片段最大长度
    # chunk_overlap：相邻片段重叠长度，避免语义被切断
    chunks = split_text(
        text,
        chunk_size=800,
        chunk_overlap=100
    )

    # 建立向量索引
    chunk_count = build_vector_index(
        file_id=uploaded_file.id,
        filename=uploaded_file.filename,
        chunks=chunks
    )

    return {
        "code": 200,
        "message": "index success",
        "data": {
            "file_id": uploaded_file.id,
            "filename": uploaded_file.filename,
            "chunk_count": chunk_count
        }
    }


@router.post("/search")
def rag_search(req: RagAskRequest, db: Session = Depends(get_db)):
    """
    向量检索版 RAG 问答接口。

    流程：
    1. 根据 file_id 查询文件
    2. 从向量数据库检索相关 chunk
    3. 根据 distance 判断相关性
    4. 拼接 Prompt
    5. 调用大模型回答
    6. 返回答案、相关片段和距离分数
    """

    uploaded_file = (
        db.query(UploadedFile)
        .filter(UploadedFile.id == req.file_id)
        .first()
    )

    if not uploaded_file:
        raise HTTPException(
            status_code=404,
            detail="文件不存在"
        )

    related_items = search_similar_chunks(
        question=req.question,
        file_id=req.file_id,
        top_k=3
    )

    if not related_items:
        raise HTTPException(
            status_code=400,
            detail="没有检索到相关内容，请先调用 /rag/index/{file_id} 建立索引"
        )

    # 设置相似度距离阈值
    # distance 越小，表示越相似
    # 这个值需要根据实际测试调整
    MAX_DISTANCE = 17.0

    # 取第一个结果，也就是最相似的 chunk
    best_distance = related_items[0].get("distance")

    retrieval_status = "matched"

    # 如果最相似的结果距离都很大，说明文档里可能没有相关内容
    if best_distance is not None and best_distance > MAX_DISTANCE:
        answer = "文档中没有找到相关内容。"

        rag_log = RagQueryLog(
            file_id=uploaded_file.id,
            filename=uploaded_file.filename,
            question=req.question,
            answer=answer,
            best_distance=best_distance,
            max_distance=MAX_DISTANCE,
            retrieval_status="not_matched"
        )

        db.add(rag_log)
        db.commit()

        return {
            "code": 200,
            "message": "success",
            "data": {
                "file_id": uploaded_file.id,
                "filename": uploaded_file.filename,
                "question": req.question,
                "answer": answer,
                "retrieval_info": {
                    "best_distance": best_distance,
                    "max_distance": MAX_DISTANCE,
                    "retrieval_status": "not_matched"
                },
                "related_chunks": [
                    {
                        "rank": index + 1,
                        "content": clean_text(item["content"])[:500],
                        "distance": item["distance"]
                    }
                    for index, item in enumerate(related_items)
                ]
            }
        }

    related_chunks = [
        item["content"] for item in related_items
    ]

    context = "\n\n".join(related_chunks)

    messages = [
        {
            "role": "system",
            "content": (
                "你是一个严谨的知识库问答助手。"
                "你只能根据用户提供的文档片段回答问题。"
                "不能使用文档片段之外的知识进行补充。"
                "如果文档片段中没有明确提到相关信息，请回答：文档中没有找到相关内容。"
                "回答时要简洁、准确，不要编造不存在的信息。"
            )
        },
        {
            "role": "user",
            "content": f"文档内容如下：\n{context}\n\n用户问题：{req.question}"
        }
    ]

    answer = call_llm(messages)

    rag_log = RagQueryLog(
        file_id=uploaded_file.id,
        filename=uploaded_file.filename,
        question=req.question,
        answer=answer,
        best_distance=best_distance,
        max_distance=MAX_DISTANCE,
        retrieval_status=retrieval_status
    )

    db.add(rag_log)
    db.commit()

    return {
        "code": 200,
        "message": "success",
        "data": {
            "file_id": uploaded_file.id,
            "filename": uploaded_file.filename,
            "question": req.question,
            "answer": answer,
            "retrieval_info": {
                "best_distance": best_distance,
                "max_distance": MAX_DISTANCE,
                "retrieval_status": retrieval_status
            },
            "related_chunks": [
                {
                    "rank": index + 1,
                    "content": clean_text(item["content"])[:500],
                    "distance": item["distance"]
                }
                for index, item in enumerate(related_items)
            ]
        }
    }

@router.get("/logs")
def get_rag_logs(
    page: int = 1,
    size: int = 10,
    db: Session = Depends(get_db)
):
    """
    查询 RAG 问答日志。

    用来查看每次 RAG 问答的效果，
    包括问题、回答、best_distance、retrieval_status 等信息。
    """

    offset = (page - 1) * size

    records = (
        db.query(RagQueryLog)
        .order_by(RagQueryLog.created_at.desc())
        .offset(offset)
        .limit(size)
        .all()
    )

    data = []

    for item in records:
        data.append({
            "id": item.id,
            "file_id": item.file_id,
            "filename": item.filename,
            "question": item.question,
            "answer": item.answer,
            "best_distance": item.best_distance,
            "max_distance": item.max_distance,
            "retrieval_status": item.retrieval_status,
            "created_at": item.created_at
        })

    return {
        "code": 200,
        "message": "success",
        "data": data
    }

@router.get("/logs/stats")
def get_rag_log_stats(db: Session = Depends(get_db)):
    """
    RAG 问答日志统计接口。

    用于统计：
    1. 总问答次数
    2. matched 数量
    3. not_matched 数量
    4. matched 占比
    5. not_matched 占比
    6. 平均 best_distance
    """

    records = db.query(RagQueryLog).all()

    total_count = len(records)

    if total_count == 0:
        return {
            "code": 200,
            "message": "success",
            "data": {
                "total_count": 0,
                "matched_count": 0,
                "not_matched_count": 0,
                "matched_rate": 0,
                "not_matched_rate": 0,
                "avg_best_distance": None
            }
        }

    matched_count = 0
    not_matched_count = 0
    distance_sum = 0
    distance_count = 0

    for item in records:
        if item.retrieval_status == "matched":
            matched_count += 1

        if item.retrieval_status == "not_matched":
            not_matched_count += 1

        if item.best_distance is not None:
            distance_sum += item.best_distance
            distance_count += 1

    matched_rate = matched_count / total_count
    not_matched_rate = not_matched_count / total_count

    avg_best_distance = None

    if distance_count > 0:
        avg_best_distance = distance_sum / distance_count

    return {
        "code": 200,
        "message": "success",
        "data": {
            "total_count": total_count,
            "matched_count": matched_count,
            "not_matched_count": not_matched_count,
            "matched_rate": matched_rate,
            "not_matched_rate": not_matched_rate,
            "avg_best_distance": avg_best_distance
        }
    }

def format_sse(data: dict) -> str:
    """
    把 Python 字典转换成 SSE 格式。

    SSE 格式要求：
        data: xxx\n\n
    """

    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/search/stream")
def rag_search_stream(req: RagAskRequest, db: Session = Depends(get_db)):
    """
    RAG 流式问答接口。

    流程：
    1. 根据 file_id 查询文件
    2. 从向量数据库检索相关 chunk
    3. 根据 best_distance 判断是否命中文档
    4. 如果 not_matched，直接返回“文档中没有找到相关内容”
    5. 如果 matched，调用大模型流式生成回答
    6. 保存 RAG 问答日志
    """

    uploaded_file = (
        db.query(UploadedFile)
        .filter(UploadedFile.id == req.file_id)
        .first()
    )

    if not uploaded_file:
        raise HTTPException(
            status_code=404,
            detail="文件不存在"
        )

    related_items = search_similar_chunks(
        question=req.question,
        file_id=req.file_id,
        top_k=3
    )

    if not related_items:
        raise HTTPException(
            status_code=400,
            detail="没有检索到相关内容，请先调用 /rag/index/{file_id} 建立索引"
        )

    MAX_DISTANCE = 17.0

    best_distance = related_items[0].get("distance")

    retrieval_status = "matched"

    if best_distance is not None and best_distance > MAX_DISTANCE:
        retrieval_status = "not_matched"

    related_chunks = [
        item["content"] for item in related_items
    ]

    context = "\n\n".join(related_chunks)

    retrieval_info = {
        "best_distance": best_distance,
        "max_distance": MAX_DISTANCE,
        "retrieval_status": retrieval_status
    }

    related_chunks_for_response = [
        {
            "rank": index + 1,
            "content": clean_text(item["content"])[:500],
            "distance": item["distance"]
        }
        for index, item in enumerate(related_items)
    ]

    messages = [
        {
            "role": "system",
            "content": (
                "你是一个严谨的知识库问答助手。"
                "你只能根据用户提供的文档片段回答问题。"
                "不能使用文档片段之外的知识进行补充。"
                "如果文档片段中没有明确提到相关信息，请回答：文档中没有找到相关内容。"
                "回答时要简洁、准确，不要编造不存在的信息。"
            )
        },
        {
            "role": "user",
            "content": f"文档内容如下：\n{context}\n\n用户问题：{req.question}"
        }
    ]

    def generate():
        answer_parts = []

        try:
            yield format_sse({
                "type": "meta",
                "file_id": uploaded_file.id,
                "filename": uploaded_file.filename,
                "question": req.question,
                "retrieval_info": retrieval_info
            })

            yield format_sse({
                "type": "references",
                "related_chunks": related_chunks_for_response
            })

            if retrieval_status == "not_matched":
                answer = "文档中没有找到相关内容。"

                answer_parts.append(answer)

                yield format_sse({
                    "type": "delta",
                    "content": answer
                })

                rag_log = RagQueryLog(
                    file_id=uploaded_file.id,
                    filename=uploaded_file.filename,
                    question=req.question,
                    answer=answer,
                    best_distance=best_distance,
                    max_distance=MAX_DISTANCE,
                    retrieval_status="not_matched"
                )

                db.add(rag_log)
                db.commit()

                yield format_sse({
                    "type": "done",
                    "message": "stream finished"
                })

                return

            for content in stream_llm(messages):
                answer_parts.append(content)

                yield format_sse({
                    "type": "delta",
                    "content": content
                })

            full_answer = "".join(answer_parts)

            rag_log = RagQueryLog(
                file_id=uploaded_file.id,
                filename=uploaded_file.filename,
                question=req.question,
                answer=full_answer,
                best_distance=best_distance,
                max_distance=MAX_DISTANCE,
                retrieval_status=retrieval_status
            )

            db.add(rag_log)
            db.commit()

            yield format_sse({
                "type": "done",
                "message": "stream finished"
            })

        except Exception as e:
            db.rollback()

            yield format_sse({
                "type": "error",
                "message": str(e)
            })

    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )