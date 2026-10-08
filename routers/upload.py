import os

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from sqlalchemy.orm import Session

from core.logging_config import get_logger
from core.config import settings
from database import get_db
from models import UploadedFile

logger = get_logger(__name__)


router = APIRouter(
    prefix="/upload",
    tags=["upload"]
)


UPLOAD_DIR = "uploads"
ALLOWED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}

os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("")
async def upload_file(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="只支持 txt、md、pdf、docx 文件"
        )

    content = await file.read()

    save_path = os.path.join(UPLOAD_DIR, filename)

    with open(save_path, "wb") as f:
        f.write(content)

    uploaded_file = UploadedFile(
        filename=filename,
        filepath=save_path,
        file_type=ext
    )

    db.add(uploaded_file)
    db.commit()
    db.refresh(uploaded_file)

    # 上传成功后自动建立向量索引，
    # 这样上传后即可检索问答，无需手动重建。
    indexed_chunks = 0
    index_error = None
    try:
        from services.vector_service import build_vector_index
        from services.rag_service import read_document_file
        from services.chunking_service import (
            build_parent_child_chunks,
        )

        file_text = read_document_file(save_path, ext)
        chunk_result = build_parent_child_chunks(
            file_text,
            file_id=uploaded_file.id,
            child_max=settings.rag_child_chunk_size,
            children_per_parent=settings.rag_children_per_parent,
        )
        indexed_chunks = build_vector_index(
            file_id=uploaded_file.id,
            filename=filename,
            children=chunk_result["children"],
        )
    except Exception as exc:
        # 索引失败不影响文件上传，但必须记录日志并告知用户，
        # 可通过 /upload/rebuild-index 手动重建。
        logger.exception(
            f"文件 {filename} (id={uploaded_file.id}) 向量索引建立失败"
        )
        indexed_chunks = 0
        index_error = str(exc)

    return {
        "code": 200,
        "message": "upload success",
        "data": {
            "id": uploaded_file.id,
            "filename": uploaded_file.filename,
            "file_type": uploaded_file.file_type,
            "size": len(content),
            "path": uploaded_file.filepath,
            "indexed_chunks": indexed_chunks,
            "index_status": "success" if indexed_chunks > 0 else "failed",
            "index_error": index_error,
            "created_at": uploaded_file.created_at
        }
    }

@router.get("/files")
def get_uploaded_files(db: Session = Depends(get_db)):
    records = (
        db.query(UploadedFile)
        .order_by(UploadedFile.created_at.desc())
        .all()
    )

    data = []

    for item in records:
        data.append({
            "id": item.id,
            "filename": item.filename,
            "filepath": item.filepath,
            "file_type": item.file_type,
            "created_at": item.created_at
        })

    return {
        "code": 200,
        "message": "success",
        "data": data
    }


@router.delete("/{file_id}")
def delete_file(
    file_id: int,
    db: Session = Depends(get_db)
):
    """
    删除指定文件。
    """
    file_record = (
        db.query(UploadedFile)
        .filter(UploadedFile.id == file_id)
        .first()
    )

    if not file_record:
        raise HTTPException(
            status_code=404,
            detail="文件不存在"
        )

    # 删除文件
    try:
        if os.path.exists(file_record.filepath):
            os.remove(file_record.filepath)
    except Exception:
        logger.warning(
            f"删除文件物理文件失败: {file_record.filepath}",
            exc_info=True,
        )

    # 删除向量索引，避免残留孤儿 chunk 影响全局检索
    try:
        from services.vector_service import (
            delete_vector_index,
        )
        delete_vector_index(file_id)
    except Exception:
        logger.warning(
            f"删除文件向量索引失败: file_id={file_id}",
            exc_info=True,
        )

    # 删除数据库记录
    db.delete(file_record)
    db.commit()

    return {
        "code": 200,
        "message": "deleted",
        "data": {
            "id": file_id,
        }
    }


@router.post("/rebuild-index")
async def rebuild_index(
    db: Session = Depends(get_db)
):
    """重建所有文件的向量索引"""
    import os
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    
    from services.vector_service import build_vector_index
    from services.rag_service import read_document_file
    from services.chunking_service import build_parent_child_chunks
    
    files = db.query(UploadedFile).all()
    results = []
    
    for file in files:
        # 检查文件是否存在
        if not os.path.exists(file.filepath):
            results.append({
                "filename": file.filename,
                "status": "failed",
                "message": "文件不存在"
            })
            continue

        # 读取文件内容
        try:
            text = read_document_file(file.filepath, file.file_type)
            if not text.strip():
                results.append({
                    "filename": file.filename,
                    "status": "failed",
                    "message": "文件内容为空"
                })
                continue
        except Exception as e:
            results.append({
                "filename": file.filename,
                "status": "failed",
                "message": f"读取失败: {str(e)}"
            })
            continue

        # 父子块切分
        chunk_result = build_parent_child_chunks(
            text,
            file_id=file.id,
            child_max=settings.rag_child_chunk_size,
            children_per_parent=settings.rag_children_per_parent,
        )

        # 建立索引（只存子块）
        try:
            chunk_count = build_vector_index(
                file_id=file.id,
                filename=file.filename,
                children=chunk_result["children"],
            )
            results.append({
                "filename": file.filename,
                "status": "success",
                "chunks": chunk_count
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "status": "failed",
                "message": f"索引建立失败: {str(e)}"
            })
    
    return {
        "code": 200,
        "message": "rebuild completed",
        "data": {
            "total": len(files),
            "results": results
        }
    }
