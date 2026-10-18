import asyncio
import os
import uuid
from pathlib import Path

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


UPLOAD_DIR = Path("uploads")
ALLOWED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
# 解析一次上传根目录，用于校验落盘路径不会逃逸到目录外
_UPLOAD_ROOT = UPLOAD_DIR.resolve()


def sanitize_filename(filename: str | None) -> str:
    """
    只保留原始文件名的最后一段，去掉任何目录分隔符，
    防止形如 ../../etc/passwd 的路径穿越。
    """
    if not filename:
        return ""
    base = filename.replace("\\", "/").split("/")[-1]
    return base.strip()


def safe_storage_path(ext: str) -> Path:
    """
    生成随机落盘路径（uuid + 扩展名）。

    - 避免不同用户上传同名文件相互覆盖
    - 解析为绝对路径后，确保其仍位于上传目录内
    """
    storage_path = (
        _UPLOAD_ROOT / f"{uuid.uuid4().hex}{ext}"
    ).resolve()

    if _UPLOAD_ROOT not in storage_path.parents:
        raise ValueError("非法的文件路径")

    return storage_path


def resolve_storage_path(stored: str | None) -> Path | None:
    """
    把数据库里保存的 filepath 解析为当前环境真实存在的物理文件。

    兼容三种历史/跨环境情况：
      1. 本机绝对路径（新上传记录，容器内为 /app/uploads/xxx，
         Windows 上为 D:\\...\\uploads\\xxx）；
      2. 相对路径且带 Windows 反斜杠（早期记录，如 uploads\\a.txt），
         在 Linux 容器内反斜杠不是分隔符；
      3. 路径在别的机器上生成、目录不一致 —— 最后按文件名
         在上传目录内回退查找。

    找不到时返回 None。
    """
    if not stored:
        return None

    normalized = Path(stored.replace("\\", "/"))
    candidates = [
        Path(stored),            # 原始路径（绝对/相对当前工作目录）
        normalized,              # 规范化分隔符后的相对路径
        _UPLOAD_ROOT / normalized.name,  # 上传目录内按文件名回退
    ]

    for candidate in candidates:
        try:
            if candidate.is_file():
                return candidate
        except OSError:
            continue
    return None


def build_index_for_file(
    file_id: int,
    filename: str,
    save_path: Path,
    ext: str,
) -> int:
    """
    读取文档并建立向量索引。

    该函数包含文档解析、切分与 embedding（CPU/IO 密集），
    通过 run_in_executor 放到线程池执行，避免阻塞事件循环。
    """
    from services.vector_service import build_vector_index
    from services.rag_service import read_document_file
    from services.chunking_service import (
        build_parent_child_chunks,
    )

    file_text = read_document_file(str(save_path), ext)
    chunk_result = build_parent_child_chunks(
        file_text,
        file_id=file_id,
        child_max=settings.rag_child_chunk_size,
        children_per_parent=settings.rag_children_per_parent,
    )
    return build_vector_index(
        file_id=file_id,
        filename=filename,
        children=chunk_result["children"],
    )


@router.post("")
async def upload_file(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    # 1. 文件名清洗 + 扩展名校验
    original_name = sanitize_filename(file.filename)
    ext = os.path.splitext(original_name)[1].lower()

    if not original_name or ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="只支持 txt、md、pdf、docx 文件",
        )

    # 2. 读取并限制大小（上传内容会读入内存，防止超大文件 OOM）
    content = await file.read()

    max_bytes = settings.upload_max_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=(
                f"文件过大，单文件不得超过 "
                f"{settings.upload_max_mb}MB"
            ),
        )

    # 3. 随机文件名安全落盘（防路径穿越 / 同名覆盖）
    storage_path = safe_storage_path(ext)
    with open(storage_path, "wb") as f:
        f.write(content)

    # 数据库里 filename 保留用户的原始文件名，
    # filepath 保存随机落盘路径。
    uploaded_file = UploadedFile(
        filename=original_name,
        filepath=str(storage_path),
        file_type=ext
    )

    db.add(uploaded_file)
    db.commit()
    db.refresh(uploaded_file)

    # 4. 上传成功后在线程池里自动建立向量索引，
    # 不阻塞事件循环；索引失败不阻断上传，
    # 可通过 /upload/rebuild-index 手动重建。
    indexed_chunks = 0
    index_error = None
    try:
        loop = asyncio.get_running_loop()
        indexed_chunks = await loop.run_in_executor(
            None,
            build_index_for_file,
            uploaded_file.id,
            original_name,
            storage_path,
            ext,
        )
    except Exception as exc:
        logger.exception(
            f"文件 {original_name} (id={uploaded_file.id}) 向量索引建立失败"
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
        # 文件大小不入库，按磁盘实际大小返回；
        # 物理文件缺失（如手动删除）时为 None，前端显示为 "-"。
        # 经跨平台路径解析，兼容容器内的反斜杠相对路径。
        file_size = None
        try:
            physical = resolve_storage_path(item.filepath)
            if physical is not None:
                file_size = physical.stat().st_size
        except OSError:
            file_size = None

        data.append({
            "id": item.id,
            "filename": item.filename,
            "filepath": item.filepath,
            "file_type": item.file_type,
            "file_size": file_size,
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

    # 删除物理文件（经跨平台路径解析，兼容容器内旧记录）
    try:
        physical = resolve_storage_path(file_record.filepath)
        if physical is not None:
            physical.unlink()
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
def rebuild_index(
    db: Session = Depends(get_db)
):
    """
    重建所有文件的向量索引。

    定义为同步函数（而非 async），FastAPI 会把它放到
    线程池执行，避免长耗时建索引阻塞事件循环。
    """
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    from services.vector_service import build_vector_index
    from services.rag_service import read_document_file
    from services.chunking_service import build_parent_child_chunks

    files = db.query(UploadedFile).all()
    results = []

    for file in files:
        # 跨平台解析物理文件（兼容容器内反斜杠相对路径）
        physical = resolve_storage_path(file.filepath)
        if physical is None:
            results.append({
                "filename": file.filename,
                "status": "failed",
                "message": "文件不存在"
            })
            continue

        # 读取文件内容
        try:
            text = read_document_file(str(physical), file.file_type)
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
