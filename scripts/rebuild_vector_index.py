# -*- coding: utf-8 -*-
"""
重建向量索引（父子块架构）。

适用场景：
    - 更换 Embedding 模型（向量维度变化）后，旧 collection 与新模型不兼容；
    - 切分策略升级（如本次固定块 -> 父子块），需要整体重建。

做法：
    删除旧 collection -> 重新创建空 collection ->
    对数据库中全部文档重新做父子块切分并索引（只存子块）。

运行（在项目根）：
    python scripts/rebuild_vector_index.py
"""
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# 项目根（scripts 的上一级）
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
sys.path.insert(0, BASE_DIR)

from core.config import settings  # noqa: E402
from database import SessionLocal  # noqa: E402
from models import UploadedFile  # noqa: E402
from services import vector_service  # noqa: E402
from services.chunking_service import (  # noqa: E402
    build_parent_child_chunks,
)
from services.rag_service import (  # noqa: E402
    read_document_file,
)


def main():
    # 1. 删除旧 collection（旧维度 / 旧切分，整体不兼容）
    try:
        vector_service.chroma_client.delete_collection(
            vector_service.COLLECTION_NAME
        )
        print("[1/3] old collection deleted")
    except Exception as exc:
        print("[1/3] delete skipped:", exc)

    # 2. 重新创建空 collection，并让模块全局 collection 指向它
    vector_service.collection = (
        vector_service.chroma_client.get_or_create_collection(
            name=vector_service.COLLECTION_NAME
        )
    )
    print("[2/3] new collection created")

    # 3. 对所有文件做父子块切分并建立索引
    db = SessionLocal()
    total_children = 0
    try:
        files = db.query(UploadedFile).all()
        print("[3/3] files to index:", len(files))
        for f in files:
            if not os.path.exists(f.filepath):
                print("  SKIP (file missing):", f.filename)
                continue

            text = read_document_file(f.filepath, f.file_type)
            if not text.strip():
                print("  SKIP (empty content):", f.filename)
                continue

            chunk_result = build_parent_child_chunks(
                text,
                file_id=f.id,
                child_max=settings.rag_child_chunk_size,
                children_per_parent=settings.rag_children_per_parent,
            )
            count = vector_service.build_vector_index(
                f.id,
                f.filename,
                chunk_result["children"],
            )
            total_children += count
            print(
                "  indexed %s -> %d children / %d parents"
                % (
                    f.filename,
                    count,
                    len(chunk_result["parents"]),
                )
            )
    finally:
        db.close()

    print("DONE, total children:", total_children)


if __name__ == "__main__":
    main()
