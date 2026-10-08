"""批量重建所有文件的向量索引（父子块架构，不删除 collection）。"""

import os
import sys

# 项目根（scripts 的上一级）
sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from core.config import settings
from database import SessionLocal
from models import UploadedFile
from services.vector_service import build_vector_index
from services.chunking_service import build_parent_child_chunks
from services.rag_service import read_document_file

db = SessionLocal()

files = db.query(UploadedFile).all()
print(f"共 {len(files)} 个文件需要重建索引")

for file in files:
    print(f"\n处理: {file.filename} (id={file.id})")

    # 检查文件是否存在
    if not os.path.exists(file.filepath):
        print(f"  [FAIL] 文件不存在: {file.filepath}")
        continue

    # 读取文件内容
    try:
        text = read_document_file(file.filepath, file.file_type)
        if not text.strip():
            print("  [FAIL] 文件内容为空")
            continue
    except Exception as e:
        print(f"  [FAIL] 读取失败: {e}")
        continue

    # 父子块切分
    chunk_result = build_parent_child_chunks(
        text,
        file_id=file.id,
        child_max=settings.rag_child_chunk_size,
        children_per_parent=settings.rag_children_per_parent,
    )
    print(
        f"  切分为 {len(chunk_result['children'])} 个子块 / "
        f"{len(chunk_result['parents'])} 个父块"
    )

    # 建立索引（build_vector_index 会先删除该 file_id 旧索引）
    try:
        chunk_count = build_vector_index(
            file_id=file.id,
            filename=file.filename,
            children=chunk_result["children"],
        )
        print(f"  [OK] 索引建立成功: {chunk_count} 个子块")
    except Exception as e:
        print(f"  [FAIL] 索引建立失败: {e}")

db.close()
print("\n全部完成")
