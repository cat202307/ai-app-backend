import os

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from sqlalchemy.orm import Session

from database import get_db
from models import UploadedFile


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

    return {
        "code": 200,
        "message": "upload success",
        "data": {
            "id": uploaded_file.id,
            "filename": uploaded_file.filename,
            "file_type": uploaded_file.file_type,
            "size": len(content),
            "path": uploaded_file.filepath,
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