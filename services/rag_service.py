import os
import re
from pypdf import PdfReader
from docx import Document


def read_text_file(filepath: str) -> str:
    if not os.path.exists(filepath):
        return ""

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(filepath, "r", encoding="gbk", errors="ignore") as f:
            return f.read()


def read_pdf_file(filepath: str) -> str:
    if not os.path.exists(filepath):
        return ""

    text_list = []

    reader = PdfReader(filepath)

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text_list.append(page_text)

    return "\n".join(text_list)


def read_docx_file(filepath: str) -> str:
    if not os.path.exists(filepath):
        return ""

    doc = Document(filepath)

    text_list = []

    for paragraph in doc.paragraphs:
        if paragraph.text.strip():
            text_list.append(paragraph.text)

    return "\n".join(text_list)


def read_document_file(filepath: str, file_type: str) -> str:
    text = ""

    if file_type in [".txt", ".md"]:
        text = read_text_file(filepath)

    elif file_type == ".pdf":
        text = read_pdf_file(filepath)

    elif file_type == ".docx":
        text = read_docx_file(filepath)

    return clean_text(text)

def split_text(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 100
) -> list[str]:
    """
    文本切分函数。

    参数：
        text：完整文档内容
        chunk_size：每个文本片段的最大长度
        chunk_overlap：相邻片段之间重叠的字符数

    为什么要加 chunk_overlap？
        防止一句话或一段内容刚好被切断，导致语义不完整。
    """

    # 如果文本为空，直接返回空列表
    if not text.strip():
        return []

    # 防止 chunk_overlap 设置得比 chunk_size 还大
    if chunk_overlap >= chunk_size:
        chunk_overlap = 0

    chunks = []

    # 当前切分开始位置
    start = 0

    # 文本总长度
    text_length = len(text)

    while start < text_length:
        # 当前 chunk 的结束位置
        end = start + chunk_size

        # 截取当前 chunk
        chunk = text[start:end]

        # 去掉首尾空格后，如果不是空内容，就加入列表
        if chunk.strip():
            chunks.append(chunk.strip())

        # 如果已经切到文档末尾，就结束循环
        if end >= text_length:
            break

        # 下一段开始位置
        # 回退 chunk_overlap 个字符，形成重叠区域
        start = end - chunk_overlap

    return chunks


def simple_search(question: str, chunks: list[str], top_k: int = 3) -> list[str]:
    question_keywords = list(set(question))

    scored_chunks = []

    for chunk in chunks:
        score = 0

        for keyword in question_keywords:
            if keyword.strip() and keyword in chunk:
                score += 1

        scored_chunks.append((score, chunk))

    scored_chunks.sort(key=lambda x: x[0], reverse=True)

    result = []

    for score, chunk in scored_chunks[:top_k]:
        if score > 0:
            result.append(chunk)

    if not result and chunks:
        result = chunks[:top_k]

    return result

def clean_text(text: str) -> str:
    # 去掉过多空白行
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 把多个空格合并成一个空格
    text = re.sub(r"[ \t]+", " ", text)

    # 去掉每一行首尾空格
    lines = text.splitlines()
    lines = [line.strip() for line in lines if line.strip()]

    text = "\n".join(lines)

    return text