# 基于 FastAPI 和大模型的 AI 知识库问答系统

## 一、项目简介

本项目是一个基于 FastAPI 和大模型 API 实现的 AI 知识库问答系统，支持用户与大模型进行多轮对话，并支持上传 PDF、Word、TXT、Markdown 等文档进行知识库问答。

系统通过 RAG 检索增强生成技术，实现文档解析、文本清洗、文本切分、Embedding 向量化、ChromaDB 向量检索和基于文档内容的大模型回答生成。同时，系统记录每次 RAG 问答的检索结果、distance 分数和命中状态，用于分析检索质量并减少大模型幻觉。

---

## 二、技术栈

### 后端框架

- Python
- FastAPI
- Uvicorn
- Pydantic

### 数据库

- SQLite
- SQLAlchemy

### 大模型

- DeepSeek API
- OpenAI SDK 兼容调用方式

### RAG 相关

- sentence-transformers
- ChromaDB
- Embedding 向量化
- 向量相似度检索

### 文档解析

- PyMuPDF
- python-docx
- TXT / Markdown 文件读取

---

## 三、项目功能

### 1. 大模型聊天

- 支持普通大模型问答
- 支持 session_id 多轮对话
- 支持聊天记录保存
- 支持按 session_id 查询历史记录

### 2. 文件上传

- 支持上传 txt、md、pdf、docx 文件
- 文件保存到 uploads 目录
- 文件信息保存到 SQLite 数据库

### 3. RAG 知识库问答

- 支持读取上传文档
- 支持 PDF / Word / TXT / Markdown 文档解析
- 支持文本清洗
- 支持 chunk 文本切分
- 支持 Embedding 向量化
- 支持 ChromaDB 向量检索
- 支持基于文档内容的大模型问答

### 4. RAG 检索质量分析

- 返回 related_chunks 检索片段
- 返回 rank 排序
- 返回 distance 距离分数
- 返回 retrieval_status 检索状态
- 支持 MAX_DISTANCE 阈值判断
- 支持 RAG 问答日志记录
- 支持 RAG 日志统计接口

---

## 四、项目目录结构

```text
ai-app-backend/
├── main.py
├── database.py
├── models.py
├── schemas.py
├── requirements.txt
├── .env
├── .gitignore
│
├── routers/
│   ├── __init__.py
│   ├── chat.py
│   ├── history.py
│   ├── upload.py
│   └── rag.py
│
├── services/
│   ├── __init__.py
│   ├── llm_service.py
│   ├── rag_service.py
│   └── vector_service.py
│
├── uploads/
├── vector_db/
└── ai_app.db