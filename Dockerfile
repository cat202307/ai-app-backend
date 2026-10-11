# ============================================================
# Stage 1: 构建 Vue 3 前端
# ============================================================
FROM node:22-slim AS frontend-builder

WORKDIR /fe

# 先复制依赖描述，充分利用 Docker 层缓存
COPY frontend/package*.json ./
RUN npm config set registry https://registry.npmmirror.com \
    && npm install

# 复制前端源码并构建（产物输出到 /fe/dist）
COPY frontend/ ./
RUN npm run build

# ============================================================
# Stage 2: Python 后端 + 前端构建产物
# ============================================================
FROM python:3.10-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=Asia/Shanghai \
    HF_HOME=/root/.cache/huggingface \
    HF_HUB_OFFLINE=0 \
    TRANSFORMERS_OFFLINE=0

# 编译相关系统依赖；tzdata 用于容器内北京时间
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    tzdata \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖（USTC 镜像加速）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    -i https://pypi.mirrors.ustc.edu.cn/simple

# 复制后端代码（.dockerignore 已排除 node_modules / dist / 数据目录）
COPY . .

# 从前端构建阶段复制干净的构建产物，实现单容器单端口
COPY --from=frontend-builder /fe/dist ./frontend/dist

# 数据目录
RUN mkdir -p /app/uploads /app/vector_db

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
