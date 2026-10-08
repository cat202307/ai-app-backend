# Docker 部署指南

本项目采用 **多阶段构建**：镜像内自动完成 Vue 3 前端构建并整合进 FastAPI，
最终只需 **一个容器、一个端口（8000）** 即可访问完整应用，无需在宿主机安装 Node.js。

## 前置要求

- Docker 20.10+
- Docker Compose 2.0+

## 快速启动

### 1. 配置环境变量

复制 `.env.example` 为 `.env`，填入 DeepSeek API Key：

```bash
cp .env.example .env   # Windows: copy .env.example .env
```

编辑 `.env`：

```env
DEEPSEEK_API_KEY=your_api_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-pro
```

### 2.（重要）首次启动前初始化数据库文件

docker-compose 将宿主机的 `./ai_app.db` 以**单文件**方式挂载进容器。
如果该文件不存在，Docker 会错误地把它创建成一个**目录**，导致 SQLite 启动失败。
首次部署请先创建空文件：

```powershell
# Windows PowerShell
if (-not (Test-Path ai_app.db)) { New-Item -ItemType File ai_app.db }
```
```bash
# Linux / macOS
[ -f ai_app.db ] || touch ai_app.db
```

> 也可以直接注释 docker-compose.yml 中 `./ai_app.db:/app/ai_app.db` 这一行，
> 让容器在内部自建数据库（但容器删除后数据不保留）。

### 3. 构建并启动

```bash
docker-compose up -d --build
```

首次构建会下载 Node / Python 镜像、安装依赖并构建前端，需要几分钟；之后有层缓存会很快。

### 4. 查看日志

```bash
docker-compose logs -f
```

### 5. 访问服务（单端口）

- **应用首页（Vue SPA）**：http://localhost:8000/
- **Swagger 文档**：http://localhost:8000/docs
- **ReDoc 文档**：http://localhost:8000/redoc
- **健康检查**：http://localhost:8000/health

## 数据持久化

| 容器路径 | 宿主机路径 | 说明 |
|---------|-----------|------|
| `/app/ai_app.db` | `./ai_app.db` | SQLite 数据库（需为文件，见步骤 2） |
| `/app/vector_db/` | `./vector_db/` | ChromaDB 向量索引 |
| `/app/uploads/` | `./uploads/` | 上传的原始文件 |

## 常用命令

```bash
docker-compose down              # 停止
docker-compose restart           # 重启
docker-compose up -d --build     # 代码变更后重新构建启动
docker exec -it ai-app-backend bash   # 进入容器
```

## 注意事项

1. **首次启动较慢**：第一次会下载 Embedding 模型（约 100MB），需要一些时间
2. **内存占用**：Embedding 模型约占 500MB 内存
3. **前端构建已内置**：多阶段构建会在镜像内执行 `npm run build`，无需手动构建
4. **健康检查**：使用 Python 内置 urllib（slim 镜像不含 curl），通过 `docker ps` 可查看健康状态
5. **GPU**：当前为 CPU 版本，如需 GPU 加速请改用 CUDA 基础镜像

## 生产环境建议

1. 前置反向代理（Nginx / 网关）并启用 HTTPS
2. 配置日志轮转与监控告警
3. 使用非 root 用户运行
4. 固定镜像版本标签，避免直接使用 `latest`
5. 设置强随机 `API_KEY` 启用 X-API-Key 认证（默认留空 = 不认证，仅适合本地）
6. 若有跨域访问，在 `CORS_ORIGINS` 显式限定可信来源，不要保留默认 localhost
7. 按所用模型的真实单价核对 `LLM_PRICE_PER_MILLION_INPUT/OUTPUT`（仅影响 `/metrics` 成本估算）
