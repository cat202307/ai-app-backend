import os

# 设置离线模式，避免测试时连接 HuggingFace
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database import Base, get_db
from main import app


# ============================================================
# 测试数据库
# ============================================================

TEST_DATABASE_URL = "sqlite:///./test.db"

test_engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
)

TestingSessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=test_engine,
)


def override_get_db():
    """
    覆盖 FastAPI 的数据库依赖，
    使用测试数据库。
    """
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db


@pytest.fixture(scope="function")
def test_db():
    """
    每个测试函数创建全新的数据库表，
    测试结束后删除。
    """
    Base.metadata.create_all(bind=test_engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=test_engine)


@pytest.fixture(scope="function")
def client(test_db):
    """
    测试客户端。
    """
    return TestClient(app)


# ============================================================
# 模型相关 fixture
# ============================================================

@pytest.fixture(scope="session")
def embedding_model():
    """
    加载 embedding 模型，整个测试会话只加载一次。
    """
    from services.vector_service import embedding_model
    return embedding_model
