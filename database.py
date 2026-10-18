import os

from sqlalchemy import create_engine, event, inspect
from sqlalchemy.orm import sessionmaker, declarative_base

from core.config import settings


DATABASE_URL = settings.database_url

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    pool_pre_ping=True,
)


@event.listens_for(engine, "connect")
def _set_sqlite_pragma(dbapi_connection, connection_record):
    """
    SQLite 连接级配置：
    - WAL 模式：允许读写并发，减少 "database is locked"
    - foreign_keys=ON：开启外键约束
    """
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# 初始迁移版本（initial schema）。
# 对历史上由 create_all 直接建表、却没有 alembic_version 记录的老库，
# 先把版本标记到该基线，再继续向上升级。
INITIAL_REVISION = "1513e1684fd4"


def init_db() -> None:
    """
    以 Alembic 迁移作为唯一的生产 schema 演进入口，在应用启动时执行。

    - 新库（无任何表）：直接 upgrade 到 head，由初始迁移建表；
    - 已纳入版本管理的库：upgrade 到 head（增量应用后续迁移）；
    - 历史遗留库（表已存在但无 alembic_version）：先 stamp 到初始版本，
      再 upgrade，避免重复建表报错。

    注意：测试库（tests/conftest.py）仍用 create_all 建表，不经过这里。
    """
    from alembic import command
    from alembic.config import Config

    project_root = os.path.dirname(os.path.abspath(__file__))
    ini_path = os.path.join(project_root, "alembic.ini")

    cfg = Config(ini_path)
    cfg.set_main_option("sqlalchemy.url", settings.database_url)

    existing_tables = set(inspect(engine).get_table_names())
    known_app_tables = {
        "chat_history",
        "uploaded_files",
        "rag_query_logs",
    }
    if "alembic_version" not in existing_tables and (
        known_app_tables & existing_tables
    ):
        # 表已存在但未纳入版本管理：标记到初始基线
        command.stamp(cfg, INITIAL_REVISION)

    command.upgrade(cfg, "head")
