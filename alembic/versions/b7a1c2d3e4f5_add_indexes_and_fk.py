"""add indexes and rag log fk

Revision ID: b7a1c2d3e4f5
Revises: 1513e1684fd4
Create Date: 2026-10-18 23:10:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b7a1c2d3e4f5'
down_revision: Union[str, Sequence[str], None] = '1513e1684fd4'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """为高频查询列加索引；为 rag_query_logs.file_id 补外键。"""
    # 聊天历史：按会话加载历史、按时间排序
    op.create_index(
        'ix_chat_history_session_id', 'chat_history', ['session_id']
    )
    op.create_index(
        'ix_chat_history_created_at', 'chat_history', ['created_at']
    )

    # 上传文件列表：按时间倒序
    op.create_index(
        'ix_uploaded_files_created_at', 'uploaded_files', ['created_at']
    )

    # RAG 日志：按文件聚合、按时间排序
    op.create_index(
        'ix_rag_query_logs_file_id', 'rag_query_logs', ['file_id']
    )
    op.create_index(
        'ix_rag_query_logs_created_at', 'rag_query_logs', ['created_at']
    )

    # SQLite 不支持 ADD CONSTRAINT，需要 batch 模式重建表
    with op.batch_alter_table('rag_query_logs', schema=None) as batch:
        batch.create_foreign_key(
            'fk_rag_logs_file_id',
            'uploaded_files',
            ['file_id'],
            ['id'],
            ondelete='CASCADE',
        )


def downgrade() -> None:
    """回滚：删外键与索引。"""
    with op.batch_alter_table('rag_query_logs', schema=None) as batch:
        batch.drop_constraint(
            'fk_rag_logs_file_id', type_='foreignkey'
        )

    op.drop_index(
        'ix_rag_query_logs_created_at', table_name='rag_query_logs'
    )
    op.drop_index(
        'ix_rag_query_logs_file_id', table_name='rag_query_logs'
    )
    op.drop_index(
        'ix_uploaded_files_created_at', table_name='uploaded_files'
    )
    op.drop_index(
        'ix_chat_history_created_at', table_name='chat_history'
    )
    op.drop_index(
        'ix_chat_history_session_id', table_name='chat_history'
    )
