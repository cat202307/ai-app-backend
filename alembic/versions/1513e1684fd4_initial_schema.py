"""initial schema

Revision ID: 1513e1684fd4
Revises:
Create Date: 2026-10-08 00:08:22.729443

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '1513e1684fd4'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema: 创建全部初始表。"""
    # 聊天历史
    op.create_table(
        'chat_history',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_message', sa.String(), nullable=False),
        sa.Column('ai_answer', sa.String(), nullable=False),
        sa.Column('session_id', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('tool_name', sa.String(), nullable=True),
        sa.Column('tool_call_count', sa.Integer(), nullable=True),
        sa.Column('duration_ms', sa.Integer(), nullable=True),
        sa.Column('citations_json', sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_chat_history_id', 'chat_history', ['id'])

    # 上传文件
    op.create_table(
        'uploaded_files',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('filename', sa.String(), nullable=False),
        sa.Column('filepath', sa.String(), nullable=False),
        sa.Column('file_type', sa.String(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_uploaded_files_id', 'uploaded_files', ['id'])

    # RAG 问答日志
    op.create_table(
        'rag_query_logs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('file_id', sa.Integer(), nullable=False),
        sa.Column('filename', sa.String(), nullable=False),
        sa.Column('question', sa.String(), nullable=False),
        sa.Column('answer', sa.String(), nullable=False),
        sa.Column('best_distance', sa.Float(), nullable=True),
        sa.Column('max_distance', sa.Float(), nullable=True),
        sa.Column('retrieval_status', sa.String(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_rag_query_logs_id', 'rag_query_logs', ['id'])


def downgrade() -> None:
    """Downgrade schema: 删除全部初始表。"""
    op.drop_index('ix_rag_query_logs_id', table_name='rag_query_logs')
    op.drop_table('rag_query_logs')
    op.drop_index('ix_uploaded_files_id', table_name='uploaded_files')
    op.drop_table('uploaded_files')
    op.drop_index('ix_chat_history_id', table_name='chat_history')
    op.drop_table('chat_history')
