"""rag logs rerank fields + nullable file_id

Revision ID: c8d2e3f4a5b6
Revises: b7a1c2d3e4f5
Create Date: 2026-10-18 23:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c8d2e3f4a5b6'
down_revision: Union[str, Sequence[str], None] = 'b7a1c2d3e4f5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """rag_query_logs 增加 Hybrid+Rerank 检索字段；file_id 改可空。"""
    with op.batch_alter_table('rag_query_logs', schema=None) as batch:
        # 全局检索没有单一文件，file_id 允许为空
        batch.alter_column(
            'file_id',
            existing_type=sa.Integer(),
            nullable=True,
        )
        batch.add_column(
            sa.Column('best_rerank_score', sa.Float(), nullable=True)
        )
        batch.add_column(
            sa.Column(
                'rerank_scores_json', sa.Text(), nullable=True
            )
        )
        batch.add_column(
            sa.Column('query_count', sa.Integer(), nullable=True)
        )
        batch.add_column(
            sa.Column(
                'retrieval_latency_ms', sa.Integer(), nullable=True
            )
        )
        batch.add_column(
            sa.Column(
                'retrieval_mode', sa.String(), nullable=True
            )
        )


def downgrade() -> None:
    """回滚：删除新增字段并把 file_id 恢复为非空。"""
    with op.batch_alter_table('rag_query_logs', schema=None) as batch:
        batch.drop_column('retrieval_mode')
        batch.drop_column('retrieval_latency_ms')
        batch.drop_column('query_count')
        batch.drop_column('rerank_scores_json')
        batch.drop_column('best_rerank_score')
        batch.alter_column(
            'file_id',
            existing_type=sa.Integer(),
            nullable=False,
        )
