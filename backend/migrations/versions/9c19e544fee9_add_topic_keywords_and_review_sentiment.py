"""add topic keywords and review sentiment_computed flag

Revision ID: 9c19e544fee9
Revises: 7e8cbb8813af
Create Date: 2026-08-22 06:30:00.000000

NOTE: hand-trimmed after autogeneration, same reason as the prior migration —
autogenerate was run against SQLite and produced spurious NUMERIC->UUID noise
that isn't real schema drift on Postgres. Only the two intended columns kept.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '9c19e544fee9'
down_revision: Union[str, Sequence[str], None] = '7e8cbb8813af'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table('topics', schema=None) as batch_op:
        batch_op.add_column(sa.Column('keywords_raw', sa.Text(), nullable=True))

    with op.batch_alter_table('reviews', schema=None) as batch_op:
        batch_op.add_column(
            sa.Column('sentiment_computed', sa.Boolean(), nullable=False, server_default=sa.false())
        )


def downgrade() -> None:
    with op.batch_alter_table('reviews', schema=None) as batch_op:
        batch_op.drop_column('sentiment_computed')

    with op.batch_alter_table('topics', schema=None) as batch_op:
        batch_op.drop_column('keywords_raw')
