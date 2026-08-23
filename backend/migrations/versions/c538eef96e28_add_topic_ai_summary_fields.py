"""add topic ai summary fields and product executive summary

Revision ID: c538eef96e28
Revises: 9c19e544fee9
Create Date: 2026-08-22 06:45:00.000000

NOTE: hand-trimmed after autogeneration (same SQLite NUMERIC->UUID noise as
prior migrations — not real schema drift on Postgres). Only the intended
columns kept.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'c538eef96e28'
down_revision: Union[str, Sequence[str], None] = '9c19e544fee9'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table('topics', schema=None) as batch_op:
        batch_op.add_column(sa.Column('common_complaints', sa.Text(), nullable=True))
        batch_op.add_column(sa.Column('common_compliments', sa.Text(), nullable=True))
        batch_op.add_column(sa.Column('suggestions', sa.Text(), nullable=True))
        batch_op.add_column(sa.Column('representative_positive_review', sa.Text(), nullable=True))
        batch_op.add_column(sa.Column('representative_negative_review', sa.Text(), nullable=True))

    with op.batch_alter_table('products', schema=None) as batch_op:
        batch_op.add_column(sa.Column('executive_summary', sa.Text(), nullable=True))
        batch_op.add_column(sa.Column('executive_summary_updated_at', sa.DateTime(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table('products', schema=None) as batch_op:
        batch_op.drop_column('executive_summary_updated_at')
        batch_op.drop_column('executive_summary')

    with op.batch_alter_table('topics', schema=None) as batch_op:
        batch_op.drop_column('representative_negative_review')
        batch_op.drop_column('representative_positive_review')
        batch_op.drop_column('suggestions')
        batch_op.drop_column('common_compliments')
        batch_op.drop_column('common_complaints')
