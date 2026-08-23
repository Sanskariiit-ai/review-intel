"""decouple topic assignment from sentiment scoring

Revision ID: 7e8cbb8813af
Revises: 0cca01711d36
Create Date: 2026-08-22 06:10:59.908380

NOTE: this file was hand-trimmed after autogeneration. Autogenerate was run
against SQLite (for sandbox testing), which has no native UUID type, so it
produced a large number of spurious `alter_column ... NUMERIC -> UUID`
operations that are just SQLite-reflection artifacts, not real schema drift
on Postgres. Only the three actually-intended changes are kept below.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '7e8cbb8813af'
down_revision: Union[str, Sequence[str], None] = '0cca01711d36'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    with op.batch_alter_table('review_aspect_sentiments', schema=None) as batch_op:
        batch_op.add_column(sa.Column('confidence', sa.Float(), nullable=True))
        batch_op.alter_column('sentiment', existing_type=sa.VARCHAR(length=16), nullable=True)
        batch_op.alter_column('sentiment_score', existing_type=sa.FLOAT(), nullable=True)
        batch_op.create_unique_constraint('uq_review_topic_assignment', ['review_id', 'topic_id'])


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table('review_aspect_sentiments', schema=None) as batch_op:
        batch_op.drop_constraint('uq_review_topic_assignment', type_='unique')
        batch_op.alter_column('sentiment_score', existing_type=sa.FLOAT(), nullable=False)
        batch_op.alter_column('sentiment', existing_type=sa.VARCHAR(length=16), nullable=False)
        batch_op.drop_column('confidence')
