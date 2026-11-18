"""Add auxiliary snippet and repository prediction tables.

Revision ID: 0002_aux_tables
Revises: 0001_core_schema
Create Date: 2026-11-18
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0002_aux_tables"
down_revision: str | None = "0001_core_schema"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _create_updated_at_trigger(table_name: str) -> None:
    op.execute(
        sa.text(
            f"""
            CREATE TRIGGER trg_{table_name}_updated_at
            BEFORE UPDATE ON {table_name}
            FOR EACH ROW EXECUTE FUNCTION set_updated_at()
            """
        )
    )


def upgrade() -> None:
    op.create_table(
        "csn_snippets",
        sa.Column("csn_snippet_id", sa.Integer(), primary_key=True),
        sa.Column("repository", sa.String(length=255), nullable=False),
        sa.Column("filepath", sa.String(length=1024), nullable=False),
        sa.Column("start_line", sa.Integer(), nullable=False),
        sa.Column("end_line", sa.Integer(), nullable=False),
        sa.Column("code", sa.Text(), nullable=False),
        sa.Column("functionality", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("clock_timestamp()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("clock_timestamp()"), nullable=False),
    )
    op.create_table(
        "repository_predictions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("repository_id", sa.Integer(), nullable=False),
        sa.Column("method", sa.String(length=32), nullable=False),
        sa.Column("classification", sa.String(length=64)),
        sa.Column("flags_json", postgresql.JSONB(astext_type=sa.Text())),
        sa.Column("justification", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("clock_timestamp()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("clock_timestamp()"), nullable=False),
        sa.ForeignKeyConstraint(["repository_id"], ["repositories.repository_id"], ondelete="CASCADE"),
    )
    _create_updated_at_trigger("csn_snippets")
    _create_updated_at_trigger("repository_predictions")


def downgrade() -> None:
    op.drop_table("repository_predictions")
    op.drop_table("csn_snippets")
