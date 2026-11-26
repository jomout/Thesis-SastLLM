"""Create the core repository analysis schema.

Revision ID: 0001_core_schema
Revises:
Create Date: 2026-11-18
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001_core_schema"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_SET_UPDATED_AT_FUNCTION = """
CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = clock_timestamp();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql
"""

_PROCESSED_FUNCTIONS = """
CREATE OR REPLACE FUNCTION recompute_file_processed(p_file_id INT)
RETURNS VOID AS $$
DECLARE
    v_has_unprocessed BOOLEAN;
    v_new BOOLEAN;
BEGIN
    SELECT EXISTS (
        SELECT 1 FROM snippets s
        WHERE s.file_id = p_file_id AND s.processed IS NOT TRUE
    ) INTO v_has_unprocessed;
    v_new := NOT v_has_unprocessed;
    UPDATE files SET processed = v_new
    WHERE file_id = p_file_id AND processed IS DISTINCT FROM v_new;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION recompute_repository_processed(p_repo_id INT)
RETURNS VOID AS $$
DECLARE
    v_has_unprocessed BOOLEAN;
    v_new BOOLEAN;
BEGIN
    SELECT EXISTS (
        SELECT 1 FROM files f
        WHERE f.repository_id = p_repo_id AND f.processed IS NOT TRUE
    ) INTO v_has_unprocessed;
    v_new := NOT v_has_unprocessed;
    UPDATE repositories SET processed = v_new
    WHERE repository_id = p_repo_id AND processed IS DISTINCT FROM v_new;
END;
$$ LANGUAGE plpgsql
"""

_PROCESSED_TRIGGER_FUNCTIONS = """
CREATE OR REPLACE FUNCTION trg_snippet_refresh_file_processed()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        PERFORM recompute_file_processed(NEW.file_id);
    ELSIF TG_OP = 'UPDATE' THEN
        IF NEW.file_id IS DISTINCT FROM OLD.file_id THEN
            PERFORM recompute_file_processed(OLD.file_id);
        END IF;
        PERFORM recompute_file_processed(NEW.file_id);
    ELSIF TG_OP = 'DELETE' THEN
        PERFORM recompute_file_processed(OLD.file_id);
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION trg_file_refresh_repository_processed()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        PERFORM recompute_repository_processed(NEW.repository_id);
    ELSIF TG_OP = 'UPDATE' THEN
        IF NEW.repository_id IS DISTINCT FROM OLD.repository_id THEN
            PERFORM recompute_repository_processed(OLD.repository_id);
        END IF;
        PERFORM recompute_repository_processed(NEW.repository_id);
    ELSIF TG_OP = 'DELETE' THEN
        PERFORM recompute_repository_processed(OLD.repository_id);
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql
"""


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
        "repositories",
        sa.Column("repository_id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("label", sa.String(length=255)),
        sa.Column("processed", sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column("split", sa.String(length=50)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("clock_timestamp()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("clock_timestamp()"), nullable=False),
        sa.UniqueConstraint("name"),
    )
    op.create_table(
        "files",
        sa.Column("file_id", sa.Integer(), primary_key=True),
        sa.Column("repository_id", sa.Integer(), nullable=False),
        sa.Column("language", sa.String(length=50), nullable=False),
        sa.Column("filename", sa.String(length=255), nullable=False),
        sa.Column("filepath", sa.String(length=1024), nullable=False),
        sa.Column("processed", sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("clock_timestamp()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("clock_timestamp()"), nullable=False),
        sa.ForeignKeyConstraint(["repository_id"], ["repositories.repository_id"], ondelete="CASCADE"),
    )
    op.create_index("idx_files_repository_id", "files", ["repository_id"])
    op.create_index("idx_files_repo_processed", "files", ["repository_id", "processed"])
    op.create_table(
        "snippets",
        sa.Column("snippet_id", sa.Integer(), primary_key=True),
        sa.Column("file_id", sa.Integer(), nullable=False),
        sa.Column("start_line", sa.Integer(), nullable=False),
        sa.Column("end_line", sa.Integer(), nullable=False),
        sa.Column("code", sa.Text(), nullable=False),
        sa.Column("processed", sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("clock_timestamp()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("clock_timestamp()"), nullable=False),
        sa.ForeignKeyConstraint(["file_id"], ["files.file_id"], ondelete="CASCADE"),
    )
    op.create_index("idx_snippets_file_id", "snippets", ["file_id"])
    op.create_index("idx_snippets_processed", "snippets", ["processed"])
    op.create_index("idx_snippets_file_id_processed", "snippets", ["file_id", "processed"])
    op.create_table(
        "functionalities",
        sa.Column("functionality_id", sa.Integer(), primary_key=True),
        sa.Column("snippet_id", sa.Integer(), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("tag", sa.Text(), nullable=False),
        sa.Column("cluster_id", sa.Integer()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("clock_timestamp()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("clock_timestamp()"), nullable=False),
        sa.ForeignKeyConstraint(["snippet_id"], ["snippets.snippet_id"], ondelete="CASCADE"),
    )
    op.create_index("idx_functionalities_snippet_id", "functionalities", ["snippet_id"])
    op.create_index("idx_functionalities_cluster_id", "functionalities", ["cluster_id"])
    op.create_index("idx_functionalities_tag", "functionalities", ["tag"])

    op.execute(_SET_UPDATED_AT_FUNCTION)
    for table_name in ("repositories", "files", "snippets", "functionalities"):
        _create_updated_at_trigger(table_name)

    op.execute(_PROCESSED_FUNCTIONS)
    op.execute(_PROCESSED_TRIGGER_FUNCTIONS)
    op.execute(
        """
        CREATE TRIGGER trg_snippets_refresh_file_processed
        AFTER INSERT OR UPDATE OR DELETE ON snippets
        FOR EACH ROW EXECUTE FUNCTION trg_snippet_refresh_file_processed()
        """
    )
    op.execute(
        """
        CREATE TRIGGER trg_files_refresh_repository_processed
        AFTER INSERT OR UPDATE OR DELETE ON files
        FOR EACH ROW EXECUTE FUNCTION trg_file_refresh_repository_processed()
        """
    )


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS trg_files_refresh_repository_processed ON files")
    op.execute("DROP TRIGGER IF EXISTS trg_snippets_refresh_file_processed ON snippets")
    op.execute("DROP FUNCTION IF EXISTS trg_file_refresh_repository_processed()")
    op.execute("DROP FUNCTION IF EXISTS trg_snippet_refresh_file_processed()")
    op.execute("DROP FUNCTION IF EXISTS recompute_repository_processed(INT)")
    op.execute("DROP FUNCTION IF EXISTS recompute_file_processed(INT)")
    op.drop_table("functionalities")
    op.drop_table("snippets")
    op.drop_table("files")
    op.drop_table("repositories")
    op.execute("DROP FUNCTION IF EXISTS set_updated_at()")
