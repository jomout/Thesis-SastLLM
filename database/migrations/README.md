# Database migrations

Alembic migrations are the authoritative PostgreSQL schema history. Run them
from the repository root with `uv run alembic upgrade head`.

The SQL files in the parent directory are retained as legacy bootstrap and
reference artifacts. They are not applied automatically by Docker Compose.
