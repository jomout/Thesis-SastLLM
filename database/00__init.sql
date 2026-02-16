-- ======================================================================
-- Database Schema for Repository Intelligence System
-- PostgreSQL-compatible
-- With automatic created_at / updated_at timestamp management
-- ======================================================================

-- ----------------------------------------------------------------------
-- Trigger function to update `updated_at` timestamps automatically
-- ----------------------------------------------------------------------
CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = clock_timestamp();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- ======================================================================
-- Table: repositories
-- ======================================================================
CREATE TABLE repositories (
    repository_id SERIAL PRIMARY KEY,
    name VARCHAR(255) UNIQUE NOT NULL,
    label VARCHAR(255),
    processed BOOLEAN NOT NULL DEFAULT FALSE,
    split VARCHAR(50),  -- e.g., 'train', 'val', 'test'
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);

CREATE TRIGGER trg_repositories_updated_at
BEFORE UPDATE ON repositories
FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- ======================================================================
-- Table: files
-- ======================================================================
CREATE TABLE files (
    file_id SERIAL PRIMARY KEY,
    repository_id INTEGER NOT NULL REFERENCES repositories(repository_id) ON DELETE CASCADE,
    language VARCHAR(50) NOT NULL,
    filename VARCHAR(255) NOT NULL,
    filepath VARCHAR(1024) NOT NULL,
    processed BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);

CREATE INDEX idx_files_repository_id ON files(repository_id);

CREATE TRIGGER trg_files_updated_at
BEFORE UPDATE ON files
FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- ======================================================================
-- Table: snippets
-- ======================================================================
CREATE TABLE snippets (
    snippet_id SERIAL PRIMARY KEY,
    file_id INTEGER NOT NULL REFERENCES files(file_id) ON DELETE CASCADE,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    code TEXT NOT NULL,
    processed BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);

CREATE INDEX idx_snippets_file_id ON snippets(file_id);
CREATE INDEX idx_snippets_processed ON snippets(processed);

CREATE TRIGGER trg_snippets_updated_at
BEFORE UPDATE ON snippets
FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- ======================================================================
-- Table: functionalities
-- ======================================================================
CREATE TABLE functionalities (
    functionality_id SERIAL PRIMARY KEY,
    snippet_id INTEGER NOT NULL REFERENCES snippets(snippet_id) ON DELETE CASCADE,
    description TEXT NOT NULL,
    tag TEXT NOT NULL,
    cluster_id INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);

CREATE INDEX idx_functionalities_snippet_id ON functionalities(snippet_id);
CREATE INDEX idx_functionalities_cluster_id ON functionalities(cluster_id);
CREATE INDEX idx_functionalities_tag ON functionalities(tag);

CREATE TRIGGER trg_functionalities_updated_at
BEFORE UPDATE ON functionalities
FOR EACH ROW EXECUTE FUNCTION set_updated_at();


-- Helpful indexes for fast EXISTS checks
CREATE INDEX IF NOT EXISTS idx_snippets_file_id_processed
  ON snippets(file_id, processed);

CREATE INDEX IF NOT EXISTS idx_files_repo_processed
  ON files(repository_id, processed);

-- ======================================================================
-- End of Schema
-- ======================================================================
