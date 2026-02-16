-- ======================================================================
-- 1__triggers.sql
-- Trigger functions and triggers for processed propagation
-- ======================================================================

-- ----------------------------------------------------------------------
-- Function: recompute_file_processed(file_id)
-- Updates a file's processed status based on its snippets
-- ----------------------------------------------------------------------
CREATE OR REPLACE FUNCTION recompute_file_processed(p_file_id INT)
RETURNS VOID AS $$
DECLARE
    v_has_unprocessed BOOLEAN;
    v_new BOOLEAN;
BEGIN
    -- Check if the file has any unprocessed snippets
    SELECT EXISTS (
        SELECT 1
        FROM snippets s
        WHERE s.file_id = p_file_id
          AND s.processed IS NOT TRUE
    )
    INTO v_has_unprocessed;

    -- File is processed if no unprocessed snippets exist (empty = true)
    v_new := NOT v_has_unprocessed;

    UPDATE files
    SET processed = v_new
    WHERE file_id = p_file_id
      AND processed IS DISTINCT FROM v_new;
END;
$$ LANGUAGE plpgsql;

-- ----------------------------------------------------------------------
-- Function: recompute_repository_processed(repository_id)
-- Updates a repository's processed status based on its files
-- ----------------------------------------------------------------------
CREATE OR REPLACE FUNCTION recompute_repository_processed(p_repo_id INT)
RETURNS VOID AS $$
DECLARE
    v_has_unprocessed BOOLEAN;
    v_new BOOLEAN;
BEGIN
    -- Check if the repository has any unprocessed files
    SELECT EXISTS (
        SELECT 1
        FROM files f
        WHERE f.repository_id = p_repo_id
          AND f.processed IS NOT TRUE
    )
    INTO v_has_unprocessed;

    -- Repository is processed if no unprocessed files exist (empty = true)
    v_new := NOT v_has_unprocessed;

    UPDATE repositories
    SET processed = v_new
    WHERE repository_id = p_repo_id
      AND processed IS DISTINCT FROM v_new;
END;
$$ LANGUAGE plpgsql;

-- ----------------------------------------------------------------------
-- Trigger Function: refresh file.processed after snippet changes
-- ----------------------------------------------------------------------
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

DROP TRIGGER IF EXISTS trg_snippets_refresh_file_processed ON snippets;
CREATE TRIGGER trg_snippets_refresh_file_processed
AFTER INSERT OR UPDATE OR DELETE ON snippets
FOR EACH ROW
EXECUTE FUNCTION trg_snippet_refresh_file_processed();

-- ----------------------------------------------------------------------
-- Trigger Function: refresh repository.processed after file changes
-- ----------------------------------------------------------------------
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
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_files_refresh_repository_processed ON files;
CREATE TRIGGER trg_files_refresh_repository_processed
AFTER INSERT OR UPDATE OR DELETE ON files
FOR EACH ROW
EXECUTE FUNCTION trg_file_refresh_repository_processed();

-- ----------------------------------------------------------------------
-- One-time backfill (safe to re-run)
-- ----------------------------------------------------------------------

-- Update files.processed based on current snippets
WITH computed AS (
    SELECT f.file_id,
           NOT EXISTS (
               SELECT 1 FROM snippets s
               WHERE s.file_id = f.file_id AND s.processed IS NOT TRUE
           ) AS new_processed
    FROM files f
)
UPDATE files f
SET processed = c.new_processed
FROM computed c
WHERE f.file_id = c.file_id
  AND f.processed IS DISTINCT FROM c.new_processed;

-- Update repositories.processed based on current files
WITH computed AS (
    SELECT r.repository_id,
           NOT EXISTS (
               SELECT 1 FROM files f
               WHERE f.repository_id = r.repository_id AND f.processed IS NOT TRUE
           ) AS new_processed
    FROM repositories r
)
UPDATE repositories r
SET processed = c.new_processed
FROM computed c
WHERE r.repository_id = c.repository_id
  AND r.processed IS DISTINCT FROM c.new_processed;

-- ======================================================================
-- End of 1__triggers.sql
-- ======================================================================
