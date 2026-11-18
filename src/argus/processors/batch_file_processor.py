import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

from argus.configs import get_logger

logger = get_logger(__name__)

TERMINAL_BATCH_STATUSES = {"completed", "failed", "expired", "cancelled"}


class BatchFileProcessor:
    """
    Handles submission and monitoring of multiple OpenAI Batch API jobs
    targeting the /v1/responses endpoint.
    """

    def __init__(
        self,
        *,
        batch_files_dir: str | Path,
        output_dir: str | Path,
        poll_interval: int = 30,
        max_wait_hours: int = 30,
    ) -> None:
        """
        Initialize a BatchFileProcessor.

        Parameters
        ----------
        batch_dir : str | Path
            Directory containing .jsonl files to submit.
        output_dir : str | Path
            Directory to store downloaded outputs and errors.
        poll_interval : int
            Polling frequency in seconds for batch status.
        max_wait_hours : int
            Maximum hours to wait for a batch before giving up.
        """
        self.client = OpenAI()
        self.batch_files_dir = Path(batch_files_dir)
        self.output_dir = Path(output_dir)

        self.poll_interval = poll_interval
        self.max_wait_hours = max_wait_hours

        self.output_dir.mkdir(parents=True, exist_ok=True)
        existing_results = sorted(self.output_dir.glob("*.jsonl"))
        if existing_results:
            raise FileExistsError(f"Refusing to mix new Batch API results with {len(existing_results)} existing JSONL files in {self.output_dir}. Move the previous results first.")

        logger.debug(f"BatchFileProcessor initialized: batch_dir={self.batch_files_dir}, output_dir={self.output_dir}, poll_interval={self.poll_interval}s")

    def _upload_file(self, path: Path):
        """Upload a JSONL file for batch processing."""
        logger.info(f"Uploading {path.name} ...")
        with path.open("rb") as f:
            uploaded = self.client.files.create(file=f, purpose="batch")
        logger.debug(f"Uploaded {path.name} → file_id={uploaded.id}")
        return uploaded.id

    def _create_batch(self, file_id: str):
        """Create a batch job targeting /v1/responses."""
        batch = self.client.batches.create(
            input_file_id=file_id,
            endpoint="/v1/responses",
            completion_window="24h",
        )
        logger.info(f"Created batch job {batch.id}")
        return batch.id

    def _poll_batch(self, batch_id: str):
        """Poll until batch completes or fails."""
        start_time = time.time()
        while True:
            batch = self.client.batches.retrieve(batch_id)
            if batch.status in TERMINAL_BATCH_STATUSES:
                logger.info(f"Batch {batch_id} finished with status: {batch.status}")
                return batch
            elapsed = time.time() - start_time
            if elapsed > self.max_wait_hours * 3600:
                raise TimeoutError(f"Batch {batch_id} exceeded the configured {self.max_wait_hours}-hour wait.")
            logger.debug(f"Batch {batch_id} status={batch.status} ... waiting {self.poll_interval}s")
            time.sleep(self.poll_interval)

    def _download_file(self, file_id: str, output_path: Path):
        """Download and save a file by ID."""
        if output_path.exists():
            raise FileExistsError(f"Refusing to overwrite existing Batch API result file: {output_path}")
        logger.debug(f"Downloading file_id={file_id} → {output_path.name}")
        content = self.client.files.content(file_id).content
        with open(output_path, "wb") as f:
            f.write(content)
        logger.info(f"Saved {output_path.name}")

    def _process_file(self, file_path: Path) -> None:
        """Upload, submit, poll, and download one Batch API request file."""
        file_id = self._upload_file(file_path)
        batch_id = self._create_batch(file_id)
        batch = self._poll_batch(batch_id)

        if getattr(batch, "output_file_id", None):
            output_path = self.output_dir / f"{file_path.stem}_output.jsonl"
            self._download_file(batch.output_file_id, output_path)  # type: ignore

        if getattr(batch, "error_file_id", None):
            error_path = self.output_dir / f"{file_path.stem}_errors.jsonl"
            self._download_file(batch.error_file_id, error_path)  # type: ignore

        if batch.status != "completed":
            raise RuntimeError(f"Batch {batch.id} ended with status {batch.status!r}.")

    def process_all(self) -> None:
        """Submit all .jsonl batch files and await their completion."""
        jsonl_files = sorted(self.batch_files_dir.glob("*.jsonl"))
        if not jsonl_files:
            logger.warning(f"No .jsonl files found in {self.batch_files_dir}.")
            return

        logger.info(f"Found {len(jsonl_files)} batch files to process.")

        failures: list[str] = []
        for file_path in tqdm(jsonl_files, desc="Submitting batches"):
            try:
                self._process_file(file_path)
            except Exception as error:
                failures.append(f"{file_path.name}: {error}")
                logger.exception("Error processing Batch API file", file=file_path.name, error=str(error))

        if failures:
            raise RuntimeError(f"{len(failures)} Batch API file(s) failed: {'; '.join(failures)}")

    def process_all_concurrently(self, max_concurrent: int = 3) -> None:
        """Process request files concurrently and fail if any file is unsuccessful."""
        if max_concurrent <= 0:
            raise ValueError("max_concurrent must be greater than zero.")

        jsonl_files = sorted(self.batch_files_dir.glob("*.jsonl"))
        if not jsonl_files:
            logger.warning(f"No .jsonl files found in {self.batch_files_dir}.")
            return

        logger.info(f"Found {len(jsonl_files)} batch files to process (running up to {max_concurrent} concurrently).")

        failures: list[str] = []
        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            future_files = {executor.submit(self._process_file, file_path): file_path for file_path in jsonl_files}
            for future in as_completed(future_files):
                file_path = future_files[future]
                try:
                    future.result()
                    logger.info("Completed Batch API file", file=file_path.name)
                except Exception as error:
                    failures.append(f"{file_path.name}: {error}")
                    logger.exception("Error processing Batch API file", file=file_path.name, error=str(error))

        if failures:
            raise RuntimeError(f"{len(failures)} Batch API file(s) failed: {'; '.join(failures)}")
