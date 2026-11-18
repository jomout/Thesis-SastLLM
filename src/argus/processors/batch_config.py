from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class BatchFunctionalityConfig(BaseModel):
    """Validated OpenAI Batch API settings for functionality generation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str = Field(min_length=1)
    reasoning_effort: Literal["none", "low", "medium", "high", "xhigh", "max"] = "none"
    max_output_tokens: int = Field(default=8192, ge=16)
    snippet_batch_size: int = Field(default=20, ge=1)
    requests_per_file: int = Field(default=500, ge=1, le=50_000)
    max_file_size_bytes: int = Field(default=190_000_000, ge=1, le=200_000_000)
    batch_files_dir: Path = Path("api_batches")
    results_dir: Path = Path("batch_results")
    max_concurrent_batches: int = Field(default=3, ge=1, le=10)
    poll_interval_seconds: int = Field(default=60, ge=1)
    max_wait_hours: int = Field(default=30, ge=24)
