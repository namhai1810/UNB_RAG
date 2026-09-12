"""Central configuration. All values overridable via .env or environment variables."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    # ---------------------------------------------------------------- paths
    raw_dir: Path = ROOT / "data" / "raw"
    processed_dir: Path = ROOT / "data" / "processed"
    markdown_dir: Path = ROOT / "data" / "processed" / "markdown"
    qdrant_path: Path = ROOT / "data" / "processed" / "qdrant"
    collection: str = "cyber_docs"

    # -------------------------------------------------------------- Docling
    # Born-digital PDFs already contain a text layer. Enable this for scans.
    docling_do_ocr: bool = False
    docling_table_mode: Literal["fast", "accurate"] = "accurate"

    # ------------------------------------------------------------ LLM layer
    # "anthropic" -> Claude API ; "openai" -> any OpenAI-compatible endpoint
    # (vLLM / TGI / llama.cpp server / Ollama / OpenAI itself)
    llm_provider: Literal["anthropic", "openai"] = "openai"

    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-opus-5"
    # Server-side fallback so a `refusal` stop_reason does not kill the pipeline.
    # Cyber-security corpora legitimately trip the "cyber" safety classifier.
    anthropic_fallbacks: bool = True

    openai_api_key: str = "EMPTY"          # vLLM ignores the value but requires one
    openai_base_url: str = "http://0.0.0.0:8000/v1"
    openai_model: str = "Qwen/Qwen3-8B"

    llm_max_tokens: int = 4096
    llm_temperature: float = 0.0
    llm_timeout: float = 300.0

    # --------------------------------------------------------------- logging
    # Prompt/output is written at INFO; -v also displays it in the terminal.
    # Disable payloads in environments where user questions may be sensitive.
    log_payloads: bool = True
    log_max_chars: int = Field(default=4_000, ge=200)
    log_file: Path = ROOT / "logs" / "cyber-rag.log"
    log_max_bytes: int = Field(default=10 * 1024 * 1024, ge=1024)
    log_backup_count: int = Field(default=5, ge=0)

    # ------------------------------------------------------------ embedding
    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = 1024
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    device: str = "cuda"                    # "cuda", "cuda:1" or "cpu"
    embedding_batch_size: int = 8
    # BGE-M3 handles 8192 tokens; queries are short so a small cap is plenty.
    query_max_length: int = 512
    passage_max_length: int = 1024

    # ------------------------------------------------------------- chunking
    chunk_size: int = 700                   # tokens, measured with the BGE-M3 tokenizer
    chunk_overlap: int = 120

    # ------------------------------------------------------------ retrieval
    top_k_dense: int = 30                   # candidates pulled from the dense branch
    top_k_sparse: int = 30                  # candidates pulled from the BM25 branch
    top_k_fused: int = 20                   # kept after reciprocal-rank fusion
    top_k_rerank: int = 6                   # reranked seed chunks
    rerank_score_threshold: float = 0.0     # bge-reranker-v2-m3 logit; >0 ~ relevant
    neighbor_chunk_window: int = Field(default=1, ge=0)
    neighbor_max_total_chunks: int = Field(default=12, ge=1)
    neighbor_same_section_only: bool = True

    # ---------------------------------------------------------- agent loops
    max_retrieval_rounds: int = 3           # verifier -> rewrite -> retrieve budget
    min_supporting_chunks: int = 2          # below this the evidence is "insufficient"

    @property
    def resolved_log_file(self) -> Path:
        """Resolve relative LOG_FILE paths from the project root."""
        return self.log_file if self.log_file.is_absolute() else ROOT / self.log_file

    @property
    def resolved_markdown_dir(self) -> Path:
        """Resolve relative MARKDOWN_DIR paths from the project root."""
        return self.markdown_dir if self.markdown_dir.is_absolute() else ROOT / self.markdown_dir

    @property
    def is_anthropic(self) -> bool:
        return self.llm_provider == "anthropic"


settings = Settings()
