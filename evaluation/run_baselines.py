"""Run reproducible LLM and retrieval baselines from evaluation/baselines.json.

This module deliberately keeps model serving outside the evaluator. Start each
vLLM endpoint with scripts/start_vllm.sh, then point the end-to-end command at
that endpoint. Retrieval indexes live in separate directories so changing an
embedding model never overwrites the reference index.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HERE = ROOT / "evaluation"
DEFAULT_MANIFEST = HERE / "baselines.json"
DEFAULT_RESULTS_DIR = HERE / "baseline_results"
DEFAULT_CHUNKS = ROOT / "data" / "processed" / "chunks.jsonl"
CPU_THREAD_ENV = {
    "OMP_NUM_THREADS": "6",
    "MKL_NUM_THREADS": "6",
    "OPENBLAS_NUM_THREADS": "6",
    "NUMEXPR_NUM_THREADS": "6",
    "VECLIB_MAXIMUM_THREADS": "6",
    "RAYON_NUM_THREADS": "6",
}


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "1.0":
        raise SystemExit("Baseline manifest schema 1.0 is required")
    return payload


def select(items: list[dict], item_id: str, kind: str) -> dict:
    for item in items:
        if item["id"] == item_id:
            return item
    choices = ", ".join(item["id"] for item in items)
    raise SystemExit(f"Unknown {kind} baseline {item_id!r}; choose one of: {choices}")


def retrieval_env(config: dict) -> dict[str, str]:
    qdrant_path = Path(config["qdrant_path"])
    if not qdrant_path.is_absolute():
        qdrant_path = ROOT / qdrant_path
    return {
        "EMBEDDING_MODEL": config["embedding_model"],
        "EMBEDDING_QUERY_PROMPT_NAME": config.get("embedding_query_prompt_name") or "",
        "RERANKER_MODEL": config["reranker_model"],
        "QDRANT_PATH": str(qdrant_path),
    }


def run_subprocess(arguments: list[str], env_overrides: dict[str, str]) -> None:
    env = os.environ.copy()
    env.update(CPU_THREAD_ENV)
    env.update(env_overrides)
    subprocess.run(arguments, cwd=ROOT, env=env, check=True)


def chunk_from_payload(payload: dict):
    """Load current or pre-chunk_index JSONL without changing stable IDs."""
    from src.ingestion.chunking import Chunk

    if "chunk_index" not in payload:
        suffix = payload.get("chunk_id", "").rsplit("-", 1)[-1]
        if not suffix.isdigit():
            raise ValueError(f"Cannot infer chunk_index from {payload.get('chunk_id')!r}")
        payload = {**payload, "chunk_index": int(suffix)}
    return Chunk(**payload)


def build_index(config: dict, chunks_path: Path) -> None:
    os.environ.update(CPU_THREAD_ENV)
    os.environ.update(retrieval_env(config))

    # Imports happen after environment selection so pydantic-settings observes
    # this baseline rather than values from .env.
    from src.ingestion.indexing import VectorStore, build_index as index_chunks

    chunks = [
        chunk_from_payload(json.loads(line))
        for line in chunks_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not chunks:
        raise SystemExit(f"No chunks found in {chunks_path}")
    with VectorStore() as store:
        index_chunks(chunks, store=store)
        print(
            f"Indexed {store.count()} fixed chunks with {config['embedding_model']} "
            f"at {store.path}"
        )


def run_retrieval(config: dict, results_dir: Path, extra: list[str]) -> None:
    output_dir = results_dir / f"retrieval__{config['id']}"
    run_subprocess(
        [
            sys.executable,
            "-m",
            "evaluation.evaluate_retrieval",
            "--details",
            str(output_dir / "retrieval_results.json"),
            "--report",
            str(output_dir / "retrieval_evaluation.md"),
            *extra,
        ],
        retrieval_env(config),
    )


def run_end_to_end(
    llm: dict,
    retrieval: dict,
    base_url: str,
    results_dir: Path,
    resume: bool,
    extra: list[str],
) -> None:
    run_id = f"{llm['id']}__{retrieval['id']}"
    output_dir = results_dir / run_id
    arguments = [
        sys.executable,
        "-m",
        "evaluation.evaluate_end_to_end",
        "--details",
        str(output_dir / "end_to_end_results.json"),
        "--report",
        str(output_dir / "end_to_end_evaluation.md"),
    ]
    if resume:
        arguments.append("--resume")
    arguments.extend(extra)
    env = {
        **retrieval_env(retrieval),
        "LLM_PROVIDER": "openai",
        "OPENAI_BASE_URL": base_url,
        "OPENAI_MODEL": llm["model"],
        "OPENAI_CHAT_TEMPLATE_KWARGS": json.dumps(
            llm.get("chat_template_kwargs", {})
        ),
    }
    run_subprocess(arguments, env)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run configured model baselines")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="show the configured matrix")
    list_parser.set_defaults(action="list")

    index_parser = subparsers.add_parser("index", help="build one isolated retrieval index")
    index_parser.add_argument("--retrieval", required=True)
    index_parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)

    retrieval_parser = subparsers.add_parser("retrieval", help="evaluate one retrieval pair")
    retrieval_parser.add_argument("--retrieval", required=True)
    retrieval_parser.add_argument("evaluator_args", nargs=argparse.REMAINDER)

    e2e_parser = subparsers.add_parser("end-to-end", help="evaluate one LLM + retrieval pair")
    e2e_parser.add_argument("--llm", required=True)
    e2e_parser.add_argument("--retrieval", default="bge_m3")
    e2e_parser.add_argument("--base-url", required=True)
    e2e_parser.add_argument("--resume", action="store_true")
    e2e_parser.add_argument("evaluator_args", nargs=argparse.REMAINDER)

    report_parser = subparsers.add_parser("report", help="regenerate the comparison report")
    report_parser.add_argument(
        "--output", type=Path, default=HERE / "model_baseline_comparison.md"
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest = load_manifest(args.manifest)
    if args.command == "list":
        print("LLM baselines:")
        for item in manifest["llm_baselines"]:
            print(f"  {item['id']}: {item['model']} ({item['role']})")
        print("Retrieval baselines:")
        for item in manifest["retrieval_baselines"]:
            print(
                f"  {item['id']}: {item['embedding_model']} + "
                f"{item['reranker_model']} ({item['role']})"
            )
        return 0

    if args.command == "report":
        from evaluation.compare_baselines import generate_report

        generate_report(args.manifest, args.results_dir, args.output)
        print(f"Wrote comparison to {args.output}")
        return 0

    retrieval = select(
        manifest["retrieval_baselines"], args.retrieval, "retrieval"
    )
    if args.command == "index":
        build_index(retrieval, args.chunks)
    elif args.command == "retrieval":
        run_retrieval(retrieval, args.results_dir, args.evaluator_args)
    elif args.command == "end-to-end":
        llm = select(manifest["llm_baselines"], args.llm, "LLM")
        run_end_to_end(
            llm,
            retrieval,
            args.base_url,
            args.results_dir,
            args.resume,
            args.evaluator_args,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

