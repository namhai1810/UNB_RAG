"""CLI entry point.

    python -m src.main ingest                 # parse PDFs -> chunk -> index
    python -m src.main ask "question"         # one query through the graph
    python -m src.main chat                   # interactive loop
    python -m src.main status                 # what is indexed, which backend
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

from rich.console import Console
from rich.logging import RichHandler
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from src.config import settings
from src.logging_utils import rotating_file_handler

console = Console()


def setup_logging(verbose: bool) -> None:
    console_handler = RichHandler(
        console=console,
        rich_tracebacks=True,
        show_path=False,
    )
    console_handler.setLevel(logging.INFO if verbose else logging.WARNING)
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[console_handler, rotating_file_handler()],
        force=True,
    )


# --------------------------------------------------------------------------- #
# ingest
# --------------------------------------------------------------------------- #
def cmd_ingest(args: argparse.Namespace) -> int:
    from src.ingestion.chunking import chunk_corpus
    from src.ingestion.indexing import VectorStore, build_index
    from src.ingestion.loaders import load_corpus

    with console.status("[bold]Converting PDFs with Docling..."):
        docs = load_corpus()
    console.print(f"Loaded [bold]{len(docs)}[/bold] documents")
    console.print(f"Saved Docling Markdown to [cyan]{settings.resolved_markdown_dir}[/cyan]")

    with console.status("[bold]Chunking..."):
        chunks = chunk_corpus(docs)
    paragraph_count = sum(chunk.chunk_type == "paragraph" for chunk in chunks)
    table_count = sum(chunk.chunk_type == "table" for chunk in chunks)
    console.print(
        f"Produced [bold]{len(chunks)}[/bold] chunks "
        f"([bold]{paragraph_count}[/bold] paragraph, [bold]{table_count}[/bold] table-row)"
    )

    if args.dump:
        out = settings.processed_dir / "chunks.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            for chunk in chunks:
                fh.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
        console.print(f"Wrote chunks to [cyan]{out}[/cyan]")

    console.print("Embedding and indexing (first run downloads the models)...")
    with VectorStore() as store:
        build_index(chunks, store=store)
        console.print(f"[green]Indexed {store.count()} chunks[/green] -> {settings.qdrant_path}")
    return 0


# --------------------------------------------------------------------------- #
# ask / chat
# --------------------------------------------------------------------------- #
_STATUS_STYLE = {
    "answered": "green",
    "answered_partial": "yellow",
    "needs_clarification": "cyan",
    "rejected": "red",
    "insufficient": "yellow",
    "error": "red",
}


def render(state: dict, show_trace: bool = False) -> None:
    status = state.get("status", "error")
    style = _STATUS_STYLE.get(status, "white")

    console.print()
    console.print(Panel(Markdown(state.get("response", "")), title=f"[{style}]{status}[/{style}]",
                        border_style=style))

    payload = state.get("answer")
    if payload and payload.citations:
        table = Table(title="Citations", show_lines=False, header_style="bold")
        table.add_column("#", width=3)
        table.add_column("Source")
        table.add_column("Section", max_width=34)
        table.add_column("Pages", width=10)
        for citation in payload.citations:
            table.add_row(
                f"[{citation.marker}]", citation.source, citation.section or "-", citation.pages
            )
        console.print(table)
        console.print(f"Confidence: [bold]{payload.confidence}[/bold]")
        if payload.caveats:
            console.print(f"[dim]Caveats: {payload.caveats}[/dim]")

    if show_trace:
        console.print("\n[bold]Trace[/bold]")
        for event in state.get("trace", []):
            console.print(
                f"  [dim]r{event['round']}[/dim] [bold]{event['node']:<9}[/bold] {event['detail']}"
            )


def _preflight() -> bool:
    """Fail with an actionable message rather than a traceback three agents deep."""
    from src.ingestion.indexing import VectorStore
    from src.llm import get_llm

    with VectorStore() as store:
        if store.count() == 0:
            console.print("[red]Index is empty.[/red] Run: python -m src.main ingest")
            return False

    ok, detail = get_llm().health()
    if not ok:
        console.print(f"[red]LLM backend unavailable[/red] - {detail}")
        if settings.is_anthropic:
            console.print("Set ANTHROPIC_API_KEY, or switch LLM_PROVIDER=openai in .env")
        else:
            console.print(
                "Start a server, e.g.\n"
                f"  [cyan]vllm serve {settings.openai_model} --port 8000 "
                "--tensor-parallel-size 2[/cyan]\n"
                "or switch to LLM_PROVIDER=anthropic in .env"
            )
        return False
    return True


def _root_cause(exc: BaseException) -> str:
    while exc.__cause__ is not None:
        exc = exc.__cause__
    return str(exc)


def cmd_ask(args: argparse.Namespace) -> int:
    if not _preflight():
        return 1
    from src.graph.workflow import run

    try:
        with console.status("[bold]Running the agent graph..."):
            state = run(args.query)
    except Exception as exc:
        console.print(f"[red]Pipeline failed:[/red] {_root_cause(exc)}")
        return 1
    render(state, show_trace=args.trace)
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    if not _preflight():
        return 1
    from src.graph.workflow import run

    console.print(Panel("Multi-agent cyber-security RAG. Ctrl-C or 'exit' to quit.",
                        border_style="blue"))
    while True:
        try:
            query = console.input("\n[bold blue]>[/bold blue] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nBye.")
            return 0
        if not query:
            continue
        if query.lower() in {"exit", "quit", ":q"}:
            return 0
        try:
            render(run(query), show_trace=args.trace)
        except Exception as exc:  # keep the session alive on a bad turn
            console.print(f"[red]Error:[/red] {_root_cause(exc)}")


# --------------------------------------------------------------------------- #
# status
# --------------------------------------------------------------------------- #
def cmd_status(args: argparse.Namespace) -> int:
    from src.ingestion.indexing import VectorStore

    table = Table(show_header=False)
    table.add_row("LLM provider", settings.llm_provider)
    table.add_row(
        "LLM model",
        settings.anthropic_model if settings.is_anthropic else
        f"{settings.openai_model} @ {settings.openai_base_url}",
    )
    table.add_row("Embedding", settings.embedding_model)
    table.add_row("Reranker", settings.reranker_model)
    table.add_row("Device", settings.device)
    table.add_row("Docling table mode", settings.docling_table_mode)
    table.add_row("Docling OCR", str(settings.docling_do_ocr))
    table.add_row("Markdown output", str(settings.resolved_markdown_dir))
    table.add_row("Qdrant path", str(settings.qdrant_path))
    with VectorStore() as store:
        table.add_row("Indexed chunks", str(store.count()))
    table.add_row("PDFs in data/raw", str(len(list(settings.raw_dir.glob('*.pdf')))))
    console.print(table)
    return 0


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="multi-agent-cyber-rag")
    parser.add_argument("-v", "--verbose", action="store_true", help="log agent activity")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="parse, chunk and index data/raw")
    p_ingest.add_argument("--dump", action="store_true", help="also write chunks.jsonl")
    p_ingest.set_defaults(func=cmd_ingest)

    p_ask = sub.add_parser("ask", help="answer one question")
    p_ask.add_argument("query")
    p_ask.add_argument("--trace", action="store_true", help="show the agent trace")
    p_ask.set_defaults(func=cmd_ask)

    p_chat = sub.add_parser("chat", help="interactive session")
    p_chat.add_argument("--trace", action="store_true")
    p_chat.set_defaults(func=cmd_chat)

    p_status = sub.add_parser("status", help="show configuration and index state")
    p_status.set_defaults(func=cmd_status)

    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
