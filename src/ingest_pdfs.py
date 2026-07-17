"""Chunk PJM study PDFs, embed them, and store the vectors in Chroma.

Loads with PyMuPDF by default, since it renders the cost tables in these studies as
markdown rather than flattening them to plain text, which keeps the numbers readable.
PyPDFLoader is kept as a fallback for a file PyMuPDF cannot parse.
"""

import argparse
from pathlib import Path

import tiktoken
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader, PyPDFLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings

COLLECTION_NAME = "pjm_studies"

# text-embedding-3-small list price, per openai's pricing page.
EMBED_PRICE_PER_MILLION_TOKENS = 0.02

# used only if the configured embed model is not in tiktoken's built in table.
FALLBACK_ENCODING = "cl100k_base"


def load_pdf_pages(path: Path, use_pypdf: bool = False) -> list[Document]:
    """Load one pdf into a list of per page documents."""
    if use_pypdf:
        loader = PyPDFLoader(str(path))
    else:
        loader = PyMuPDFLoader(str(path), extract_tables="markdown")
    return loader.load()


def chunk_pages(pages: list[Document], queue_id: str, source: str) -> list[Document]:
    """Split pages into overlapping chunks tagged with the project and file they came from."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )
    chunks = splitter.split_documents(pages)
    for chunk in chunks:
        chunk.metadata["queue_id"] = queue_id
        chunk.metadata["source"] = source
    return chunks


def load_and_chunk_pdf(path: Path, use_pypdf: bool = False) -> list[Document]:
    """Load and split one pdf, the queue id and file name come from the path itself."""
    pages = load_pdf_pages(path, use_pypdf=use_pypdf)
    return chunk_pages(pages, queue_id=path.stem, source=path.name)


def count_tokens(text: str) -> int:
    """Estimate the token count of a piece of text under the configured embedding model."""
    try:
        encoding = tiktoken.encoding_for_model(settings.embed_model)
    except KeyError:
        encoding = tiktoken.get_encoding(FALLBACK_ENCODING)
    return len(encoding.encode(text))


def estimate_embedding_cost(chunks: list[Document]) -> tuple[int, float]:
    """Return the total token count and dollar cost estimate for embedding these chunks."""
    tokens = sum(count_tokens(chunk.page_content) for chunk in chunks)
    cost = tokens / 1_000_000 * EMBED_PRICE_PER_MILLION_TOKENS
    return tokens, cost


def open_store(embeddings: Embeddings | None) -> Chroma:
    """Open the persisted chroma store, creating an empty one on first use."""
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(settings.chroma_dir),
    )


def existing_queue_ids(store: Chroma) -> set[str]:
    """Return the queue ids that already have chunks in the store."""
    records = store.get(include=["metadatas"])
    return {metadata["queue_id"] for metadata in records["metadatas"]}


def get_retriever(k: int | None = None) -> VectorStoreRetriever:
    """Return a retriever over the persisted Chroma store, top k from config if k is None."""
    embeddings = OpenAIEmbeddings(model=settings.embed_model, api_key=settings.openai_api_key)
    store = open_store(embeddings)
    return store.as_retriever(search_kwargs={"k": k or settings.retrieval_k})


def embed_new_pdfs(rebuild: bool = False, use_pypdf: bool = False) -> dict[str, float]:
    """Embed pdfs from disk that are not yet in the store, up to max_pdfs, respecting dry_run."""
    settings.pdf_raw_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(settings.pdf_raw_dir.glob("*.pdf"))

    store = open_store(None)
    if rebuild:
        store.reset_collection()
        already = set()
    else:
        already = existing_queue_ids(store)

    to_process = [path for path in paths if path.stem not in already][: settings.max_pdfs]
    chunks = [chunk for path in to_process for chunk in load_and_chunk_pdf(path, use_pypdf=use_pypdf)]
    tokens, cost = estimate_embedding_cost(chunks)

    print(
        f"{len(to_process)} new pdfs, {len(chunks)} chunks, "
        f"estimated tokens {tokens}, estimated cost ${cost:.4f}"
    )

    if settings.dry_run:
        print("dry run, no embedding call made")
        return {"pdfs": len(to_process), "chunks": len(chunks), "tokens": tokens, "cost": cost}

    if chunks:
        embeddings = OpenAIEmbeddings(model=settings.embed_model, api_key=settings.openai_api_key)
        open_store(embeddings).add_documents(chunks)
    print(f"embedded {len(chunks)} chunks from {len(to_process)} pdfs")
    return {"pdfs": len(to_process), "chunks": len(chunks), "tokens": tokens, "cost": cost}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chunk and embed PJM study PDFs into the Chroma vector store."
    )
    parser.add_argument(
        "--rebuild", action="store_true", help="wipe the store and re-embed every pdf on disk"
    )
    parser.add_argument(
        "--loader",
        choices=["pymupdf", "pypdf"],
        default="pymupdf",
        help="pdf text loader, pypdf is a fallback for files pymupdf cannot parse",
    )
    args = parser.parse_args()
    embed_new_pdfs(rebuild=args.rebuild, use_pypdf=(args.loader == "pypdf"))


if __name__ == "__main__":
    main()
