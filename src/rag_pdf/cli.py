import argparse
import json
import os
from typing import List, Dict, Any
from rich import print
from tqdm import tqdm

from .config import load_settings
from .pdf_processing import convert_doc_to_images, extract_text_from_doc
from .vision_analytics import VisionAnalyzer
from .chunking import merge_and_chunk, clean_content
from .embeddings import Embedder
from .retrieval import RAG


def list_pdfs(pdf_dir: str) -> List[str]:
    files = []
    for item in os.listdir(pdf_dir):
        fpath = os.path.join(pdf_dir, item)
        if os.path.isfile(fpath) and item.lower().endswith('.pdf'):
            files.append(fpath)
    if not files:
        print(f"[yellow]No PDFs found in {pdf_dir}. Place real PDFs there and try again.[/yellow]")
    return files


def run_pipeline(pdf_dir: str | None = None, skip_vision: bool = False, skip_text: bool = False) -> None:
    settings = load_settings()
    if pdf_dir is None:
        pdf_dir = settings.pdf_dir

    # Collect files
    files = list_pdfs(pdf_dir)
    docs: List[Dict[str, Any]] = []

    # Initialize clients
    analyzer = VisionAnalyzer(api_key=settings.openai_api_key, model=settings.chat_model)

    print(f"[bold]Processing {len(files)} PDF(s) from {pdf_dir}[/bold]")
    for f in files:
        print(f"[cyan]Processing:[/cyan] {os.path.basename(f)}")
        doc: Dict[str, Any] = {"filename": os.path.basename(f)}

        if not skip_text and settings.include_text_extraction:
            try:
                text = extract_text_from_doc(f)
            except Exception as e:
                print(f"[yellow]Text extraction failed: {e}[/yellow]")
                text = ""
            doc["text"] = text
        else:
            doc["text"] = ""

        # Convert to images
        try:
            images = convert_doc_to_images(f)
        except Exception as e:
            print(f"[red]Image conversion failed for {f}: {e}[/red]")
            images = []

        # Remove first page if set
        if settings.remove_first_page and len(images) > 1:
            images_for_vision = images[1:]
        else:
            images_for_vision = images

        pages_description: List[str] = []
        if not skip_vision and images_for_vision:
            print("Analyzing pages with GPT-4o...")
            for img in tqdm(images_for_vision):
                try:
                    desc = analyzer.analyze_image(img)
                except Exception as e:
                    desc = f""
                    print(f"[yellow]Vision analysis failed for one page: {e}[/yellow]")
                pages_description.append(desc)
        doc["pages_description"] = pages_description
        docs.append(doc)

    # Persist
    os.makedirs(settings.data_dir, exist_ok=True)
    with open(settings.parsed_json_path, 'w') as f:
        json.dump(docs, f)
    print(f"[green]Saved parsed docs to {settings.parsed_json_path}[/green]")

    # Chunk and clean
    pieces = merge_and_chunk(docs, remove_first_page=settings.remove_first_page)
    cleaned = clean_content(pieces)

    # Embeddings
    embedder = Embedder(api_key=settings.openai_api_key, model=settings.embedding_model)
    print("Creating embeddings (this can take a while)...")
    df = embedder.build_df(cleaned)
    embedder.save_df(df, settings.embeddings_csv_path)
    print(f"[green]Saved embeddings to {settings.embeddings_csv_path}[/green]")


def run_query(query: str, top_k: int = 3) -> None:
    settings = load_settings()
    embedder = Embedder(api_key=settings.openai_api_key, model=settings.embedding_model)
    try:
        df = embedder.load_df(settings.embeddings_csv_path)
    except FileNotFoundError:
        print(f"[red]Embeddings file not found at {settings.embeddings_csv_path}. Run the pipeline first.[/red]")
        return

    rag = RAG(api_key=settings.openai_api_key, chat_model=settings.chat_model)
    ranked = rag.search(df, query, embedder)
    top = ranked.head(top_k)

    print("[grey37][b]Matching content:[/b][/grey37]")
    for _, row in top.iterrows():
        sim = row["similarity"]
        if isinstance(sim, (list, tuple)):
            sim = sim[0][0]
        elif hasattr(sim, "shape"):
            sim = float(sim[0][0])
        print(f"[grey37][i]Similarity: {sim:.2f}[/i][/grey37]")
        content_preview = row["content"][:120]
        if len(row["content"]) > 120:
            content_preview += "..."
        print(f"[grey37]{content_preview}[/grey37]\n")

    reply = rag.generate(query, top)
    print(f"[turquoise4][b]REPLY:[/b][/turquoise4]\n\n[spring_green4]{reply}[/spring_green4]")


def main():
    parser = argparse.ArgumentParser(description="RAG over PDF documents")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Run end-to-end pipeline on a directory of PDFs")
    p_run.add_argument("--pdf_dir", type=str, default=None, help="Directory containing PDF files")
    p_run.add_argument("--skip_vision", action="store_true", help="Skip GPT-4o image analysis")
    p_run.add_argument("--skip_text", action="store_true", help="Skip text extraction")

    p_query = sub.add_parser("query", help="Query the built embeddings store")
    p_query.add_argument("query", type=str, help="User query text")
    p_query.add_argument("--top_k", type=int, default=3)

    args = parser.parse_args()

    if args.cmd == "run":
        run_pipeline(pdf_dir=args.pdf_dir, skip_vision=args.skip_vision, skip_text=args.skip_text)
    elif args.cmd == "query":
        run_query(args.query, top_k=args.top_k)


if __name__ == "__main__":
    main()
