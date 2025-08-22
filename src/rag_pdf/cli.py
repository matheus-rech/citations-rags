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
from .schema_extractor import extract_structured_from_text
from .schemas import StudyExtraction
from .structured_pipeline import run_strict_extraction_for_pdf


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

    if top.empty:
        print("[yellow]No content available to answer the query. Run the pipeline with real PDFs first.[/yellow]")
        return

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

    p_extract = sub.add_parser("extract-json", help="Extract structured JSON matching clinical schema from a single PDF")
    p_extract.add_argument("pdf", type=str, help="Path to a single PDF file under data/")
    p_extract.add_argument("--out", type=str, default=None, help="Output JSON path (default under data/<stem>_extraction.json)")
    p_extract.add_argument("--max_chars", type=int, default=20000, help="Max chars of linearized text to feed the model")
    p_extract.add_argument("--strict", action="store_true", help="Use Responses API with two-pass provenance resolution for meta-analysis schema")

    p_harmonize = sub.add_parser("harmonize", help="Build a harmonized CSV for meta-analysis from multiple JSON extractions")
    p_harmonize.add_argument("inputs", nargs='+', help="One or more JSON extraction files under data/")
    p_harmonize.add_argument("--out", type=str, default=None, help="Output CSV path (default data/harmonized_meta.csv)")

    args = parser.parse_args()

    if args.cmd == "run":
        run_pipeline(pdf_dir=args.pdf_dir, skip_vision=args.skip_vision, skip_text=args.skip_text)
    elif args.cmd == "query":
        run_query(args.query, top_k=args.top_k)
    elif args.cmd == "extract-json":
        settings = load_settings()
        pdf_path = args.pdf
        if not os.path.isfile(pdf_path):
            # Also try relative to data dir
            candidate = os.path.join(settings.data_dir, pdf_path)
            if os.path.isfile(candidate):
                pdf_path = candidate
            else:
                print(f"[red]PDF not found: {args.pdf}[/red]")
                return
        if args.strict:
            data = run_strict_extraction_for_pdf(pdf_path, max_chars=args.max_chars)
        else:
            text = extract_text_from_doc(pdf_path)
            if not text:
                print("[yellow]Text extraction returned empty. Attempting to use prior GPT-4o page descriptions as fallback context.[/yellow]")
                # Fallback: load pages_description from parsed json if present
                try:
                    with open(settings.parsed_json_path, 'r') as f:
                        docs = json.load(f)
                    fname = os.path.basename(pdf_path)
                    for d in docs:
                        if d.get('filename') == fname and d.get('pages_description'):
                            text = "\n\n".join(d.get('pages_description'))
                            break
                except Exception:
                    pass
                if not text:
                    print("[yellow]No fallback descriptions found. The extractor may have limited context; consider running pipeline with vision first.[/yellow]")
            data = extract_structured_from_text(settings.openai_api_key, settings.chat_model, text[:args.max_chars] if text else "")
            try:
                # try strict pydantic validation for confidence
                _ = StudyExtraction.model_validate(data)
            except Exception as e:
                print(f"[yellow]Warning: pydantic validation not strict-pass: {e}[/yellow]")
        out_path = args.out
        if out_path is None:
            stem = os.path.splitext(os.path.basename(pdf_path))[0]
            out_path = os.path.join(settings.data_dir, f"{stem}_extraction.json")
        with open(out_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"[green]Saved structured extraction to {out_path}[/green]")
    elif args.cmd == "harmonize":
        settings = load_settings()
        from .harmonize import harmonize_to_csv
        ins: List[str] = []
        for p in args.inputs:
            pth = p if os.path.isabs(p) else os.path.join(settings.data_dir, p)
            if os.path.isfile(pth):
                ins.append(pth)
            else:
                print(f"[yellow]Skipping missing input: {pth}[/yellow]")
        if not ins:
            print("[red]No valid inputs provided.[/red]")
            return
        out_path = args.out or os.path.join(settings.data_dir, 'harmonized_meta.csv')
        import json as _json
        exs = []
        for pth in ins:
            with open(pth, 'r') as f:
                exs.append(_json.load(f))
        harmonize_to_csv(exs, out_path)
        print(f"[green]Wrote harmonized meta-analysis CSV to {out_path}[/green]")


if __name__ == "__main__":
    main()
