RAG over PDF documents with OpenAI (productionized from a notebook)

Overview
This project converts the provided exploratory Jupyter notebook into a runnable, debuggable Python application suitable for real-world use.

It processes PDF documents in two ways:
- Extract text using pdfminer.six
- Convert pages to images and analyze them with GPT-4o vision

Then it cleans and chunks content, generates embeddings using text-embedding-3-small, and runs a minimal retrieval-augmented generation (RAG) query flow.

Structure
- src/rag_pdf/
  - config.py: Configuration handling (API key, model names, toggles)
  - pdf_processing.py: PDF text extraction and image conversion
  - vision_analytics.py: GPT-4o image analysis
  - chunking.py: Content merging and cleanup
  - embeddings.py: Embedding generation and local CSV persistence
  - retrieval.py: Similarity search and RAG answer generation
  - cli.py: Command line interface to run end-to-end pipeline
  - utils.py: Helpers (base64 image conversion, concurrency utilities)
- data/
  - example_pdfs/: Place your real PDFs here
  - parsed_pdf_docs.json: Intermediate output (pages + text)
  - parsed_pdf_docs_with_embeddings.csv: Final dataset with embeddings

Quick start
1) Put real PDFs into data/example_pdfs/ (no placeholders)
2) Install dependencies and set your OpenAI API Key
   - pip install -r requirements.txt
   - export OPENAI_API_KEY=your_key_here  # Or set via .env
3) Run the pipeline
   - python -m src.rag_pdf.cli run --pdf_dir data/example_pdfs
4) Ask queries (after embeddings are built)
   - python -m src.rag_pdf.cli query "Which embedding model should I use for non-English use cases?"

Notes
- We avoid long-running blocking servers unless requested. This CLI completes and exits.
- You can safely paste your API key via environment variable; the app never logs the key.
- The pipeline stores outputs in data/ and can be re-run incrementally.
