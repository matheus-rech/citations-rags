import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load .env if present
load_dotenv()


def getenv_strict(name: str) -> str:
    val = os.getenv(name)
    if val is None or val.strip() == "":
        raise RuntimeError(f"Environment variable {name} is required but not set. See README.md")
    return val


def getenv_default(name: str, default: str) -> str:
    val = os.getenv(name)
    return val if val not in (None, "") else default


def getenv_optional(name: str) -> str | None:
    val = os.getenv(name)
    return val if val not in (None, "") else None


@dataclass
class Settings:
    # API & models
    openai_api_key: str
    chat_model: str = getenv_default("OPENAI_CHAT_MODEL", "gpt-4o")
    embedding_model: str = getenv_default("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    anthropic_api_key: str | None = getenv_optional("ANTHROPIC_API_KEY")
    anthropic_model: str = getenv_default("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")

    # Paths
    data_dir: str = getenv_default("DATA_DIR", "data")
    pdf_dir: str = getenv_default("PDF_DIR", "data/example_pdfs")
    parsed_json_path: str = getenv_default("PARSED_JSON_PATH", "data/parsed_pdf_docs.json")
    embeddings_csv_path: str = getenv_default("EMBEDDINGS_CSV_PATH", "data/parsed_pdf_docs_with_embeddings.csv")

    # Execution
    max_workers: int = int(getenv_default("MAX_WORKERS", "8"))
    include_text_extraction: bool = getenv_default("INCLUDE_TEXT_EXTRACTION", "true").lower() == "true"
    remove_first_page: bool = getenv_default("REMOVE_FIRST_PAGE", "true").lower() == "true"


def load_settings() -> Settings:
    # Presence of OpenAI key is mandatory
    getenv_strict("OPENAI_API_KEY")
    return Settings(openai_api_key=os.environ["OPENAI_API_KEY"])
