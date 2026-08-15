"""
config.py
---------
Single source of truth for configuration, credentials, and shared client
factories (Pinecone index, Hugging Face local embeddings, OpenAI chat model).

Every other module in this project (ingest.py, agent.py, app.py) imports
from here instead of re-reading environment variables or re-instantiating
clients.
"""

import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore

# Load variables from a local .env file
load_dotenv()

# --------------------------------------------------------------------------
# Credentials
# --------------------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# --------------------------------------------------------------------------
# Pinecone index configuration
# --------------------------------------------------------------------------
# Updated default index name to avoid dimension conflict with legacy 1536-dim index
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "crag-ops-10k-hf")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

# --------------------------------------------------------------------------
# Model configuration
# --------------------------------------------------------------------------
# Hugging Face local embedding model (free, offline, no API rate limits)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384  # Native output dimension for all-MiniLM-L6-v2
LLM_MODEL = "gpt-4o-mini"

# --------------------------------------------------------------------------
# Ingestion / retrieval tuning knobs
# --------------------------------------------------------------------------
DATA_DIR = os.getenv("DATA_DIR", "data")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 4                     # chunks pulled from Pinecone per query
MAX_GENERATION_RETRIES = 2    # hallucination self-correction loop cap

# --------------------------------------------------------------------------
# PDF text-extraction robustness knobs (see ingest.py for the full
# multi-tier extraction cascade: pypdf -> PyMuPDF -> pypdfium2 -> OCR)
# --------------------------------------------------------------------------
MIN_EXTRACTABLE_CHARS = 50     # below this, a PDF is treated as textless
ENABLE_OCR_FALLBACK = True     # last-resort tier for genuinely image-only PDFs
OCR_DPI = 200                  # render resolution for OCR (higher = slower, more accurate)

# --------------------------------------------------------------------------
# A note on the local embedding model's cache (relevant to "will restarting
# my computer force a re-download?"):
#
# HuggingFaceEmbeddings downloads model weights from the Hub ONCE and caches
# them on disk (by default under ~/.cache/huggingface on macOS/Linux, or
# C:\Users\<you>\.cache\huggingface on Windows). Every subsequent run --
# including after a full computer restart -- reuses that cache and does NOT
# re-download, as long as the cache directory itself isn't wiped. The only
# time you'll see a fresh download is the very first run ever, or after
# deploying to a brand-new/ephemeral environment (e.g. a fresh Streamlit
# Cloud container) whose disk doesn't persist between deploys -- that's a
# one-time ~90MB download on cold start, not a bug.
#
# Uncomment and set this if you want the cache in a specific location:
# os.environ.setdefault("HF_HOME", "/path/to/persistent/cache")
# --------------------------------------------------------------------------


def _validate_env() -> None:
    """Fail fast and loudly if required secrets are missing."""
    missing = [
        name
        for name, val in [
            ("OPENAI_API_KEY", OPENAI_API_KEY),
            ("PINECONE_API_KEY", PINECONE_API_KEY),
            ("TAVILY_API_KEY", TAVILY_API_KEY),
        ]
        if not val
    ]
    if missing:
        raise EnvironmentError(
            f"Missing required environment variable(s): {', '.join(missing)}. "
            "Set them in a local .env file or in Streamlit Cloud's Secrets manager."
        )


def get_pinecone_client() -> Pinecone:
    """Return an authenticated Pinecone control-plane client."""
    _validate_env()
    return Pinecone(api_key=PINECONE_API_KEY)


def get_or_create_index(index_name: str = PINECONE_INDEX_NAME):
    """
    Return a handle to the Pinecone index, creating it as a Serverless
    index if it does not already exist with the 384-dimension spec.
    """
    pc = get_pinecone_client()

    if not pc.has_index(index_name):
        pc.create_index(
            name=index_name,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )

    return pc.Index(index_name)


def get_embeddings() -> HuggingFaceEmbeddings:
    """Return the local HuggingFace embeddings client used for ingestion and retrieval."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},  # Change to "cuda" if running on an NVIDIA GPU
        encode_kwargs={"normalize_embeddings": True},
    )


def get_llm(temperature: float = 0.0) -> ChatOpenAI:
    """Return the chat model used for grading, rewriting, and generation."""
    return ChatOpenAI(model=LLM_MODEL, temperature=temperature, api_key=OPENAI_API_KEY)


def get_vectorstore(index=None) -> PineconeVectorStore:
    """Wrap a raw Pinecone index in a LangChain-compatible vector store."""
    if index is None:
        index = get_or_create_index()
    return PineconeVectorStore(index=index, embedding=get_embeddings())