"""
config.py
---------
Single source of truth for configuration, credentials, and shared client
factories (Pinecone index, Hugging Face local embeddings, and a swappable
LLM provider for grading/rewriting/generation).

Every other module in this project (ingest.py, agent.py, app.py) imports
from here instead of re-reading environment variables or re-instantiating
clients. In particular, `get_llm()` is a single provider-agnostic factory --
switching between OpenAI, Gemini, and Groq is a one-variable change here
(LLM_PROVIDER), with zero changes needed anywhere else in the project.
"""

import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

# Load variables from a local .env file
load_dotenv()

# --------------------------------------------------------------------------
# Credentials
# --------------------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # for Gemini
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
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

# --------------------------------------------------------------------------
# LLM provider selection (grading / rewriting / generation -- NOT embeddings,
# those are always local, see above).
#
# Set LLM_PROVIDER in your .env to one of: "openai", "gemini", "groq".
# This is the ONLY place the provider choice lives -- agent.py just calls
# get_llm() and doesn't know or care which provider is behind it.
#
#   openai -> needs OPENAI_API_KEY   (paid; what you started with)
#   gemini -> needs GOOGLE_API_KEY   (free tier: no credit card, ~1000+ req/day
#             on gemini-2.5-flash-lite -- recommended default if you're on a
#             free budget, since this agent makes several LLM calls per
#             question: one grader call per retrieved chunk, plus rewrite,
#             generate, and hallucination-check calls)
#   groq   -> needs GROQ_API_KEY     (free tier: no credit card, very fast
#             inference on open-weight models like Llama 3.3)
# --------------------------------------------------------------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
# "gemini-flash-lite-latest" is Google's auto-updating alias, not a pinned
# dated model. Google frequently retires specific dated models (sometimes
# only for NEW API keys, while existing keys keep working -- which is why
# this can 404 differently for different people on the exact same code).
# The "-latest" alias is Google's own recommended way to avoid this: it
# always points at their current best lite model, with a 2-week email
# notice before anything changes underneath it. Pin a dated model (e.g.
# "gemini-3.1-flash-lite") only if you specifically need version stability
# for reproducibility and are OK manually updating it when it's retired.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

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


_LLM_PROVIDER_KEY_REQUIREMENTS = {
    "openai": ("OPENAI_API_KEY", OPENAI_API_KEY),
    "gemini": ("GOOGLE_API_KEY", GOOGLE_API_KEY),
    "groq": ("GROQ_API_KEY", GROQ_API_KEY),
}


def _validate_env() -> None:
    """Fail fast and loudly if required secrets are missing.

    Only the API key for the currently-selected LLM_PROVIDER is required --
    e.g. if LLM_PROVIDER=gemini, OPENAI_API_KEY is NOT required. This means
    switching providers never requires you to keep an unused key around.
    """
    if LLM_PROVIDER not in _LLM_PROVIDER_KEY_REQUIREMENTS:
        raise EnvironmentError(
            f"Unknown LLM_PROVIDER='{LLM_PROVIDER}'. Must be one of: "
            f"{', '.join(_LLM_PROVIDER_KEY_REQUIREMENTS)}."
        )
    llm_key_name, llm_key_val = _LLM_PROVIDER_KEY_REQUIREMENTS[LLM_PROVIDER]

    missing = [
        name
        for name, val in [
            (llm_key_name, llm_key_val),
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


def get_llm(temperature: float = 0.0):
    """
    Return the chat model used for grading, rewriting, and generation.

    Which provider you get back is controlled entirely by LLM_PROVIDER
    (see the "Model configuration" section above) -- agent.py and every
    other caller are completely unaware of which one is active. Low
    temperature (default 0.0) is intentional: grader/router nodes need
    deterministic, consistent binary decisions, not creative variance.
    """
    if LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=OPENAI_MODEL, temperature=temperature, api_key=OPENAI_API_KEY)

    if LLM_PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=GEMINI_MODEL, temperature=temperature, google_api_key=GOOGLE_API_KEY
        )

    if LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(model=GROQ_MODEL, temperature=temperature, api_key=GROQ_API_KEY)

    raise EnvironmentError(
        f"Unknown LLM_PROVIDER='{LLM_PROVIDER}'. Must be one of: openai, gemini, groq."
    )


def get_vectorstore(index=None) -> PineconeVectorStore:
    """Wrap a raw Pinecone index in a LangChain-compatible vector store."""
    if index is None:
        index = get_or_create_index()
    return PineconeVectorStore(index=index, embedding=get_embeddings())
