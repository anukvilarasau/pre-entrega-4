"""
config.py — Configuración central del sistema RAG (reutilizado de la Pre-entrega 3).

Todo lo que puede cambiar (claves, nombres, modelos, parámetros) vive acá, en UN
solo lugar. Los demás archivos importan estos valores: así es imposible indexar
con un modelo de embeddings y consultar con otro.
"""

import os

# Variables de entorno para torch / sentence-transformers. Tienen que definirse
# ANTES de importar esas librerías, por eso config se importa primero en todos lados.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # aviso inofensivo en Windows
os.environ.setdefault("HF_HUB_VERBOSITY", "error")  # sin el aviso de "unauthenticated requests"

import sys  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

from dotenv import load_dotenv  # noqa: E402

# En la consola de Windows, forzamos UTF-8 para que se vean bien las tildes y la ñ.
for _flujo in (sys.stdout, sys.stderr):
    if hasattr(_flujo, "reconfigure"):
        _flujo.reconfigure(encoding="utf-8")

# langchain-community avisa que está en retiro; BM25Retriever sigue funcionando.
warnings.filterwarnings("ignore", message=".*langchain-community.*", category=DeprecationWarning)

# ---------------------------------------------------------------------------
# RUTAS
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")  # las claves se leen del .env (que NO se sube)

DATA_DIR = BASE_DIR / "data"
GOLDEN_SET = BASE_DIR / "golden_set.json"
EXTENSIONES = (".md", ".pdf", ".json")

# ---------------------------------------------------------------------------
# PINECONE
# ---------------------------------------------------------------------------
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "").strip()
INDEX_NAME = os.getenv("INDEX_NAME", "").strip()
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "").strip()

PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"
METRICA = "cosine"
UPSERT_BATCH = 100  # vectores por lote al subir a Pinecone (la consigna pide 100-200)

# ---------------------------------------------------------------------------
# EMBEDDINGS (el MISMO modelo para indexar y para consultar)
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DEVICE = "cpu"

# ---------------------------------------------------------------------------
# CHUNKING (medido en TOKENS con tiktoken, no en caracteres)
# ---------------------------------------------------------------------------
CHUNK_SIZE = 600     # techo de tokens por fragmento (la consigna pide 500-800)
CHUNK_OVERLAP = 80   # tokens compartidos entre fragmentos vecinos

# ---------------------------------------------------------------------------
# RECUPERACIÓN
# ---------------------------------------------------------------------------
TOP_K = 5
PESOS_HIBRIDO = (0.5, 0.5)  # (vectorial, BM25) dentro del EnsembleRetriever

# ---------------------------------------------------------------------------
# GENERACIÓN (solo ask.py)
# ---------------------------------------------------------------------------
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
# La consigna pide temperatura 0. Claude Sonnet 4.6 la acepta; los modelos Claude 4.7
# y posteriores (por ej. claude-opus-5-5) la RECHAZAN con un error 400. Si cambiás a
# uno de esos, dejá TEMPERATURE vacía en el .env.
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6").strip()
_temperatura = os.getenv("TEMPERATURE", "0").strip()
TEMPERATURE: float | None = float(_temperatura) if _temperatura else None
MAX_TOKENS = 1024
NO_SE = "No tengo esa información en los documentos"


class ConfigError(RuntimeError):
    """Falta una variable obligatoria en el .env."""


def requerir(*nombres: str) -> None:
    """Corta con un mensaje claro si falta alguna variable del .env.

    Ejemplo: requerir("PINECONE_API_KEY", "INDEX_NAME")
    """
    faltantes = [n for n in nombres if not globals().get(n)]
    if faltantes:
        raise ConfigError(
            f"Faltan estas variables en el archivo .env: {', '.join(faltantes)}.\n"
            f"Copiá .env.example como .env y completalas (ver README.md)."
        )
