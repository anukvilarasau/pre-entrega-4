"""
ingest.py — Ingesta: documentos -> fragmentos -> embeddings -> Pinecone.

Flujo:
    data/**/*.md|.pdf|.json  ->  clean_text()                       (loaders.py)
                             ->  fragmentos de <= 600 tokens, overlap 80 (tiktoken)
                             ->  metadata validada con ChunkMetadata (Pydantic)
                             ->  embeddings all-MiniLM-L6-v2 (384)
                             ->  upsert por lotes de 100 en el namespace del .env
                             ->  describe_index_stats()

IDs determinísticos: el ID de cada vector es "<doc_id>:p<página>:c<n° de fragmento>".
Si se vuelve a correr, Pinecone SOBRESCRIBE esos mismos IDs en lugar de duplicarlos.

Se ejecuta con:  python ingest.py
(antes: python setup_index.py)
"""

import config  # noqa: F401  (primero)

import sys
import time
from collections import Counter
from datetime import datetime, timezone

import tiktoken
from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from pydantic import ValidationError

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    INDEX_NAME,
    PINECONE_NAMESPACE,
    UPSERT_BATCH,
    ConfigError,
)
from loaders import DatasetError, cargar_documentos
from pinecone_utils import IndiceError, abrir_indice, get_embeddings
from schemas import ChunkMetadata

ENCODING = "cl100k_base"  # tokenizador de tiktoken con el que se mide el tamaño


# ---------------------------------------------------------------------------
# 1. FRAGMENTACIÓN
# ---------------------------------------------------------------------------
def fragmentar(documentos: list[Document]) -> list[Document]:
    """Corta los documentos en fragmentos de como máximo CHUNK_SIZE tokens.

    Se usan los separadores de Markdown: el splitter intenta cortar primero por
    títulos (##, ###), después por bloques de código, párrafos, líneas y, como
    último recurso, palabras. Así un fragmento suele ser una sección entera.
    """
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=ENCODING,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=RecursiveCharacterTextSplitter.get_separators_for_language(Language.MARKDOWN),
    )
    return splitter.split_documents(documentos)


# ---------------------------------------------------------------------------
# 2. METADATA ESTRICTA E IDs DETERMINÍSTICOS
# ---------------------------------------------------------------------------
def armar_metadata(fragmentos: list[Document]) -> list[ChunkMetadata]:
    """Valida la metadata de cada fragmento con Pydantic y le asigna su chunk_id."""
    ingested_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    contador: Counter[tuple[str, int]] = Counter()
    metadatas = []
    for fragmento in fragmentos:
        m = fragmento.metadata
        clave = (m["doc_id"], m["page"])
        chunk_id = f"{m['doc_id']}:p{m['page']}:c{contador[clave]:03d}"
        contador[clave] += 1
        try:
            metadatas.append(ChunkMetadata(
                doc_id=m["doc_id"],
                source=m["source"],
                page=m["page"],
                category=m["category"],
                chunk_id=chunk_id,
                text=fragmento.page_content,  # el texto original va DENTRO de la metadata
                ingested_at=ingested_at,
            ))
        except ValidationError as error:
            raise DatasetError(f"Metadata inválida en el fragmento {chunk_id}:\n{error}") from error
    return metadatas


# ---------------------------------------------------------------------------
# 3. EMBEDDINGS + UPSERT POR LOTES
# ---------------------------------------------------------------------------
def subir(indice, metadatas: list[ChunkMetadata]) -> None:
    """Genera los embeddings y los sube a Pinecone en lotes de UPSERT_BATCH."""
    embeddings = get_embeddings()
    total = len(metadatas)
    for inicio in range(0, total, UPSERT_BATCH):
        lote = metadatas[inicio:inicio + UPSERT_BATCH]
        vectores = embeddings.embed_documents([m.text for m in lote])
        indice.upsert(
            vectors=[
                {"id": m.chunk_id, "values": v, "metadata": m.model_dump()}
                for m, v in zip(lote, vectores)
            ],
            namespace=PINECONE_NAMESPACE,  # SIEMPRE en el namespace configurado
        )
        print(f"  Lote {inicio // UPSERT_BATCH + 1}: vectores {inicio + 1}-{inicio + len(lote)} de {total} subidos")


def esperar_conteo(indice, esperado: int, espera_maxima: int = 60) -> dict:
    """Pinecone Serverless tarda unos segundos en reflejar el upsert en las estadísticas."""
    inicio = time.monotonic()
    while True:
        stats = indice.describe_index_stats()
        ns = stats.namespaces.get(PINECONE_NAMESPACE)
        if (ns and ns.vector_count >= esperado) or time.monotonic() - inicio > espera_maxima:
            return stats
        time.sleep(3)


# ---------------------------------------------------------------------------
# EJECUCIÓN
# ---------------------------------------------------------------------------
def main() -> None:
    indice = abrir_indice()  # valida: existe, está listo y la dimensión es 384

    documentos = cargar_documentos()
    print(f"Documentos cargados: {len({d.metadata['doc_id'] for d in documentos})} archivos "
          f"({len(documentos)} páginas/ítems)")

    fragmentos = fragmentar(documentos)
    metadatas = armar_metadata(fragmentos)
    tokenizador = tiktoken.get_encoding(ENCODING)
    tamanos = [len(tokenizador.encode(m.text)) for m in metadatas]
    print(f"Fragmentos: {len(metadatas)} (tokens por fragmento: mín {min(tamanos)}, "
          f"promedio {sum(tamanos) // len(tamanos)}, máx {max(tamanos)}; techo {CHUNK_SIZE}, overlap {CHUNK_OVERLAP})")

    print(f"Subiendo a '{INDEX_NAME}' / namespace '{PINECONE_NAMESPACE}' con {EMBEDDING_MODEL}...")
    subir(indice, metadatas)

    stats = esperar_conteo(indice, len(metadatas))
    print("\ndescribe_index_stats():")
    print(stats)


if __name__ == "__main__":
    try:
        main()
    except (ConfigError, IndiceError, DatasetError) as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
