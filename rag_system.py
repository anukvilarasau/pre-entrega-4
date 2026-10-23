"""
rag_system.py — Recuperación híbrida: Pinecone (semántica) + BM25 (léxica).

    pregunta ─┬─> PineconeVectorStore (embeddings all-MiniLM-L6-v2) ─ top-k ─┐
              │                                                              ├─> EnsembleRetriever ─> top-k
              └─> BM25Retriever (palabras exactas)                ─ top-k ─┘    (Reciprocal Rank Fusion)

- Semántica: encuentra fragmentos que "dicen lo mismo con otras palabras".
- BM25: encuentra fragmentos que contienen las palabras EXACTAS de la pregunta
  (nombres de clases, parámetros, códigos: allow_origins, HTTPException, 422...).
- EnsembleRetriever combina los dos rankings con Reciprocal Rank Fusion: un
  fragmento que aparece arriba en ambas listas sube; los repetidos se unifican
  por chunk_id.

BM25 se arma sobre los MISMOS fragmentos que están en Pinecone: se leen del
propio índice (la metadata guarda el texto original), no se vuelven a cortar.

Uso:
    rag = RAGSystem()
    for r in rag.retrieve("How do I enable CORS?", k=5):
        print(r.posicion, r.doc_id, r.source)
"""

import config  # noqa: F401  (primero)

import re
import sys
from typing import Literal

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_pinecone import PineconeVectorStore

from config import INDEX_NAME, PESOS_HIBRIDO, PINECONE_NAMESPACE, TOP_K, ConfigError
from pinecone_utils import IndiceError, abrir_indice, get_embeddings
from schemas import Resultado

Modo = Literal["vectorial", "bm25", "hibrido"]
MODOS: tuple[Modo, ...] = ("vectorial", "bm25", "hibrido")


def tokenizar(texto: str) -> list[str]:
    """Tokenizador para BM25: minúsculas y palabras con letras, números y "_".

    El de LangChain por defecto solo separa por espacios, así "`HTTPException`,"
    (con comillas y coma) no coincidiría con "HTTPException".
    """
    return re.findall(r"\w+", texto.lower())


class RAGSystem:
    """Encapsula el retriever híbrido sobre el índice de Pinecone."""

    def __init__(self, namespace: str = PINECONE_NAMESPACE, pesos: tuple[float, float] = PESOS_HIBRIDO):
        self.namespace = namespace
        self.pesos = list(pesos)
        self.indice = abrir_indice()  # existe + listo + dimensión correcta (si no, IndiceError)

        # Mismo modelo de embeddings que en ingest.py (viene de pinecone_utils).
        self.vectorstore = PineconeVectorStore(
            index=self.indice,
            embedding=get_embeddings(),
            text_key="text",          # el texto original está en metadata["text"]
            namespace=self.namespace,
        )

        self.chunks = self._leer_chunks_del_indice()
        if not self.chunks:
            raise IndiceError(
                f"El namespace '{self.namespace}' del índice '{INDEX_NAME}' está vacío. "
                f"Cargá los documentos con: python ingest.py"
            )

    # -----------------------------------------------------------------------
    # Fragmentos para BM25 (leídos de Pinecone)
    # -----------------------------------------------------------------------
    def _leer_chunks_del_indice(self) -> list[Document]:
        """Trae todos los fragmentos del namespace: lista los IDs y los pide en lotes de 100."""
        ids = [i for pagina in self.indice.list(namespace=self.namespace) for i in pagina]
        chunks = []
        for inicio in range(0, len(ids), 100):
            respuesta = self.indice.fetch(ids=ids[inicio:inicio + 100], namespace=self.namespace)
            for vector in respuesta.vectors.values():
                metadata = dict(vector.metadata)
                texto = metadata.pop("text")
                chunks.append(Document(page_content=texto, metadata=metadata))
        # Orden estable (el orden de fetch no está garantizado): BM25 desempata por posición.
        return sorted(chunks, key=lambda d: d.metadata["chunk_id"])

    # -----------------------------------------------------------------------
    # Retrievers
    # -----------------------------------------------------------------------
    def _retriever_vectorial(self, k: int, category: str | None) -> BaseRetriever:
        busqueda = {"k": k}
        if category:
            busqueda["filter"] = {"category": {"$eq": category}}  # el filtro lo aplica Pinecone
        return self.vectorstore.as_retriever(search_kwargs=busqueda)

    def _retriever_bm25(self, k: int, category: str | None) -> BM25Retriever:
        chunks = self.chunks
        if category:
            chunks = [c for c in chunks if c.metadata["category"] == category]
            if not chunks:
                raise ValueError(f"No hay fragmentos con category='{category}'")
        return BM25Retriever.from_documents(chunks, k=k, preprocess_func=tokenizar)

    def _retriever(self, modo: Modo, k: int, category: str | None) -> BaseRetriever:
        if modo == "vectorial":
            return self._retriever_vectorial(k, category)
        if modo == "bm25":
            return self._retriever_bm25(k, category)
        if modo == "hibrido":
            return EnsembleRetriever(
                retrievers=[self._retriever_vectorial(k, category), self._retriever_bm25(k, category)],
                weights=self.pesos,
                id_key="chunk_id",  # los fragmentos repetidos se unifican por su ID
            )
        raise ValueError(f"Modo desconocido: {modo!r}. Opciones: {', '.join(MODOS)}")

    # -----------------------------------------------------------------------
    # API pública
    # -----------------------------------------------------------------------
    def retrieve(self, query: str, k: int = TOP_K, modo: Modo = "hibrido",
                 category: str | None = None) -> list[Resultado]:
        """Devuelve los top-k fragmentos con su doc_id y su fuente.

        modo:     "hibrido" (por defecto), "vectorial" o "bm25" (para comparar en evaluate.py).
        category: filtro opcional por metadata, por ejemplo "tutorial" o "advanced".
        """
        if not query.strip():
            raise ValueError("La consulta está vacía")
        documentos = self._retriever(modo, k, category).invoke(query)[:k]
        return [
            Resultado(
                posicion=n,
                doc_id=d.metadata["doc_id"],
                source=d.metadata["source"],
                category=d.metadata["category"],
                page=int(d.metadata["page"]),  # Pinecone devuelve los números como float
                chunk_id=d.metadata["chunk_id"],
                text=d.page_content,
            )
            for n, d in enumerate(documentos, start=1)
        ]


# ---------------------------------------------------------------------------
# Prueba rápida:  python rag_system.py "tu pregunta" [categoria]
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    pregunta = sys.argv[1] if len(sys.argv) > 1 else "How do I allow requests from another origin (CORS)?"
    categoria = sys.argv[2] if len(sys.argv) > 2 else None
    try:
        rag = RAGSystem()
    except (ConfigError, IndiceError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
    print(f"Fragmentos en el namespace '{rag.namespace}': {len(rag.chunks)}")
    print(f"Pregunta: {pregunta}" + (f"  (filtro category={categoria})" if categoria else ""))
    for modo in MODOS:
        print(f"\n[{modo}]")
        for r in rag.retrieve(pregunta, k=5, modo=modo, category=categoria):
            print(f"  {r.posicion}. {r.doc_id:<32} {r.source:<40} {r.chunk_id}")
