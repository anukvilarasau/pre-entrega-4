"""
schemas.py — Modelos Pydantic del sistema.

ChunkMetadata define la metadata ESTRICTA de cada vector que se sube a Pinecone.
Si un fragmento no cumple el modelo (falta un campo, tipo incorrecto, campo de
más), la ingesta corta ANTES de subir nada: así el índice nunca queda con
metadata incompleta o inconsistente.

Resultado es lo que devuelve RAGSystem.retrieve() por cada fragmento recuperado.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ChunkMetadata(BaseModel):
    """Metadata de un fragmento tal como se guarda en Pinecone."""

    # strict: no convierte tipos "a la fuerza" (por ej. "3" no se acepta como int).
    # extra="forbid": un campo que no está en el modelo es un error.
    model_config = ConfigDict(strict=True, extra="forbid")

    doc_id: str = Field(min_length=1, description="Identificador del documento (nombre del archivo sin extensión)")
    source: str = Field(min_length=1, description="Ruta del archivo dentro de data/, por ej. tutorial/cors.md")
    page: int = Field(ge=1, description="Página (PDF) o ítem (JSON). Los .md son una sola página: 1")
    category: str = Field(min_length=1, description="Categoría: subcarpeta de data/ (tutorial, advanced...)")
    chunk_id: str = Field(min_length=1, description="ID determinístico del fragmento; también es el ID del vector")
    text: str = Field(min_length=1, description="Texto ORIGINAL del fragmento (PineconeVectorStore lo lee de acá)")
    ingested_at: str = Field(description="Fecha y hora de ingesta en formato ISO 8601 (UTC)")

    def model_post_init(self, _contexto) -> None:
        # Pinecone solo guarda strings/números/booleanos: la fecha va como texto ISO,
        # pero validamos que sea una fecha real.
        datetime.fromisoformat(self.ingested_at)


class Resultado(BaseModel):
    """Un fragmento recuperado por RAGSystem.retrieve()."""

    posicion: int = Field(ge=1, description="Puesto en el ranking (1 = el más relevante)")
    doc_id: str
    source: str
    category: str
    page: int
    chunk_id: str
    text: str
