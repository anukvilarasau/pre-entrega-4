"""
pinecone_utils.py — Piezas compartidas por setup_index.py, ingest.py y rag_system.py.

- get_embeddings():     el modelo de embeddings (UNO solo para indexar y consultar).
- dimension_modelo():   la dimensión que produce ese modelo (384 para all-MiniLM-L6-v2).
- cliente_pinecone():   conexión a Pinecone con la API key del .env.
- abrir_indice():       verifica que el índice exista, esté listo y tenga la
                        dimensión correcta; si no, corta con un mensaje claro.
"""

import config  # noqa: F401  (primero: variables de entorno antes de torch)

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from config import EMBEDDING_DEVICE, EMBEDDING_MODEL, INDEX_NAME, PINECONE_API_KEY, requerir


class IndiceError(RuntimeError):
    """El índice no existe, no está listo o no es compatible con el modelo de embeddings."""


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Modelo de embeddings local. Se carga una sola vez (la primera vez baja ~90 MB).

    normalize_embeddings=True deja los vectores con largo 1, que es lo natural para
    la métrica cosine del índice.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": EMBEDDING_DEVICE},
        encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache(maxsize=1)
def dimension_modelo() -> int:
    """Dimensión REAL del modelo: se mide generando un embedding, no se escribe a mano."""
    return len(get_embeddings().embed_query("dimension"))


def cliente_pinecone():
    """Devuelve el cliente de Pinecone. Corta si falta la API key o si es inválida."""
    requerir("PINECONE_API_KEY", "INDEX_NAME", "PINECONE_NAMESPACE")
    from pinecone import Pinecone
    from pinecone.exceptions import UnauthorizedException

    cliente = Pinecone(api_key=PINECONE_API_KEY)
    try:
        cliente.list_indexes()  # llamada liviana para validar la clave
    except UnauthorizedException as error:
        raise IndiceError(
            "Pinecone rechazó la API key (401). Revisá PINECONE_API_KEY en el .env."
        ) from error
    return cliente


def validar_dimension(dimension_indice: int) -> None:
    """El error clásico: un índice creado con otra dimensión que la del modelo."""
    esperada = dimension_modelo()
    if dimension_indice != esperada:
        raise IndiceError(
            f"El índice '{INDEX_NAME}' tiene dimensión {dimension_indice}, pero el modelo "
            f"{EMBEDDING_MODEL} genera vectores de {esperada}. No son compatibles.\n"
            f"Solución: usá otro INDEX_NAME en el .env (o borrá ese índice en la consola de "
            f"Pinecone) y corré de nuevo: python setup_index.py"
        )


def abrir_indice(cliente=None):
    """Devuelve el índice listo para usar, después de validar que sea el correcto."""
    cliente = cliente or cliente_pinecone()
    if not cliente.has_index(INDEX_NAME):
        raise IndiceError(
            f"El índice '{INDEX_NAME}' no existe todavía. Crealo con: python setup_index.py"
        )
    descripcion = cliente.describe_index(INDEX_NAME)
    if not descripcion.status["ready"]:
        raise IndiceError(
            f"El índice '{INDEX_NAME}' existe pero todavía no está listo "
            f"(estado: {descripcion.status['state']}). Esperá unos segundos y probá de nuevo."
        )
    validar_dimension(descripcion.dimension)
    return cliente.Index(INDEX_NAME)
