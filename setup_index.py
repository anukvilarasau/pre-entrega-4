"""
setup_index.py — Crea (si hace falta) el índice Serverless de Pinecone.

1. Mide la dimensión del modelo de embeddings (384 para all-MiniLM-L6-v2).
2. Si el índice INDEX_NAME no existe, lo crea en modo Serverless
   (aws / us-east-1, métrica cosine) con esa dimensión.
3. Espera a que esté listo (status "ready").
4. Si ya existía, valida que su dimensión coincida con la del modelo: si no
   coincide, corta con un mensaje claro (subir vectores de 384 a un índice de
   1536 falla, y consultar mezclando modelos da resultados sin sentido).

Se puede correr las veces que quieras: si el índice ya está bien, no toca nada.

Se ejecuta con:  python setup_index.py
"""

import config  # noqa: F401  (primero)

import sys
import time

from config import EMBEDDING_MODEL, INDEX_NAME, METRICA, PINECONE_CLOUD, PINECONE_REGION, ConfigError
from pinecone_utils import IndiceError, cliente_pinecone, dimension_modelo, validar_dimension

ESPERA_MAXIMA_SEGUNDOS = 180


def esperar_listo(cliente, nombre: str) -> None:
    """Consulta el estado del índice cada 2 segundos hasta que esté listo."""
    inicio = time.monotonic()
    while True:
        estado = cliente.describe_index(nombre).status
        if estado["ready"]:
            print(f"  Índice listo ({time.monotonic() - inicio:.0f} s).")
            return
        if time.monotonic() - inicio > ESPERA_MAXIMA_SEGUNDOS:
            raise IndiceError(
                f"El índice '{nombre}' no quedó listo en {ESPERA_MAXIMA_SEGUNDOS} s "
                f"(estado: {estado['state']}). Revisá la consola de Pinecone y volvé a correr este script."
            )
        print(f"  Esperando... estado: {estado['state']}")
        time.sleep(2)


def main() -> None:
    from pinecone import ServerlessSpec

    print(f"Modelo de embeddings: {EMBEDDING_MODEL}")
    dimension = dimension_modelo()
    print(f"Dimensión del modelo: {dimension}")

    cliente = cliente_pinecone()

    if cliente.has_index(INDEX_NAME):
        print(f"El índice '{INDEX_NAME}' ya existe: se valida.")
        descripcion = cliente.describe_index(INDEX_NAME)
        validar_dimension(descripcion.dimension)
        if descripcion.metric != METRICA:
            print(f"  AVISO: la métrica del índice es '{descripcion.metric}' (se esperaba '{METRICA}').")
    else:
        print(f"Creando el índice '{INDEX_NAME}' (Serverless {PINECONE_CLOUD}/{PINECONE_REGION}, "
              f"{METRICA}, dimensión {dimension})...")
        cliente.create_index(
            name=INDEX_NAME,
            dimension=dimension,
            metric=METRICA,
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )

    esperar_listo(cliente, INDEX_NAME)
    d = cliente.describe_index(INDEX_NAME)
    print(f"\nOK -> índice '{d.name}' | dimensión {d.dimension} | métrica {d.metric} | host {d.host}")


if __name__ == "__main__":
    try:
        main()
    except (ConfigError, IndiceError) as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
