"""
evaluate.py — Evalúa la recuperación con el golden set: Recall@5 y Precision@5.

Para cada pregunta de golden_set.json se recuperan los top-5 fragmentos en tres
modos (solo vectorial, solo BM25 e híbrido) y se calcula:

- Recall@5:    1 si el documento esperado aparece entre los 5 fragmentos, 0 si no.
               (Hay un solo documento correcto por pregunta, así que es "lo encontró o no".)
- Precision@5: qué proporción de los 5 fragmentos pertenece al documento esperado.
               Ejemplo: 3 de 5 fragmentos son de "settings" -> 3/5 = 0.60.

Al final imprime una tabla por pregunta y los promedios por modo.

Se ejecuta con:  python evaluate.py
"""

import config  # noqa: F401  (primero)

import json
import sys

from config import GOLDEN_SET, TOP_K, ConfigError
from pinecone_utils import IndiceError
from rag_system import MODOS, RAGSystem

NOMBRES = {"vectorial": "Vectorial", "bm25": "BM25", "hibrido": "Híbrido"}


def cargar_golden_set() -> list[dict]:
    """Lee y valida golden_set.json."""
    try:
        preguntas = json.loads(GOLDEN_SET.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ConfigError(f"No existe {GOLDEN_SET.name}")
    except json.JSONDecodeError as error:
        raise ConfigError(f"{GOLDEN_SET.name} no es un JSON válido: {error}")
    for n, p in enumerate(preguntas, start=1):
        if not {"pregunta", "documento_id_esperado"} <= p.keys():
            raise ConfigError(f"La pregunta {n} del golden set no tiene 'pregunta' y 'documento_id_esperado'")
    return preguntas


def recall_at_k(doc_ids: list[str], esperado: str) -> float:
    return 1.0 if esperado in doc_ids else 0.0


def precision_at_k(doc_ids: list[str], esperado: str, k: int) -> float:
    return sum(d == esperado for d in doc_ids) / k


def main() -> None:
    preguntas = cargar_golden_set()
    rag = RAGSystem()

    # Avisa si el golden set apunta a un documento que no está indexado (error de tipeo).
    indexados = {c.metadata["doc_id"] for c in rag.chunks}
    for p in preguntas:
        if p["documento_id_esperado"] not in indexados:
            raise ConfigError(f"'{p['documento_id_esperado']}' no está en el índice. Documentos: {sorted(indexados)}")

    # resultados[modo] = lista de (recall, precision, doc_ids) por pregunta
    resultados: dict[str, list[tuple[float, float, list[str]]]] = {m: [] for m in MODOS}
    for p in preguntas:
        for modo in MODOS:
            doc_ids = [r.doc_id for r in rag.retrieve(p["pregunta"], k=TOP_K, modo=modo)]
            esperado = p["documento_id_esperado"]
            resultados[modo].append((recall_at_k(doc_ids, esperado), precision_at_k(doc_ids, esperado, TOP_K), doc_ids))

    imprimir_reporte(preguntas, resultados)


def imprimir_reporte(preguntas: list[dict], resultados: dict) -> None:
    linea = "=" * 92
    print(f"\n{linea}\nEVALUACIÓN DE RECUPERACIÓN — {len(preguntas)} preguntas, k = {TOP_K}\n{linea}")

    # Detalle por pregunta: qué documentos trajo cada modo.
    for n, p in enumerate(preguntas, start=1):
        print(f"\nP{n}. {p['pregunta']}")
        print(f"    Esperado: {p['documento_id_esperado']}   ({p.get('tipo', '')})")
        for modo in MODOS:
            r, pr, doc_ids = resultados[modo][n - 1]
            marcados = ", ".join(f"[{d}]" if d == p["documento_id_esperado"] else d for d in doc_ids)
            print(f"    {NOMBRES[modo]:<9} R={r:.0f} P={pr:.2f} -> {marcados}")

    # Tabla resumen.
    print(f"\n{linea}\nTABLA RESUMEN (R = Recall@{TOP_K}, P = Precision@{TOP_K})\n{linea}")
    encabezado = f"{'Pregunta':<10}" + "".join(f"| {NOMBRES[m] + ' R':>11} {NOMBRES[m] + ' P':>11} " for m in MODOS)
    print(encabezado)
    print("-" * len(encabezado))
    for n in range(len(preguntas)):
        fila = f"{'P' + str(n + 1):<10}"
        for modo in MODOS:
            r, pr, _ = resultados[modo][n]
            fila += f"| {r:>11.2f} {pr:>11.2f} "
        print(fila)
    print("-" * len(encabezado))
    fila = f"{'PROMEDIO':<10}"
    for modo in MODOS:
        recalls = [r for r, _, _ in resultados[modo]]
        precisiones = [pr for _, pr, _ in resultados[modo]]
        fila += f"| {sum(recalls) / len(recalls):>11.2f} {sum(precisiones) / len(precisiones):>11.2f} "
    print(fila)
    print("\n[doc] = fragmento del documento esperado")


if __name__ == "__main__":
    try:
        main()
    except (ConfigError, IndiceError) as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
