"""
download_docs.py — Descarga el dataset: 16 páginas de la documentación oficial de FastAPI.

Baja los .md desde el repositorio de GitHub de FastAPI, fijado a un commit
concreto (así el dataset es siempre el mismo, aunque la doc cambie después), y
los guarda en ./data/<categoria>/<nombre>.md.

Las páginas originales no traen el código de ejemplo adentro: usan marcadores
como  {* ../../docs_src/handling_errors/tutorial001_py310.py hl[11] *}.
Este script reemplaza cada marcador por el código real en un bloque ```python,
para que cada página quede completa (y BM25 pueda encontrar nombres exactos).

Se ejecuta con:  python download_docs.py
"""

import config  # noqa: F401  (primero: consola en UTF-8)

import re
import sys
import urllib.error
import urllib.request

from config import DATA_DIR

# Commit fijo del repositorio fastapi/fastapi (rama master al 22/10/2026).
COMMIT = "69ddc2badf9da1123bba4d360225ae6b085985f6"
RAW = f"https://raw.githubusercontent.com/fastapi/fastapi/{COMMIT}"

# Páginas elegidas, agrupadas por categoría (la categoría va a la metadata).
PAGINAS = {
    "tutorial": [
        "first-steps",
        "path-params",
        "query-params",
        "query-params-str-validations",
        "body",
        "response-model",
        "handling-errors",
        "background-tasks",
        "cors",
        "testing",
    ],
    "advanced": [
        "settings",
        "websockets",
        "custom-response",
        "events",
        "middleware",
        "additional-status-codes",
    ],
}

# {* ../../docs_src/carpeta/archivo.py hl[1,3] *}  ->  captura la ruta dentro de docs_src
MARCADOR_CODIGO = re.compile(r"\{\*\s*(?:\.\./)+(docs_src/\S+?\.py)[^*]*\*\}")


def bajar(url: str) -> str:
    """Descarga un archivo de texto. Corta con un mensaje claro si falla."""
    try:
        with urllib.request.urlopen(url, timeout=30) as respuesta:
            return respuesta.read().decode("utf-8")
    except urllib.error.URLError as error:
        raise RuntimeError(f"No se pudo descargar {url}: {error}") from error


def insertar_codigo(texto: str) -> str:
    """Reemplaza cada marcador {* ... *} por el código real del ejemplo."""

    def reemplazo(coincidencia: re.Match) -> str:
        ruta = coincidencia.group(1)
        try:
            codigo = bajar(f"{RAW}/{ruta}").strip()
        except RuntimeError:
            return ""  # si un ejemplo no existe, se omite (no frena la descarga)
        return f"```python\n{codigo}\n```"

    return MARCADOR_CODIGO.sub(reemplazo, texto)


def main() -> None:
    total = 0
    for categoria, nombres in PAGINAS.items():
        carpeta = DATA_DIR / categoria
        carpeta.mkdir(parents=True, exist_ok=True)
        for nombre in nombres:
            texto = bajar(f"{RAW}/docs/en/docs/{categoria}/{nombre}.md")
            texto = insertar_codigo(texto)
            (carpeta / f"{nombre}.md").write_text(texto, encoding="utf-8")
            total += 1
            print(f"  [{total:2d}] {categoria}/{nombre}.md  ({len(texto):,} caracteres)")
    print(f"\nListo: {total} páginas guardadas en {DATA_DIR}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
