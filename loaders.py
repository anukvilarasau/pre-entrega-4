"""
loaders.py — Carga y limpieza de documentos .md, .pdf y .json (base: Pre-entrega 3).

Cada archivo de ./data se convierte en uno o más Document de LangChain con esta
metadata: doc_id, source, page y category.

- .md   -> un Document (page = 1).
- .pdf  -> un Document por página (page = número de página).
- .json -> un Document por ítem. Acepta una lista de objetos o un objeto con una
           lista adentro; de cada objeto se toma "content"/"text"/"body" (y "title"
           si existe). Un objeto suelto cuenta como un ítem.

La categoría es la subcarpeta: data/tutorial/cors.md -> category "tutorial".
Los archivos sueltos en data/ quedan con category "general".
"""

import json
import re
from collections import Counter
from pathlib import Path

from langchain_core.documents import Document

from config import DATA_DIR, EXTENSIONES

CAMPOS_TEXTO_JSON = ("content", "text", "body")


class DatasetError(RuntimeError):
    """Problema con los archivos de ./data (vacío, formato inválido, IDs repetidos...)."""


# ---------------------------------------------------------------------------
# LIMPIEZA
# ---------------------------------------------------------------------------
def clean_text(texto: str, quitar_repetidas: bool = False) -> str:
    """Normaliza el texto antes de fragmentarlo.

    - Unifica los saltos de línea (\\r\\n de Windows -> \\n).
    - Quita las anclas de los títulos de la doc de FastAPI: "## Título { #ancla }".
    - Quita las marcas de los recuadros de la doc: "/// tip", "/// note", "///".
    - Quita los espacios al final de cada línea.
    - Deja como máximo una línea en blanco entre párrafos.
    - quitar_repetidas=True (solo PDF): borra las líneas idénticas que aparecen 3+
      veces, típicamente encabezados/pies de página. En Markdown NO se usa: borraría
      líneas de código legítimas que se repiten, como "```" o "app = FastAPI()".
    """
    texto = texto.replace("\r\n", "\n").replace("\r", "\n")
    texto = re.sub(r"[ \t]*\{ #[\w-]+ \}", "", texto)
    texto = re.sub(r"^///.*$", "", texto, flags=re.MULTILINE)
    lineas = [linea.rstrip() for linea in texto.split("\n")]

    if quitar_repetidas:
        conteo = Counter(linea.strip() for linea in lineas if linea.strip())
        repetidas = {linea for linea, veces in conteo.items() if veces >= 3}
        lineas = [linea for linea in lineas if linea.strip() not in repetidas]

    texto = "\n".join(lineas)
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    return texto.strip()


# ---------------------------------------------------------------------------
# LECTORES POR FORMATO (devuelven una lista de (página, texto))
# ---------------------------------------------------------------------------
def _leer_md(archivo: Path) -> list[tuple[int, str]]:
    return [(1, clean_text(archivo.read_text(encoding="utf-8")))]


def _leer_pdf(archivo: Path) -> list[tuple[int, str]]:
    from pypdf import PdfReader  # import local: solo hace falta si hay PDFs

    try:
        lector = PdfReader(archivo)
    except Exception as error:
        raise DatasetError(f"No se pudo abrir el PDF {archivo.name}: {error}") from error
    # Se limpian todas las páginas juntas para detectar encabezados repetidos entre páginas.
    separador = "\n\f\n"
    completo = separador.join(pagina.extract_text() or "" for pagina in lector.pages)
    paginas = clean_text(completo.replace(separador, "\n<<<PAGINA>>>\n"), quitar_repetidas=True)
    return [(n, clean_text(t)) for n, t in enumerate(paginas.split("<<<PAGINA>>>"), start=1)]


def _leer_json(archivo: Path) -> list[tuple[int, str]]:
    try:
        datos = json.loads(archivo.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise DatasetError(f"{archivo.name} no es un JSON válido: {error}") from error

    # Si es un objeto que contiene una lista (por ej. {"items": [...]}), se usa esa lista.
    if isinstance(datos, dict):
        listas = [v for v in datos.values() if isinstance(v, list)]
        datos = listas[0] if listas else [datos]
    if not isinstance(datos, list):
        raise DatasetError(f"{archivo.name}: se esperaba una lista de objetos o un objeto")

    items = []
    for n, item in enumerate(datos, start=1):
        if isinstance(item, str):
            texto = item
        elif isinstance(item, dict):
            cuerpo = next((item[c] for c in CAMPOS_TEXTO_JSON if isinstance(item.get(c), str)), None)
            if cuerpo is None:
                raise DatasetError(
                    f"{archivo.name}, ítem {n}: no tiene ninguno de los campos {CAMPOS_TEXTO_JSON}"
                )
            titulo = item.get("title")
            texto = f"# {titulo}\n\n{cuerpo}" if isinstance(titulo, str) else cuerpo
        else:
            raise DatasetError(f"{archivo.name}, ítem {n}: tipo no soportado ({type(item).__name__})")
        items.append((n, clean_text(texto)))
    return items


LECTORES = {".md": _leer_md, ".pdf": _leer_pdf, ".json": _leer_json}


# ---------------------------------------------------------------------------
# CARGA
# ---------------------------------------------------------------------------
def cargar_documentos(carpeta: Path = DATA_DIR) -> list[Document]:
    """Lee todos los .md/.pdf/.json de la carpeta (y sus subcarpetas)."""
    if not carpeta.is_dir():
        raise DatasetError(f"No existe la carpeta de datos: {carpeta}. Corré: python download_docs.py")

    archivos = sorted(p for p in carpeta.rglob("*") if p.suffix.lower() in EXTENSIONES)
    if not archivos:
        raise DatasetError(f"No hay archivos {EXTENSIONES} en {carpeta}. Corré: python download_docs.py")

    documentos: list[Document] = []
    vistos: dict[str, Path] = {}
    for archivo in archivos:
        doc_id = archivo.stem
        # doc_id tiene que ser único: es la "respuesta correcta" del golden set.
        if doc_id in vistos:
            raise DatasetError(
                f"Dos archivos generan el mismo doc_id '{doc_id}': {vistos[doc_id]} y {archivo}. Renombrá uno."
            )
        vistos[doc_id] = archivo

        relativa = archivo.relative_to(carpeta)
        categoria = relativa.parts[0] if len(relativa.parts) > 1 else "general"
        for pagina, texto in LECTORES[archivo.suffix.lower()](archivo):
            if not texto:
                continue  # página vacía (por ej. una página de PDF que es solo una imagen)
            documentos.append(Document(
                page_content=texto,
                metadata={
                    "doc_id": doc_id,
                    "source": relativa.as_posix(),
                    "page": pagina,
                    "category": categoria,
                },
            ))
    return documentos
