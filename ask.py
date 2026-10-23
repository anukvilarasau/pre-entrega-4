"""
ask.py — (Extra) Preguntas en lenguaje natural: RAGSystem + Claude en una cadena LCEL.

    {"pregunta", "category"}
      | assign(resultados = RAGSystem.retrieve(pregunta))       # top-5 híbrido
      | assign(contexto   = fragmentos numerados [1]..[5])
      | assign(respuesta  = prompt | ChatAnthropic(temperature=0) | StrOutputParser)
      | armar_salida                                            # fuentes desde la METADATA

Claude cita los fragmentos que usa con [n]. Las fuentes NO las escribe el modelo:
el código toma esos números y busca el archivo en la metadata de Pinecone, así no
puede inventar una fuente. Si la respuesta no está en los documentos, Claude
responde exactamente: "No tengo esa información en los documentos".

Uso:
    python ask.py "How do I enable CORS in FastAPI?"
    python ask.py "How do I enable CORS in FastAPI?" --category tutorial
    python ask.py                      (modo interactivo: escribí "salir" para terminar)
"""

import config  # noqa: F401  (primero)

import argparse
import re
import sys

import anthropic
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough

from config import ANTHROPIC_MODEL, MAX_TOKENS, NO_SE, TEMPERATURE, TOP_K, ConfigError, requerir
from pinecone_utils import IndiceError
from rag_system import RAGSystem
from schemas import Resultado

PROMPT_SISTEMA = f"""Sos un asistente técnico que responde preguntas sobre FastAPI usando \
ÚNICAMENTE los fragmentos de documentación del CONTEXTO.

Reglas:
1. Usá solo la información del CONTEXTO. No agregues conocimiento propio.
2. Citá cada afirmación con el número del fragmento entre corchetes, por ejemplo [2].
3. Si el CONTEXTO no alcanza para responder, respondé exactamente y solo: {NO_SE}
4. Respondé en el mismo idioma de la pregunta, de forma breve y clara. Podés incluir \
código si está en el CONTEXTO."""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", PROMPT_SISTEMA),
    ("human", "CONTEXTO:\n{contexto}\n\nPREGUNTA: {pregunta}"),
])


def formatear_contexto(resultados: list[Resultado]) -> str:
    """Numera los fragmentos [1]..[k] para que Claude pueda citarlos."""
    return "\n\n".join(f"[{r.posicion}] (fuente: {r.source})\n{r.text}" for r in resultados)


def armar_salida(datos: dict) -> dict:
    """Agrega las fuentes citadas, tomadas de la metadata (no del texto del modelo)."""
    respuesta = datos["respuesta"].strip()
    if respuesta.startswith(NO_SE):
        return {"respuesta": NO_SE, "fuentes": []}
    citados = {int(n) for n in re.findall(r"\[(\d+)\]", respuesta)}
    fuentes = []
    for r in datos["resultados"]:
        if r.posicion in citados and r.source not in fuentes:
            fuentes.append(r.source)
    return {"respuesta": respuesta, "fuentes": fuentes}


def crear_cadena(rag: RAGSystem) -> Runnable:
    """Arma la cadena LCEL completa (recuperación + generación + fuentes)."""
    parametros = {"model": ANTHROPIC_MODEL, "max_tokens": MAX_TOKENS, "max_retries": 2}
    if TEMPERATURE is not None:
        parametros["temperature"] = TEMPERATURE
    llm = ChatAnthropic(**parametros)

    recuperar = RunnableLambda(lambda x: rag.retrieve(x["pregunta"], k=TOP_K, category=x.get("category")))
    return (
        RunnablePassthrough.assign(resultados=recuperar)
        | RunnablePassthrough.assign(contexto=lambda x: formatear_contexto(x["resultados"]))
        | RunnablePassthrough.assign(respuesta=PROMPT | llm | StrOutputParser())
        | RunnableLambda(armar_salida)
    )


def preguntar(cadena: Runnable, pregunta: str, category: str | None) -> None:
    try:
        salida = cadena.invoke({"pregunta": pregunta, "category": category})
    except anthropic.AuthenticationError:
        raise ConfigError("Anthropic rechazó la clave (401). Revisá ANTHROPIC_API_KEY en el .env.")
    except anthropic.BadRequestError as error:
        if "temperature" in str(error):
            raise ConfigError(
                f"El modelo {ANTHROPIC_MODEL} no acepta 'temperature'. Usá ANTHROPIC_MODEL=claude-sonnet-4-6 "
                f"o dejá TEMPERATURE vacía en el .env."
            )
        raise
    print(f"\nRespuesta:\n{salida['respuesta']}")
    if salida["fuentes"]:
        print("\nFuentes:")
        for fuente in salida["fuentes"]:
            print(f"  - data/{fuente}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preguntas sobre la documentación de FastAPI.")
    parser.add_argument("pregunta", nargs="?", help="si se omite, entra en modo interactivo")
    parser.add_argument("--category", help="filtra por categoría: tutorial o advanced")
    args = parser.parse_args()

    requerir("ANTHROPIC_API_KEY")
    cadena = crear_cadena(RAGSystem())
    print(f"Modelo: {ANTHROPIC_MODEL} | temperatura: {TEMPERATURE}")

    if args.pregunta:
        preguntar(cadena, args.pregunta, args.category)
        return
    while True:
        pregunta = input("\nPregunta (o 'salir'): ").strip()
        if pregunta.lower() in {"salir", "exit", ""}:
            break
        preguntar(cadena, pregunta, args.category)


if __name__ == "__main__":
    try:
        main()
    except (ConfigError, IndiceError) as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
