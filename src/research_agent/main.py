import urllib.error
import urllib.request

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()


@tool
def fetch_text_from_url(url: str) -> str:
    """Fetch the document from a URL.
    """
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; quickstart-research/1.0)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read()
    except urllib.error.URLError as e:
        return f"Fetch failed: {e}"
    text = raw.decode("utf-8", errors="replace")
    return text

SYSTEM_PROMPT = """Sos un agente de research. Tu trabajo es investigar temas a fondo usando las herramientas de búsqueda disponibles y dar respuestas precisas, verificadas y bien fundamentadas.

Reglas:
1. Nunca respondas de memoria si el tema puede haber cambiado recientemente (noticias, versiones de software, datos, personas en cargos). Buscá primero.
2. Si la pregunta tiene varias partes, buscá cada una por separado en vez de una sola query genérica.
3. Usá múltiples fuentes cuando el tema sea controversial o poco claro; no te quedes con el primer resultado.
4. Si las fuentes se contradicen, decilo explícitamente en la respuesta en vez de elegir una al azar.
5. Citá de dónde sacaste cada dato relevante (nombre de la fuente, no solo la URL).
6. Si después de buscar no encontrás información confiable, decilo — no inventes.
7. Sé conciso en la respuesta final: sintetizá, no pegues resultados crudos de búsqueda.

Formato de respuesta final: primero la respuesta directa, después un breve listado de fuentes usadas."""


model = init_chat_model(
    model="ollama:qwen3:8b",
    temperature=0.2,
    timeout=300,
    max_output_tokens=1024,
)


agent = create_agent(
    tools=[fetch_text_from_url],
    model=model,
    system_prompt=SYSTEM_PROMPT,
    checkpointer=checkpointer
)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user", 
                "content": "Hola, necesito que me ayudes a investigar sobre Langgraph."
            }
        ]
    },
    config={
        "configurable": {
            "thread_id": "research_thread_001"
        }
    }
)

print(result["messages"][-1].content_blocks)
