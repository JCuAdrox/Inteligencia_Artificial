# Agente LLM con loop manual de tool calling (API compatible con OpenAI, por defecto Groq).
# Herramientas: consultar_precios, simular_bateria, buscar_normativa
import os
import json
from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd

from src import mdp, rag

load_dotenv()
cliente = OpenAI(base_url=os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
                 api_key=os.getenv("LLM_API_KEY") or os.getenv("GROQ_API_KEY"))
MODELO = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
T_MAX = 6  # maximo de vueltas del loop

precios = mdp.cargar_precios()
train = precios[precios["fecha"] < "2026-01-01"]
modelos = {}  # cache de politicas entrenadas por parametros

SISTEMA = """Eres un asistente que ayuda a entender la operacion de una bateria que compra y vende
energia en la bolsa de XM (Colombia). Tienes herramientas para consultar precios reales,
simular politicas de la bateria y buscar en la normativa (Ley 143 de 1994, Ley 1715 de 2014,
Ley 2099 de 2021, Resoluciones CREG 098 de 2019, 101 113 de 2026 y 174 de 2021, y documentos de la
UPME sobre baterias). Ojo: la CREG 098 de 2019 fue derogada por la CREG 101 113 de 2026. Reglas:
- Para preguntas sobre precios o fechas usa consultar_precios. Para ganancias o decisiones de la bateria usa simular_bateria.
- Para preguntas de normativa usa buscar_normativa y cita la fuente asi: [ley_143_1994, Artículo 23] o [creg_101_113_2026, Artículo 5].
- No inventes datos ni articulos. Si una herramienta devuelve error, explicalo y corrige si puedes.
- Los datos disponibles van de 2023-01-01 a 2026-10-31. El modelo se entreno con 2023-2025.
- Esto es un ejercicio academico, no es asesoria legal ni financiera; dilo si te piden recomendaciones de inversion.
Responde en español, corto y claro."""

herramientas = [
    {"type": "function", "function": {
        "name": "consultar_precios",
        "description": "Estadisticas del precio de bolsa nacional (COP/kWh) entre dos fechas: promedio, min, max, horas mas baratas y mas caras.",
        "parameters": {"type": "object", "properties": {
            "fecha_inicio": {"type": "string", "description": "YYYY-MM-DD"},
            "fecha_fin": {"type": "string", "description": "YYYY-MM-DD"}},
            "required": ["fecha_inicio", "fecha_fin"]}}},
    {"type": "function", "function": {
        "name": "simular_bateria",
        "description": "Simula una bateria comprando/vendiendo en bolsa entre dos fechas y devuelve la ganancia en COP de la politica pedida y de las demas para comparar. Politicas: mdp (iteracion de valor), ingenua (carga 1-4am, descarga 6-9pm), perfil (horas baratas/caras promedio), oraculo (conoce el futuro, cota superior).",
        "parameters": {"type": "object", "properties": {
            "fecha_inicio": {"type": "string", "description": "YYYY-MM-DD"},
            "fecha_fin": {"type": "string", "description": "YYYY-MM-DD"},
            "politica": {"type": "string", "enum": ["mdp", "ingenua", "perfil", "oraculo"]},
            "capacidad": {"type": "number", "description": "MWh, por defecto 1"},
            "potencia": {"type": "number", "description": "MWh por hora, por defecto 0.25"},
            "eficiencia": {"type": "number", "description": "eficiencia ida y vuelta entre 0 y 1, por defecto 0.85"}},
            "required": ["fecha_inicio", "fecha_fin"]}}},
    {"type": "function", "function": {
        "name": "buscar_normativa",
        "description": "Busca fragmentos relevantes en la normativa del sector electrico colombiano: Ley 143/1994, Ley 1715/2014, Ley 2099/2021, Resoluciones CREG 098/2019 (derogada), 101 113/2026 (baterias SAEB) y 174/2021 (autogeneracion), y documentos de la UPME sobre baterias.",
        "parameters": {"type": "object", "properties": {
            "pregunta": {"type": "string"}}, "required": ["pregunta"]}}},
]


def consultar_precios(fecha_inicio, fecha_fin):
    d = precios[(precios["fecha"] >= fecha_inicio) & (precios["fecha"] <= fecha_fin)]
    if len(d) == 0:
        return {"error": "no hay datos en ese rango, los datos van de 2023-01-01 a 2026-10-31"}
    por_hora = d.groupby("hora")["precio"].mean().sort_values()
    return {"fuente": "XM, PrecBolsNaci (API servapibi.xm.com.co)", "dias": int(len(d) / 24),
            "promedio": round(d["precio"].mean(), 1), "minimo": round(d["precio"].min(), 1),
            "maximo": round(d["precio"].max(), 1),
            "horas_mas_baratas": [int(h) for h in por_hora.index[:4]],
            "horas_mas_caras": [int(h) for h in por_hora.index[-4:]]}


def simular_bateria(fecha_inicio, fecha_fin, politica="mdp", capacidad=1.0, potencia=0.25, eficiencia=0.85):
    if potencia <= 0 or capacidad < potencia or not (0 < eficiencia <= 1):
        return {"error": "parametros invalidos: se necesita 0 < potencia <= capacidad y 0 < eficiencia <= 1"}
    d = precios[(precios["fecha"] >= fecha_inicio) & (precios["fecha"] <= fecha_fin)]
    if len(d) == 0:
        return {"error": "no hay datos en ese rango, los datos van de 2023-01-01 a 2026-10-31"}
    clave = (capacidad, potencia, eficiencia)
    if clave not in modelos:
        modelos[clave] = mdp.entrenar(train, capacidad, potencia, eficiencia)
    m = modelos[clave]
    ganancias = {}
    for p in ["mdp", "ingenua", "perfil", "oraculo"]:
        ganancias[p] = round(float(mdp.simular(d, m, p)["ganancia"].sum()))
    r = mdp.simular(d, m, politica)
    aviso = ""
    if fecha_inicio < "2026-01-01":
        aviso = "ojo: el rango incluye datos de entrenamiento (2023-2025), el resultado es optimista"
    return {"politica": politica, "ganancia_cop": ganancias[politica], "comparacion_cop": ganancias,
            "horas_cargando": int((r["accion"] == "cargar").sum()),
            "horas_descargando": int((r["accion"] == "descargar").sum()), "aviso": aviso}


def buscar_normativa(pregunta):
    return rag.buscar(pregunta)


funciones = {"consultar_precios": consultar_precios, "simular_bateria": simular_bateria,
             "buscar_normativa": buscar_normativa}


def preguntar(pregunta, historial=None):
    """Corre el loop agentico. Devuelve la respuesta y la lista de herramientas llamadas."""
    mensajes = [{"role": "system", "content": SISTEMA}] + (historial or [])
    mensajes.append({"role": "user", "content": pregunta})
    llamadas = []
    for paso in range(T_MAX):
        r = cliente.chat.completions.create(model=MODELO, messages=mensajes,
                                            tools=herramientas, temperature=0)
        msg = r.choices[0].message
        if not msg.tool_calls:
            return msg.content, llamadas
        mensajes.append({"role": "assistant", "content": msg.content or "",
                         "tool_calls": [tc.model_dump() for tc in msg.tool_calls]})
        for tc in msg.tool_calls:
            nombre = tc.function.name
            # los errores de las herramientas se devuelven al modelo como observacion
            try:
                args = json.loads(tc.function.arguments or "{}")
                obs = funciones[nombre](**args)
            except Exception as e:
                args = tc.function.arguments
                obs = {"error": str(e)}
            llamadas.append({"herramienta": nombre, "args": args})
            mensajes.append({"role": "tool", "tool_call_id": tc.id,
                             "content": json.dumps(obs, ensure_ascii=False, default=str)})
    return "No alcance a responder en %d pasos, intenta con una pregunta mas concreta." % T_MAX, llamadas
