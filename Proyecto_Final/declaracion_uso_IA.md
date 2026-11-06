# Declaración de uso de IA generativa

Proyecto final SI3003 - Inteligencia Artificial - EAFIT 2026-2
Equipo: Jerónimo Castaño Herrera, Emanuel Gonzalez Florez, Jerónimo Zapata Cuadros

Conforme al pacto pedagógico del curso, declaramos el uso de IA generativa en este proyecto.

## Herramientas usadas y para qué

| Herramienta | Para qué la usamos |
|---|---|
| Claude (Claude Code, de Anthropic) | Asistente de programación: ayudó a escribir y depurar el código (descarga de datos de XM y de los documentos, MDP con iteración de valor, RAG, loop del agente, dashboard en Streamlit), a generar y ejecutar los notebooks, y a redactar borradores del README, del informe y de los análisis. También se usó para buscar qué documentos normativos estaban accesibles. |
| `openai/gpt-oss-20b` vía Groq | Es el LLM que usa el agente dentro de la aplicación (parte del producto, no de la elaboración del trabajo). |
| `intfloat/multilingual-e5-small` | Modelo de embeddings del RAG (parte del producto). |

## Cómo verificamos lo generado

- Todos los números del informe, el README y los notebooks salen de ejecutar el código sobre los datos reales; los notebooks se entregan ejecutados.
- Varias decisiones salieron de revisar resultados que no funcionaban: por ejemplo el MDP con rangos de precio absolutos y con γ = 0.95 quedaba por debajo de la regla ingenua, y se cambió a precio relativo y γ = 0.999 después de analizar por qué.
- Las respuestas del agente se revisaron una por una (notebook 02) y se reportan también los errores.
- Los textos legales del corpus son descargados directamente de las fuentes oficiales, no generados por IA.

> **Pendiente del equipo antes de entregar:** cada integrante debe revisar el código y los análisis, verificar que entiende cómo funciona cada parte, y completar aquí qué hizo cada uno (aporte individual).

## Aporte individual

- Jerónimo Castaño Herrera: _(completar)_
- Emanuel Gonzalez Florez: _(completar)_
- Jerónimo Zapata Cuadros: _(completar)_
