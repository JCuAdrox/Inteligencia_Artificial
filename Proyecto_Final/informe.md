# Informe - Batería en la bolsa de energía de XM: MDP con iteración de valor y agente RAG

SI3003 Inteligencia Artificial - EAFIT 2026-2
Jerónimo Castaño Herrera, Emanuel Gonzalez Florez, Jerónimo Zapata Cuadros

## 1. Problema

En Colombia el precio de bolsa de la energía cambia mucho durante el día: en 2026 el promedio a las 9 a.m. fue de unos 400 COP/kWh y a las 7 p.m. de más de 720 COP/kWh. Una batería podría aprovechar esa diferencia cargando en las horas baratas y descargando en las caras (arbitraje), y además ayuda al sistema en el pico de la noche. La pregunta es **cuándo cargar y cuándo descargar** sin saber los precios futuros, y tener en cuenta que en cada ciclo se pierde energía (eficiencia menor al 100 %). Además, quien quiera entender este tema necesita conocer la normativa del sector (leyes, papel de la CREG y la UPME, reglas para baterías), que está dispersa en documentos largos.

Nuestra solución: un MDP para decidir la operación de la batería, un agente LLM que puede consultar datos, simular políticas y buscar en la normativa, y un dashboard para usar todo eso.

## 2. Datos

- **Precio de Bolsa Nacional horario** (métrica `PrecBolsNaci`) de la API pública de XM, del 1 de enero de 2023 al 31 de octubre de 2026: 1.400 días, 33.600 horas, sin faltantes. Se descargó mes a mes porque la API limita el rango por consulta (`src/descargar_datos.py`).
- Lo que encontramos en el EDA: el nivel del precio cambia muchísimo entre años (2024 fue muy caro, con un trimestre por encima de 1.100 COP/kWh, y 2025 muy barato, alrededor de 200), pero el patrón dentro del día se repite: el pico siempre es entre 6 y 9 p.m. Lo que sí cambió fue la hora más barata: antes era la madrugada y en 2026 es la media mañana, seguramente por la entrada de la generación solar.
- **Corpus del RAG:** Ley 143 de 1994, Ley 1715 de 2014 y Ley 2099 de 2021 (Gestor Normativo de Función Pública), las Resoluciones CREG 098 de 2019 (baterías, derogada), 101 113 de 2026 (baterías, vigente) y 174 de 2021 (autogeneración), y tres documentos de la UPME sobre sistemas de almacenamiento con baterías (Circular 030 de 2025, su análisis de impacto normativo y las respuestas de la convocatoria STR 01-2021). En total 953 fragmentos. Al principio el sitio de la CREG no respondía; descubrimos que el servidor no contesta por IPv6, y forzando IPv4 se pudo descargar.
- Partición temporal: entrenamiento 2023-01 a 2025-06, validación 2025-07 a 2025-12 (para escoger hiperparámetros), prueba 2026-01 a 2026-10. El modelo final se entrena con todo 2023-2025.

## 3. Método de la Parte 1: MDP con iteración de valor

- **Estado** s = (hora 0-23, nivel de carga 0..n, rango k del precio relativo de la hora anterior). El precio relativo es el precio dividido por el promedio de las 24 horas anteriores; lo usamos porque el nivel en pesos cambia demasiado entre años. Se divide en K rangos por cuantiles.
- **Acciones**: cargar, esperar, descargar (un paso de `potencia` MWh). No se puede cargar llena ni descargar vacía.
- **Transiciones**: la hora avanza de forma determinística, el nivel según la acción, y el rango de precio según una matriz T[h, k, k'] estimada contando transiciones en los datos de entrenamiento (con suavizado de Laplace).
- **Recompensa**: cargar = -E[p_h | k] · E, descargar = +E[p_h | k] · E · eficiencia, donde E[p_h | k] es el precio relativo esperado en la hora h dado el rango de la hora anterior (estimado con los datos). Solo usamos el precio de la hora anterior porque el precio de bolsa real se conoce después.
- **Solución**: iteración de valor vectorizada con γ = 0.999. Con γ = 0.95 el modelo casi no hacía arbitraje porque entre cargar a las 3 a.m. y vender a las 7 p.m. pasan 16 horas y 0.95^16 = 0.44. Con γ tan cercano a 1 usamos como criterio de parada que el cambio del valor sea casi igual en todos los estados (span < 0.01), que es cuando la política deja de cambiar. Converge en ~9.000 iteraciones y 2.5 segundos.
- **K** se escogió por validación: K = 3 dio la mayor ganancia (28.48 M vs 28.39 M con K = 1 y 26.6 M con K = 8).
- **Líneas base**: *ingenua* (carga de 1 a 4 a.m. y descarga desde las 6 p.m.), *perfil* (cada día carga en las 4 horas más baratas y descarga en las 4 más caras según el promedio de los últimos 30 días) y *oráculo* (programación dinámica hacia atrás conociendo todos los precios, cota superior).

## 4. Diseño del agente y del RAG

- **Loop agéntico manual** (`src/agente.py`): se le manda al LLM el historial y la lista de herramientas; si pide herramientas se ejecutan y el resultado se agrega como mensaje `tool`; se repite hasta que responda en texto o hasta **T_max = 6** pasos. Si una herramienta falla (argumentos raros, fechas sin datos, parámetros imposibles) el error se devuelve como observación en vez de romper el programa, para que el modelo lo explique o corrija.
- **LLM**: `openai/gpt-oss-20b` en Groq (capa gratis, API compatible con OpenAI), temperatura 0. Se puede cambiar de proveedor con variables de entorno.
- **Herramientas**: `consultar_precios(fecha_inicio, fecha_fin)` (estadísticas y horas más baratas/caras), `simular_bateria(fecha_inicio, fecha_fin, politica, capacidad, potencia, eficiencia)` (entrena/usa el MDP y compara con las líneas base) y `buscar_normativa(pregunta)` (RAG).
- **RAG**: las leyes y resoluciones se parten por artículo y los PDF de la UPME en fragmentos de ~1.500 caracteres. Embeddings con `intfloat/multilingual-e5-small` (local, sin costo) e índice FAISS de producto interno. Se devuelven los 4 fragmentos más parecidos con una etiqueta de fuente (`ley_143_1994, Artículo 23`) y el prompt de sistema le pide al modelo citar con ese formato. Primero probamos `paraphrase-multilingual-MiniLM-L12-v2` pero traía fragmentos menos relevantes (por ejemplo, para "¿qué es la bolsa de energía?" traía un artículo sobre energía de residuos).
- **Dashboard** (`app.py`, Streamlit): el usuario escoge fechas, capacidad, horas de llenado, eficiencia y política. Muestra la ganancia de las 4 políticas, el precio con las decisiones de carga/descarga, la ganancia acumulada, el nivel de la batería, la política aprendida (mapa de calor hora x nivel), el histórico diario y un mapa de calor de precio por hora y mes, y una pestaña de chat con el agente donde se ven las herramientas usadas.

## 5. Resultados

**MDP en prueba (ene-oct 2026, batería de 1 MWh, 4 h, eficiencia 85 %):**

| política | ganancia (M COP) | % oráculo | vs ingenua |
|---|---|---|---|
| MDP | 41.6 | 74 % | +12.5 % |
| ingenua | 37.0 | 65 % | - |
| perfil 30 días | 47.7 | 84 % | +28.9 % |
| oráculo | 56.5 | 100 % | +52.7 % |

- El MDP le gana a la regla ingenua en 9 de 10 meses; en agosto la ingenua perdió plata (-1.9 M) y el MDP ganó 0.25 M.
- Con eficiencia de 70 % el MDP gana 16.6 M y la ingenua solo 6.7 M: el MDP tiene en cuenta las pérdidas y no hace ciclos que no valen la pena.
- La regla adaptativa (perfil de 30 días) le gana al MDP en 8 de 10 meses. Re-entrenar el MDP cada mes con los últimos 180 días sube su ganancia a 43.4 M, pero no alcanza. Lo interpretamos como que los datos no son estacionarios (cambio de horas baratas por la solar) y que nuestro estado tiene poca memoria.
- La política aprendida es razonable: con precio normal carga de día (6 a.m. a 6 p.m.) y descarga en el pico de la noche; si la hora anterior fue barata carga casi a cualquier hora, y si fue cara solo carga en unas pocas horas de la tarde (1 a 5 p.m.).

**Agente (15 preguntas fijas, notebook 02):** herramienta correcta en 14/15 (93 %) y cita de la fuente en 6/6 preguntas de normativa. El único "fallo" fue pedir una simulación para 2030: no llamó la herramienta pero respondió correctamente que no hay datos. Los errores inyectados (eficiencia de 150 %) se devolvieron como observación y el agente los explicó. A la pregunta de quién programa la carga y descarga de un SAEB respondió bien (el CND, según la CREG 101 113 de 2026, que derogó la 098 de 2019). Revisando a mano, 12 respuestas están completamente bien y 2 parcialmente: en una agregó un "incentivo" que no es tributario y en otra la cita textual mezcla dos artículos de la misma resolución. También vimos que la respuesta sobre licencia ambiental de los SAEB cambió entre dos corridas, aunque usamos temperatura 0. Agregar las resoluciones de la CREG mejoró la pregunta sobre las funciones de la CREG, que antes traía un artículo que no era. Las preguntas de normativa tardan entre 18 y 57 s por los límites de la capa gratis.

## 6. Limitaciones

- Modelo de batería simplificado: tomadora de precios, sin degradación, sin cargos de red, impuestos ni restricciones del operador; en Colombia los SAEB regulados los despacha el CND y se remuneran de otra forma, así que las ganancias no son reales, solo sirven para comparar políticas.
- El estado del MDP es pequeño (3 rangos de precio) y la matriz de transición es un promedio de 3 años distintos entre sí; una regla simple adaptativa funcionó mejor. Mejoras posibles: meter en el estado información del perfil reciente, re-entrenar con ventanas móviles o usar Q-learning con más variables.
- El corpus RAG es pequeño (3 leyes, 3 resoluciones, 3 documentos UPME) y el modelo de embeddings también, por eso a veces mezcla fragmentos parecidos. La métrica de "citó la fuente" solo revisa el formato, no que la cita sea la correcta (eso se revisó a mano).
- La evaluación del agente son solo 15 preguntas y el LLM puede variar entre corridas.
- **Nada de esto es asesoría financiera, legal ni de inversión.**
