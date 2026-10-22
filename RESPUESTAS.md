# Respuestas del workshop RAG

> Todas las tablas salen de ejecuciones reales de los notebooks de `notebooks/` (con sus salidas guardadas) y las capturas están en `evidencias/`. Modelos: generación `nvidia/nemotron-3.5-lightning-30b-a3b`, embeddings `nvidia/llama-nemotron-embed-vl-1b-v2`.

---

# Bloque 0 · Teoría

## T1 · Cuántos nodos viven en cada capa

_Evidencia:_ cálculo a mano + simulación (código abajo).

**Cálculo a mano.** Con `mL = 1/ln(M)`, P(nivel ≥ k) = M⁻ᵏ. Con M = 16 y n = 1 000 000:

| Capa | Probabilidad | Nodos esperados |
|---|---|---:|
| ≥ 1 | 16⁻¹ = 1/16 = 0.0625 | 1 000 000 / 16 = **62 500** |
| ≥ 2 | 16⁻² = 1/256 ≈ 0.0039 | 1 000 000 / 256 ≈ **3 906** |
| ≥ 3 | 16⁻³ = 1/4096 ≈ 0.00024 | 1 000 000 / 4096 ≈ **244** |

**Simulación** (semilla 0):

```python
import numpy as np
M, n = 16, 1_000_000
levels = np.floor(-np.log(np.random.default_rng(0).random(n)) / np.log(M)).astype(int)
for k in (1, 2, 3):
    print(f"capa >= {k}: teórico {n*M**-k:,.1f}  simulado {(levels>=k).sum():,}")
```

```
capa >= 1: teórico 62,500.0  simulado 62,397
capa >= 2: teórico 3,906.2  simulado 3,937
capa >= 3: teórico 244.1  simulado 260
capa máxima simulada: 5
```

La simulación coincide con la teoría con errores de 0.2 %, 0.8 % y 6.5 %. El error crece en las capas altas porque son pocos nodos y el ruido relativo de una binomial es ~1/√(n·p) (para 244 nodos, ±6 %). La capa máxima es 5, cerca de log₁₆(10⁶) ≈ 5.

**Por qué conviene que haya pocos hubs arriba.** Cada capa tiene ~16 veces menos nodos que la de abajo, así que en las capas altas cada salto greedy cruza una fracción enorme del espacio: con unos pocos saltos por capa llegas a la zona correcta en O(log n) pasos. Si las capas altas tuvieran muchos nodos, el descenso tendría que recorrer casi tantos vecinos como la capa 0 y se perdería la ventaja de la jerarquía, además de gastar memoria en aristas redundantes.

## T2 · ef, k y el tamaño de la cola

_Evidencia:_ presentación «Búsqueda en HNSW» y reto C1.

1. **k = 10 con ef = 5.** La cola de candidatos de la capa 0 solo guarda los 5 mejores vistos, así que es imposible devolver 10: el algoritmo descarta candidatos que deberían estar en el top-10. Las librerías lo «arreglan» de forma distinta (hnswlib lanza error o usa `max(ef, k)`), pero conceptualmente la regla es **ef ≥ k**.
2. **Valor inicial.** Empezar con **ef = 2·k** y subirlo si el recall medido es insuficiente (la presentación da ef = 50–100 con k = 5–10 como rango típico en RAG). En C1 lo vi: con M=16 y datos aleatorios ef=64 da recall 0.77 y hace falta ef=256 para pasar de 0.95; con datos agrupados ef=10 ya da 0.97.
3. **ef se puede cambiar en producción sin reconstruir; ef_construction no.** `ef_construction` solo se usa al insertar para decidir con qué vecinos conectar cada nodo: queda «horneado» en las aristas del grafo y cambiarlo exige reindexar. `ef` es un parámetro de la consulta, no modifica el grafo, así que puedes ajustarlo en cada búsqueda (en C1 cambio `index.hnsw.efSearch` sobre el mismo índice ya construido).

## T3 · Inserción frente a búsqueda

_Evidencia:_ presentaciones «Inserción» y «Búsqueda».

| | Inserción | Búsqueda | Por qué |
|---|---|---|---|
| ¿Se sortea la capa? | **Sí**: l = ⌊−ln(U)·mL⌋ | **No**: siempre se entra por la capa más alta | El sorteo decide hasta dónde «vive» el nodo nuevo; una consulta no se guarda en el grafo, así que no necesita capa propia. |
| Valor de ef en las capas altas | **ef = 1** en las capas por encima de l (solo descender) | **ef = 1** en las capas L_max…1 | En ambos casos las capas altas solo sirven para acercarse rápido a la región correcta; basta con el mejor candidato. |
| Parámetro de la fase principal | **ef_construction** (de la capa l a la 0) | **ef** (solo en la capa 0) | En la inserción importa encontrar buenos vecinos para crear aristas de calidad en cada capa donde vive el nodo; en la búsqueda solo importa el resultado final en la capa 0, que tiene todos los nodos. |
| ¿Modifica el índice? | **Sí**: crea aristas (hasta M por capa) y poda conexiones de los vecinos | **No**: solo lee | La inserción construye el grafo; la búsqueda lo recorre. Por eso ef se puede cambiar en caliente y ef_construction no. |

En el notebook 03, `vectorstore.add_documents(all_docs)` dispara la **inserción** (cada chunk se convierte en vector y se hace *upsert*), y `retriever.invoke(pregunta)` dispara la **búsqueda** (embebe la pregunta y pide los k vecinos).

> Matiz que encontré: `QdrantClient(":memory:")` es el modo local del cliente de Python, que según su implementación hace búsqueda exhaustiva con NumPy en vez de construir un grafo HNSW. El servidor Qdrant sí construye HNSW. Con 150 chunks la diferencia de velocidad es irrelevante (C1 muestra que el exacto gana en datasets pequeños).

## T4 · El invariante de FAISSVectorStore

_Evidencia:_ presentación «FAISS explicado».

| Posición FAISS | Antes: `self.documents[i]` | Después de borrar B solo de la lista |
|---:|---|---|
| 0 | A | A |
| 1 | B | **C** |
| 2 | C | **D** |
| 3 | D | **— (no existe → IndexError)** |

El índice FAISS sigue teniendo 4 vectores y no sabe que B se borró. Una búsqueda que antes devolvía `[1, 2, 3]` (B, C, D) ahora se traduce como:

- `documents[1]` → **C** en lugar de B: **error silencioso** (devuelve un texto equivocado con apariencia válida).
- `documents[2]` → **D** en lugar de C: **error silencioso**.
- `documents[3]` → **IndexError**: la lista solo tiene 3 elementos, es el único que **lanza excepción**.

Lo peligroso son los dos errores silenciosos: el RAG citaría un chunk que no corresponde al vector que coincidió, y nadie se entera. Por eso existen bases vectoriales como **Qdrant**: guardan el *payload* (texto y metadata) **junto al vector**, bajo el mismo id, y gestionan borrados y actualizaciones de forma nativa, así que no hay dos estructuras paralelas que se puedan desincronizar.

## T5 · El costo de la fuerza bruta

_Evidencia:_ cuenta abajo y C1 (experimento 2).

- 1 000 000 vectores × 768 dimensiones = **7.68 × 10⁸** multiplicaciones por consulta.
- × 100 consultas/s = **7.68 × 10¹⁰ multiplicaciones por segundo** (77 mil millones).

Además del cómputo está la memoria: cada consulta recorre 10⁶ × 768 × 4 B ≈ 3.1 GB de vectores float32; a 100 consultas/s son **~307 GB/s** de ancho de banda, más de lo que da la RAM de un servidor típico. Y el costo crece linealmente con n: con 10 millones de documentos se multiplica por 10. En C1 lo medí: el exacto pasa de 0.2 ms (n=20 000) a 14.6 ms por consulta (n=500 000), 73 veces más lento para 25 veces más datos (el salto extra viene de que los vectores dejan de caber en caché).

**ANN:** se sacrifica la garantía de encontrar exactamente los k más cercanos (recall < 1) a cambio de búsquedas sublineales (≈ log n) y latencias de milisegundos que no explotan con el tamaño del corpus.

---

# Bloque A · App RAG empresarial

> Modelos usados en toda la entrega: generación `nvidia/nemotron-3.5-lightning-30b-a3b` (con el razonamiento desactivado: `enable_thinking=False`, porque si no gasta todos los tokens pensando y devuelve respuestas vacías) y embeddings `nvidia/llama-nemotron-embed-vl-1b-v2` (dimensión 2048). El endpoint de embeddings respondía 502 en ~40 % de las llamadas, así que los notebooks envuelven los embeddings en `RetryEmbeddings` (timeout corto, reintentos y caché de consultas).

## A1 · Ejecuta la app RAG de punta a punta

_Evidencia:_ [`evidencias/A1_gradio.png`](evidencias/A1_gradio.png) · [`notebooks/02b_rag_app_original_A1.ipynb`](notebooks/02b_rag_app_original_A1.ipynb) (copia del original con los documentos de demostración, ejecutada)

Índice: 4 documentos → **4 chunks** (cada documento de demostración mide menos de 400 caracteres, así que no se parte).

| # | Pregunta | Respuesta del RAG | Fuentes recuperadas | ¿Correcta? |
|---:|---|---|---|:---:|
| 1 | ¿Cuál es el precio del plan Enterprise de CloudSync Pro? | $49/mes por usuario con compromiso anual. | CloudSync Pro — Precios, Política de reembolsos, Política de trabajo remoto | Sí |
| 2 | ¿Cuántos días pueden trabajar remotamente los empleados? | Hasta 3 días por semana. | Política de trabajo remoto, Onboarding, Política de reembolsos | Parcial (omite «con aprobación del jefe directo») |
| 3 | ¿Qué pasa con los reembolsos mayores a $100? | Requieren aprobación del supervisor y se procesan en 3 a 5 días hábiles al método de pago original. | Política de reembolsos, CloudSync Pro — Precios, Política de trabajo remoto | Sí |
| 4 | ¿Qué ocurre durante la primera semana de onboarding? | Se completan módulos obligatorios de seguridad, cumplimiento y cultura de la empresa. | Onboarding, Política de trabajo remoto, Política de reembolsos | Sí |
| 5 | ¿Cuántos días de vacaciones tiene un empleado al año? *(fuera de contexto)* | No tengo esa información en el contexto proporcionado. | Política de trabajo remoto, Onboarding, CloudSync Pro — Precios | Sí (no inventó) |

**Qué observé.** Las 4 preguntas de ejemplo salen bien y la fuente correcta siempre es la primera. Pero **siempre se recuperan 3 fuentes**, incluso en la pregunta de vacaciones: el umbral 0,3 no filtró nada (en B2 descubrí por qué: el retriever compara (1 + coseno) / 2 contra el umbral, así que 0,3 equivale a coseno ≥ −0,4); lo que evitó la alucinación en la pregunta 5 fue el prompt («si el contexto no contiene la respuesta, indica que no tienes esa información»). La respuesta 2 es correcta pero incompleta: el modelo resume demasiado y pierde la condición de aprobación.

## A2 · Tu propio corpus de documentos

_Evidencia:_ [`data/company_documents.json`](data/company_documents.json) · [`evidencias/A2_corpus.png`](evidencias/A2_corpus.png) · celda «4b» de [`notebooks/02b_rag_app_nvidia_nim_colab.ipynb`](notebooks/02b_rag_app_nvidia_nim_colab.ipynb)

**Corpus:** *Tostadora Altamira*, una tostadora de café de especialidad (inventada) en Envigado. 10 documentos con metadata `department` (ventas, logistica, servicio, produccion, recursos_humanos, finanzas) y `type` (pricing, policy, procedure). Algunos se parecen a propósito (dos hablan de aprobaciones por monto, dos de reembolsos, dos de tiempos de entrega) para que la recuperación no sea trivial.

**Chunking.** Los documentos miden entre 434 y 586 caracteres (promedio 514) y cada uno trata un solo tema en dos párrafos.

| chunk_size / overlap | chunks | Efecto |
|---|---:|---|
| 400 / 40 (original) | 21 | Casi todos se parten en dos; el segundo párrafo queda sin el tema del primero. |
| **600 / 60 (elegido)** | **10** | Un chunk por documento, coherente; con k=3 el contexto sigue siendo ≈1 500 caracteres. |

Elegí 600 porque con documentos cortos y de un solo tema, partirlos solo diluye el embedding: un fragmento como «Se puede pausar hasta dos meses por año sin costo» pierde la palabra *suscripción*. Si el corpus tuviera manuales largos volvería a un tamaño menor.

Resultados (prompt A3 activo, k=3, umbral 0,3; scores sin umbral):

| # | Pregunta | Respuesta del RAG | Fuentes (score) | Esperada | ¿Correcta? |
|---:|---|---|---|---|:---:|
| 1 | ¿Cuánto cuesta la libra de café Reserva? | $52.000 la libra [Catálogo y precios de café 2026]. | Catálogo y precios (0.401), Suscripción (0.356), Tueste y calidad (0.259) | $52.000 la libra | Sí |
| 2 | ¿Hasta qué día puedo cancelar la suscripción sin que me cobren el mes siguiente? | No encontraré esa información en los documentos de Altamira. | Suscripción Altamira Club (0.405), Mayoristas (0.225), Devoluciones (0.197) | Antes del día 25 | **No** (falso «no sé») |
| 3 | ¿Qué descuento reciben los clientes mayoristas y cuándo pueden tener crédito? | 18 % de descuento sobre el precio de lista y crédito a 30 días después de seis meses de compras puntuales. | Mayoristas (0.543), Café verde (0.150), Suscripción (0.140) | 18 %; crédito a 30 días tras 6 meses | Sí (sin cita) |
| 4 | ¿Qué pasa con un lote que obtiene menos de 82 puntos en la catación? | No se vende como Reserva y pasa a la línea Clásica. [Proceso de tueste y control de calidad] | Tueste y calidad (0.304), Café verde (0.228), Mayoristas (0.163) | Pasa a la línea Clásica | Sí |
| 5 | ¿Quién debe aprobar un gasto de viaje de $450.000? | La gerencia financiera, ya que supera el límite de $300.000 para el líder del área. [Reembolso de gastos y viáticos] | Viáticos (0.336), Suscripción (0.099), Envíos (0.098) | Gerencia financiera | Sí |
| 6 | ¿Venden máquinas de espresso para el hogar? | No encuentro esa información en los documentos de Altamira. | Catálogo (0.246), Mayoristas (0.205), Horarios (0.197) | No está | Sí |
| 7 | ¿Cuál es el salario de un operario de tostión? | No encuentro esa información en los documentos de Altamira. | Horarios (0.318), Tueste y calidad (0.245), Viáticos (0.223) | No está | Sí |

**Resultado: 6 de 7 correctas.** La captura `A2_corpus.png` muestra justo el caso fallido, con el documento correcto recuperado (0.405 ✅). El fallo es interesante: el documento correcto se recuperó primero (0.405) y le llegó al LLM (junto con los otros dos, porque el umbral no filtra; ver B2), y aun así el modelo dijo que no tenía la información. El texto dice «Para cancelar hay que avisar antes del día 25», y la pregunta habla de «hasta qué día… sin que me cobren»: el modelo necesitaba una pequeña inferencia y el prompt estricto lo empujó a rendirse. Las preguntas 6 y 7 se rechazaron bien. En la 7 el umbral no habría bastado: «Horarios, turnos y trabajo remoto» pasó con 0.318.

## A3 · Ingeniería del prompt (antes y después)

_Evidencia:_ celda «A3 · Prompt antes y después» del notebook 02b (mismo índice, solo cambia el prompt)

**Diff del prompt:**

```diff
- Eres un asistente empresarial. Responde únicamente con información del contexto
- proporcionado. Si el contexto no contiene la respuesta, indica que no tienes esa
- información. Responde en español y de forma concisa.
+ Eres el asistente interno de Tostadora Altamira. Responde únicamente con información del CONTEXTO.
+ Reglas obligatorias:
+ 1. Cita el título exacto del documento que usaste entre corchetes, por ejemplo
+    [Política de devoluciones y garantía de frescura].
+ 2. Si el contexto no alcanza para responder, responde exactamente:
+    "No encuentro esa información en los documentos de Altamira." y no agregues nada más.
+ 3. Si la pregunta es ambigua, da la respuesta para cada caso que cubra el contexto.
+ 4. Máximo 3 oraciones, en español.
```

| Tipo | Pregunta | Antes (prompt original) | Después (prompt A3) |
|---|---|---|---|
| fácil | ¿Cuánto cuesta la libra de café Reserva? | $52.000 | $52.000 [Catálogo y precios de café 2026] |
| ambigua | ¿Cuánto cuesta el envío? | Varía según ubicación y monto: Valle de Aburrá $9.000 (gratis > $120.000); resto de Colombia $16.000, 2 a 4 días hábiles. No hay envíos internacionales. *(lista con viñetas y negritas)* | Varía según ubicación y monto. Valle de Aburrá $9.000, gratis en compras > $120.000. Resto de Colombia, $16.000. *(3 oraciones, sin cita)* |
| fuera de contexto | ¿Venden máquinas de espresso para el hogar? | «No, soy un modelo de lenguaje y no tengo información sobre la venta de máquinas de espresso para el hogar en el contexto proporcionado.» | «No encuentro esa información en los documentos de Altamira.» |

**Conclusión.**
- **Más efecto: la frase fija (regla 2).** Antes, la respuesta fuera de contexto era confusa: empieza con «No», como si la tienda no vendiera máquinas (algo que el corpus no dice), y mezcla un descargo de «soy un modelo de lenguaje». Después es exacta y fácil de detectar automáticamente.
- **La longitud (regla 4) también funcionó:** la respuesta ambigua pasó de una lista con viñetas a 3 oraciones con la misma información.
- **Ignorada a medias: la cita (regla 1).** Se cumplió en la pregunta fácil, pero no en la ambigua ni en la pregunta 3 de A2.
- **Efecto secundario:** la regla 2 volvió al modelo demasiado conservador. En A2-2 respondió «no sé» teniendo el documento correcto y, además, con una variante de la frase («No encontraré…»), así que tampoco la copió literal. Para corregirlo agregaría: «si el contexto contiene un dato relacionado, úsalo y explica la deducción».

## A4 · Muestra el score de similitud en las fuentes

_Evidencia:_ [`evidencias/A4_scores.png`](evidencias/A4_scores.png) · función `chat()` y celda «A4» del notebook 02b

Cambios: `build_pipeline` devuelve `(pipeline, chunks, vectorstore)`, y `chat()` llama a `vectorstore.similarity_search_with_score(question, k=K)`. Cada fuente se muestra con su score y con ✅ / ⛔ según supere `SCORE_THRESHOLD`.

| Pregunta | Tipo | top-1 | top-2 | top-3 |
|---|---|---|---|---|
| ¿Cuánto cuesta la libra de café Reserva? | buena | **0.401** · Catálogo y precios | 0.356 · Suscripción | 0.259 · Tueste y calidad |
| ¿Qué incluye la primera semana de un colaborador nuevo? | buena | **0.529** · Onboarding | 0.260 · Horarios | 0.132 · Suscripción |
| ¿Cuántos días tarda un envío a Bogotá? | buena | **0.490** · Envíos | 0.364 · Devoluciones | 0.349 · Mayoristas |
| ¿Venden máquinas de espresso para el hogar? | fuera | 0.248 · Catálogo | 0.203 · Mayoristas | 0.199 · Suscripción |
| ¿Quién ganó el mundial de fútbol de 2022? | fuera | 0.029 · Café verde | 0.024 · Tueste | 0.023 · Envíos |

> **Corrección importante (descubierta en B2):** las marcas ✅/⛔ de la interfaz comparan el **coseno** contra 0,3, pero el retriever `similarity_score_threshold` de `langchain-qdrant` compara **(1 + coseno) / 2**. Por eso, en la práctica, al LLM siempre le llegaron los 3 chunks. Las marcas muestran lo que *debería* filtrar el umbral, y la propuesta de abajo está en escala de coseno.

**Umbral propuesto: coseno 0,30, es decir `score_threshold = (1 + 0,30) / 2 = 0,65` en el retriever.**
- En esta tabla, un umbral de 0,35 separaría perfecto: las buenas tienen un top-1 ≥ 0,40 y las de fuera ≤ 0,25.
- Pero con los datos de A2 los rangos se solapan. La pregunta 4 (lote < 82 puntos), que es buena, tiene un top-1 de **0,304**, y la 7 (salario), que está fuera de contexto, tiene **0,318**. Un umbral de 0,35 habría dejado sin contexto a la pregunta 4 y no habría bloqueado a la 7.
- Los scores de este modelo son bajos en general (las buenas rondan 0,30–0,55), así que 0,30 deja pasar el documento correcto en todas las preguntas buenas que probé y descarta los de relleno (0,10–0,25).
- La decisión final la tiene que tomar el prompt: el umbral sirve para quitar ruido, no para detectar preguntas fuera de contexto.
- Un dato curioso: una pregunta totalmente ajena (fútbol) da scores ≈ 0,03, mientras que una «cercana pero sin respuesta» (máquinas de espresso) da ≈ 0,25. Lo difícil son las preguntas cercanas al dominio.

---

# Bloque B · RAG sobre papers

## B1 · Ejecuta el RAG de papers y evalúa con el ground truth

_Evidencia:_ [`notebooks/03_rag_workshop_papers_nvidia_nim_colab.ipynb`](notebooks/03_rag_workshop_papers_nvidia_nim_colab.ipynb) (Partes 0–5, Bonus y celda «B1 · Tabla»)

Índice: Attention **55** chunks + RAG **95** = **150** chunks de 900/120, embeddings de dimensión **2048**.

Nota: MarkItDown pega muchas palabras de estos PDFs (`WeusedtheAdamoptimizer[20]withβ =0.9`). Eso afecta a los embeddings y obliga a que las comprobaciones por palabra clave ignoren los espacios.

| # | Respuesta del RAG (resumida) | ¿Correcta? | ¿Paper correcto? | Chunks recuperados |
|---|---|:---:|:---:|---|
| Q1 | «28.4» | Sí (escueta: no dice que supera en más de 2 BLEU) | Sí | attention#28, #1, #26, #31, #24 |
| Q2 | Seq2seq preentrenado como memoria paramétrica + índice denso de Wikipedia vía retriever neuronal, entrenados end-to-end | Sí (no nombra BART ni DPR) | Sí | rag#8, #1, #7, #32, #50 |
| Q3 | Contrarresta productos punto grandes que llevan el softmax a zonas de gradientes muy pequeños | Sí | Sí | attention#11, #12, #13, #16, #10 |
| Q4 | RAG-Sequence: mismo documento para toda la secuencia; RAG-Token: un documento distinto por token | Sí | Sí | rag#10, #11, #24, #12, #91 |
| Q5 | 8 cabezas, dk = dv = 64 | Sí | Sí | attention#14, #31, #9, #39, #15 |
| Bonus | «RAG usa un generador seq2seq basado en el Transformer… implementado con HuggingFace Transformers» | **Parcial**: dice que es Transformer, pero **no nombra BART** | Solo RAG | rag#10, #12, #13, #86, #11 |

**Bonus:** los 5 fragmentos recuperados son del paper de RAG; **no apareció ningún chunk de Attention**. La pregunta es larga y casi todo su texto habla de RAG, FAISS y HNSW, así que el embedding queda «del lado» del paper de RAG. El modelo infirió bien que el generador es un Transformer, pero el chunk que dice «BART-large» no entró en el top-5. En la corrida de C3 con el mismo índice sí nombró BART: con temperatura 0,2 la respuesta varía entre ejecuciones.

## B2 · Experimentos con chunk_size, k y score_threshold

_Evidencia:_ celdas «B2» del notebook 03 (`build_store`, `build_chain` y el bucle de las 9 configuraciones)

| Experimento | Valor | chunks en índice | chunks recuperados (prom.) | Aciertos (de 5) | Observación |
|---|---:|---:|---:|---:|---|
| chunk_size | 400 | 326 | 5.0 | 5/5 | Indexar tardó 201 s (326 llamadas en lotes con reintentos) |
| chunk_size | 900 | 150 | 5.0 | 5/5 | Base del notebook |
| chunk_size | 1500 | 96 | 5.0 | 5/5 | Indexar tardó 31 s; contexto ~7 500 caracteres |
| k | 3 | 150 | 3.0 | 5/5 | El chunk clave ya está en el top-3 en las 5 preguntas |
| k | 5 | 150 | 5.0 | 5/5 | |
| k | 8 | 150 | 8.0 | 5/5 | Más contexto sin mejora |
| score_threshold | 0.15 | 150 | 5.0 | 5/5 | **Siempre llegan 5 chunks** |
| score_threshold | 0.25 | 150 | 5.0 | 5/5 | **Siempre llegan 5 chunks** |
| score_threshold | 0.40 | 150 | 5.0 | 5/5 | **Siempre llegan 5 chunks**, aunque Q3 tiene chunks con coseno 0,24–0,26 |

**Conclusión.**
- **El quiz es demasiado fácil para distinguir configuraciones.** Las 9 configuraciones aciertan 5/5 porque cada pregunta tiene un dato muy característico («BLEU», «RAG-Token», «sqrt(dk)») que el retriever encuentra con cualquier tamaño de chunk. La métrica más fina de C4 sí separa: hit@3 es 70 % con chunk_size=400 y 90 % con 900 y 1500.
- **`score_threshold` no filtró nada, y encontré por qué.** Con umbral 0,40 seguían llegando 5 chunks aunque Q3 tenía chunks con coseno 0,24. Lo verifiqué con vectores de prueba: en `langchain-qdrant`, el retriever `similarity_score_threshold` no compara el coseno, sino la «relevancia» = (1 + coseno) / 2.
  - Un umbral de 0,25 equivale a coseno ≥ −0,5, y uno de 0,40 a coseno ≥ −0,2.
  - Con este modelo los cosenos van de 0,1 a 0,6, así que **ningún umbral entre 0 y 0,5 filtra nada**. Esto también pasa en el notebook original del curso.
  - Para filtrar de verdad por coseno ≥ c hay que pedir `score_threshold = (1 + c) / 2`. Por ejemplo, coseno ≥ 0,30 → 0,65.
- **k:** con k=3 basta para estas preguntas. Subir a 8 solo agrega texto, y más contexto da más oportunidades de mezclar información de otro chunk.
- **chunk_size:** con 400 se pierde cohesión (baja hit@3); con 1500 se recupera más contexto relevante, pero cada consulta manda ~7 500 caracteres al LLM.
- **Configuración que propongo:** chunk_size 900–1500, k=3–5 y umbral corregido a (1 + 0,30) / 2 = 0,65. La del notebook funciona, pero su umbral es decorativo.

## B3 · Preguntas trampa: ¿el RAG admite que no sabe?

_Evidencia:_ celdas «B3» del notebook 03

| # | Pregunta trampa | Respuesta (prompt original) | Clasificación |
|---|---|---|---|
| 1 | What BLEU score did GPT-4 achieve on the WMT 2014 English-to-German task? | Los extractos no mencionan GPT-4; aclara que el Transformer obtuvo 28,4 | Admite que no sabe |
| 2 | How does RAG compare against Llama 2 70B on Natural Questions? | No hay evidencia; Llama 2 no aparece, RAG solo se compara con DPR, T5 y BART | Admite que no sabe |
| 3 | How many GPUs were used to train RAG-Token, and how many days did training take? | **«The training used 8 GPUs and took 3 days.»** | **Mezcla verdad con invención** |

En la 3, «8 GPUs» es cierto: el paper dice «distributed across 8, 32GB NVIDIA V100 GPUs». Pero «3 days» no aparece en el paper de RAG. Probablemente salió de la memoria del modelo o se contaminó con el paper de Attention, que sí habla de 3,5 días. Es la trampa más peligrosa porque mezcla un dato verdadero con uno inventado y suena creíble.

**Diff del prompt** (celda `SYSTEM_PROMPT_B3`):

```diff
- Answer using only the supplied academic-paper excerpts. If the excerpts do not contain
- enough evidence, say so explicitly. Answer in the same language as the question and be
- concise. Mention the paper title supporting the answer.
+ You answer questions about two academic papers using ONLY the excerpts in CONTEXT.
+ Never use prior knowledge, even if you are confident.
+ Every number, model name or result you mention must appear literally in the excerpts.
+ If the question asks about a model, dataset or result that is not mentioned in the excerpts
+ (for example, models published after these papers), reply exactly:
+ "The papers do not contain this information." (or its translation to the question's language)
+ and then, in one sentence, what the excerpts DO say that is closest.
+ Answer in the same language as the question, be concise, and cite the paper title in brackets.
```

| Pregunta | Después (prompt B3) | Clasificación |
|---|---|---|
| 1 · GPT-4 | «The papers do not contain this information. The excerpts state that the Transformer (big) achieved 28.4…» | Admite |
| 2 · Llama 2 | «The papers do not contain this information. The excerpts state that RAG sets a new state-of-the-art on… Natural Questions» | Admite |
| 3 · GPUs/días | «The papers do not contain this information. The excerpts state that training was distributed across 8, 32GB NVIDIA V100 GPUs.» | **Corregida**: no inventa los días y da el dato real |

La regla que más ayudó fue «cada número debe aparecer literalmente en los extractos». La pregunta de control Q1 siguió respondiendo «28.4», así que el prompt más estricto no rompió las respuestas válidas.

## B4 · Preguntas en español sobre papers en inglés

_Evidencia:_ celdas «B4» del notebook 03

| # | Chunks EN (score) | Chunks ES (score) | Coincidencia | Calidad EN | Calidad ES |
|---|---|---|:---:|:---:|:---:|
| Q1 | att#28 (0.58), #1 (0.49), #26 (0.46), #31 (0.41), #24 (0.38) | att#28 (0.53), #1 (0.50), #26 (0.46), #31 (0.40), #40 (0.37) | 4/5 | Correcta | Correcta, pero cita «atención_paper.pdf» (¡tradujo el nombre del archivo!) |
| Q3 | att#11 (0.42), #12 (0.41), #13 (0.40), #16 (0.26), #10 (0.24) | att#12 (0.40), #11 (0.40), #13 (0.40), #16 (0.25), #10 (0.23) | 5/5 | Correcta | Correcta y más completa |
| Q4 | rag#10 (0.48), #11 (0.47), #24 (0.42), #12 (0.38), #91 (0.38) | rag#10 (0.43), #11 (0.42), #24 (0.39), #91 (0.34), #12 (0.34) | 5/5 | Correcta | Correcta, con viñetas |

**Coseno entre la pregunta EN y su traducción ES:** Q1 = **0.848**, Q3 = **0.966**, Q4 = **0.929**. Como control, dos preguntas EN distintas (Q1 vs Q4) dan **0.101**.

**Hipótesis y comprobación.** `llama-nemotron-embed-vl-1b-v2` proyecta una pregunta y su traducción casi al mismo punto: coseno 0,85–0,97, frente a 0,10 entre preguntas distintas del mismo idioma. Por eso la búsqueda entre idiomas funciona: se recuperan los mismos chunks (14 de 15 coinciden) y en el mismo orden. Solo se pierde un poco de score (−0,01 a −0,05) porque los chunks están en inglés. Q1 es la que menos se parece (0,848): contiene la expresión «inglés-alemán», que en el texto aparece como «English-to-German»; las palabras traducidas mueven más el vector que los términos que se conservan (BLEU, RAG-Token, sqrt(dk)). Para comprobarlo más a fondo, repetiría con preguntas cuyo término clave sí se traduce («cabezas de atención» vs *attention heads*) y mediría la caída de coincidencias.

## B5 · Escribe tu propio quiz con ground truth

_Evidencia:_ celdas «B5 · Mi propio quiz» del notebook 03 (código + bloques `<details>` con ground truth)

Los ground truth los verifiqué leyendo el texto extraído de los PDFs.

| # | Pregunta | Ground truth | Sección | Respuesta RAG (resumida) | ¿Correcta? |
|---|---|---|---|---|:---:|
| 1 | What optimizer and learning rate schedule were used to train the Transformer, and how many warmup steps? | Adam (β1=0,9, β2=0,98, ε=10⁻⁹); lr sube lineal 4000 pasos y luego decae ∝ 1/√step | Attention §5.3 | Adam con β₁=0,9, β₂=0,98, ε=10⁻⁹; warmup de 4000 pasos y luego decae con la inversa de la raíz del paso | **Sí** |
| 2 | What knowledge source does RAG use as its non-parametric memory, and how is it split and indexed? | Wikipedia dic-2018, fragmentos de 100 palabras, 21M documentos, índice MIPS en FAISS con HNSW | RAG §3 y §2.2 | Wikipedia como índice vectorial denso, accedido por un retriever neuronal | **Parcial**: falta el dump, los fragmentos de 100 palabras, los 21M y FAISS/HNSW. El chunk con esos datos (rag#86) no entró en el top-5 |
| 3 *(cruza ambos)* | Which decoding strategies do the Transformer paper and the RAG paper use to generate their outputs? | Transformer: beam 4, α=0,6. RAG-Token: beam estándar; RAG-Sequence: Thorough / Fast Decoding | Attention §6.1 + RAG §2.5 | Transformer: beam search. RAG: Thorough Decoding para RAG-Sequence y beam estándar para RAG-Token | **Parcial**: los 5 chunks fueron de RAG; lo del Transformer es correcto pero sin beam=4 ni α=0,6 (probablemente de memoria del modelo) y omite Fast Decoding |

La pregunta que cruza papers vuelve a mostrar el problema del Bonus: con k=5, la búsqueda por similitud llena el top con un solo paper. C5 (MMR) y C3 (filtro por paper) atacan justo eso.

---

# Reflexión

## R1 · Reflexión final con tus datos

_Evidencia:_ tablas de A1–A4, B1–B5 y C1–C6 de este archivo

**1. ¿Qué ocurrió cuando el contexto no contenía información suficiente?**
- En la mayoría de los casos el modelo lo reconoció: la pregunta de vacaciones (A1), las de máquinas de espresso y salarios (A2) y las de GPT-4 y Llama 2 (B3) respondieron que no había información.
- Pero falló de dos formas opuestas.
  - **Inventó a medias:** en la trampa de las GPUs de RAG-Token respondió «8 GPUs y 3 días». Las 8 GPUs son reales, los 3 días no. Es el error más difícil de detectar porque viene acompañado de un dato verdadero.
  - **Fue demasiado prudente:** con el prompt estricto de A3 dijo «no encuentro esa información» sobre la cancelación de la suscripción, teniendo el documento correcto en primer lugar.
- Conclusión: el prompt controla esta frontera. Endurecerlo arregló la invención (B3), pero en otro caso aumentó los falsos «no sé» (A2-2).

**2. ¿Qué fragmentos se recuperaron para la pregunta bonus? ¿Aparecieron ambos papers?**
- No. Los 5 chunks fueron del paper de RAG (rag#10, #12, #13, #86, #11), con scores de 0,45 a 0,49, y ninguno de Attention.
- La pregunta menciona FAISS, HNSW y RAG, así que su embedding se parece mucho más a los chunks de RAG. La respuesta fue parcial: dijo que el generador es un Transformer, pero no nombró BART.
- Con MMR (C5) sí entró un chunk de Attention (att#0) y la redundancia bajó de 0,59 a 0,39. Es decir, la diversidad ayuda en preguntas que cruzan documentos.
- Lo mismo pasó en mi pregunta B5-3: los 5 chunks fueron de RAG.

**3. ¿Qué efecto tuvo modificar `chunk_size`, `chunk_overlap`, `k` o `score_threshold`?**
- Con las 5 preguntas del quiz, las 9 configuraciones de B2 acertaron 5/5. El quiz no discrimina.
- La métrica de C4, más fina, sí lo hizo: hit@3 subió de 70 % (chunk_size 400) a 90 % (900 y 1500), y hit@5 llegó a 100 % con 1500. Los chunks pequeños separan el dato de su contexto.
- Mi hallazgo principal fue que `score_threshold` no tuvo ningún efecto: con 0,15, 0,25 o 0,40 siempre llegaron 5 chunks. Lo verifiqué con vectores de prueba: `langchain-qdrant` aplica el umbral sobre (1 + coseno) / 2, así que 0,40 equivale a coseno ≥ −0,2 y no filtra nada.
- `k` solo cambió la cantidad de contexto: con k=3 ya bastaba.
- En mi corpus (A2), pasar de 400 a 600 caracteres dejó un chunk por documento y evitó fragmentos sin tema.

**4. ¿Por qué conviene separar el modelo de embeddings del modelo generativo?**
- Hacen trabajos distintos con costos distintos. El de embeddings (1B parámetros, vectores de 2048) se ejecuta una vez por chunk al indexar y una vez por pregunta. El generativo (30B, 3B activos) se ejecuta en cada respuesta.
- Gracias a la separación reutilicé el mismo índice con tres prompts distintos (original, A3 y B3) sin volver a calcular ni un embedding. También pude cambiar de modelo generativo a mitad del taller (de nemotron-3.5-super a lightning) sin reindexar.
- Lo inverso también importa: cambiar el modelo de embeddings obliga a reindexar todo, porque los vectores de modelos distintos no son comparables. Cada uno se puede elegir y escalar según su función; por ejemplo, el de embeddings tiene que ser multilingüe (B4 mostró cosenos de 0,85–0,97 entre EN y ES).

**5. ¿Qué cambiaría para llevar este prototipo a producción?**
- **Persistencia:** el índice en memoria obliga a pagar ~87 s de embeddings en cada reinicio, frente a 14 s para reabrirlo de disco (C6). Además, el modo local de Qdrant solo admite un proceso, así que usaría un servidor Qdrant con snapshots.
- **Control de acceso:** aplicaría filtros de Qdrant sobre la metadata, como hice con `metadata.source` en C3, para que cada usuario solo recupere documentos de su `department`.
- **Evaluación:** un conjunto de preguntas con ground truth y una métrica automática (hit@k de C4) que se ejecute en cada cambio. El quiz de 5 preguntas no detectaba diferencias que hit@k sí detectó.
- **Observabilidad:** registrar pregunta, chunks, scores, prompt y respuesta, como muestra la interfaz de A4. Así se habrían detectado antes el umbral que no filtraba y los 502 del endpoint de embeddings (~40 % de las llamadas, que tuve que resolver con reintentos y timeout).
- **Costos:** cachear embeddings de consultas repetidas (lo hice en el notebook), elegir un modelo generativo más pequeño cuando alcanza (lightning respondió bien todo el quiz) y, a gran escala, usar un índice ANN (C1 muestra 15–43× de aceleración con 500 000 vectores).

---

# Retos

## C1 · HNSW contra fuerza bruta con FAISS

_Evidencia:_ `notebooks/04_challenges.ipynb` (ejecutado) · `evidencias/C1_recall.png`

Configuración: d = 64, 200 consultas, k = 10, efConstruction = 100, datos uniformes en [0,1). Las búsquedas se midieron con 1 hilo (la construcción con 4) en el contenedor donde preparé el repo; en Colab los tiempos absolutos cambian pero las tendencias deberían mantenerse.

**Barrido M × efSearch (n = 20 000; exacto = 0.186 ms/consulta)**

| M | efSearch | recall@10 | ms/consulta | construcción (s) |
|---:|---:|---:|---:|---:|
| 8 | 10 | 0.168 | 0.017 | 0.60 |
| 8 | 64 | 0.510 | 0.075 | 0.60 |
| 8 | 256 | 0.819 | 0.241 | 0.60 |
| 16 | 10 | 0.329 | 0.016 | 0.84 |
| 16 | 16 | 0.425 | 0.022 | 0.84 |
| 16 | 32 | 0.599 | 0.041 | 0.84 |
| 16 | 64 | 0.770 | 0.072 | 0.84 |
| 16 | 128 | 0.905 | 0.135 | 0.84 |
| 16 | 256 | **0.971** | 0.293 | 0.84 |
| 32 | 10 | 0.468 | 0.027 | 1.06 |
| 32 | 64 | 0.909 | 0.155 | 1.06 |
| 32 | 128 | 0.983 | 0.227 | 1.06 |
| 32 | 256 | 0.998 | 0.426 | 1.06 |

(La tabla completa con los 18 puntos está en el notebook.)

![Recall vs efSearch y recall vs latencia](evidencias/C1_recall.png)

**¿Qué efSearch para recall ≥ 0.95 con M = 16?** En el barrido, **ef = 256** (0.971); con 128 se queda en 0.905. Con M = 32 basta ef = 128 (0.983). Un M mayor da más aristas por nodo, así que cada paso explora más y necesita una cola menor, a cambio de más memoria y más tiempo de construcción (0.60 s → 1.06 s).

**¿Desde qué n HNSW es más rápido que el exacto? (M = 16)**

| n | efSearch | exacto ms/q | HNSW ms/q | aceleración | recall@10 | construcción (s) |
|---:|---:|---:|---:|---:|---:|---:|
| 20 000 | 64 | 0.201 | 0.105 | 1.9× | 0.776 | 0.9 |
| 20 000 | 128 | 0.201 | 0.146 | 1.4× | 0.910 | 0.9 |
| 20 000 | 256 | 0.201 | 0.301 | 0.7× | 0.970 | 0.9 |
| 100 000 | 64 | 1.798 | 0.190 | 9.5× | 0.528 | 5.9 |
| 100 000 | 128 | 1.798 | 0.307 | 5.9× | 0.713 | 5.9 |
| 100 000 | 256 | 1.798 | 0.639 | 2.8× | 0.848 | 5.9 |
| 500 000 | 64 | 14.610 | 0.339 | 43.1× | 0.340 | 45.1 |
| 500 000 | 128 | 14.610 | 0.543 | 26.9× | 0.492 | 45.1 |
| 500 000 | 256 | 14.610 | 0.961 | 15.2× | 0.634 | 45.1 |

- Con **n = 20 000** HNSW solo gana si aceptas recall < 0.95: con el ef necesario para 0.95 (256) es **más lento** que el exacto (0.30 vs 0.20 ms). A ese tamaño la fuerza bruta es un recorrido secuencial de 5 MB que cabe en caché y la SIMD lo hace muy rápido.
- Con **n = 100 000** HNSW ya es 3–10× más rápido, y con **n = 500 000** es 15–43× más rápido: el exacto crece linealmente (y peor, cuando los datos dejan de caber en caché) y HNSW casi no crece (0.29 → 0.96 ms con ef = 256).
- El precio: **con ef fijo el recall cae al crecer n** (ef = 256: 0.97 → 0.85 → 0.63). Para mantener recall ≥ 0.95 con 500 000 vectores aleatorios habría que subir ef o M bastante, y construir el índice ya tarda 45 s frente a 0 s del exacto.

**¿Por qué los datos aleatorios son un caso difícil?** En 64 dimensiones uniformes todas las distancias se concentran alrededor del mismo valor (maldición de la dimensionalidad): el vecino 10 está casi tan lejos como el 1000, así que no hay «vecindarios» claros que el grafo pueda aprovechar y el descenso greedy se equivoca a menudo. Lo comprobé con un tercer experimento (n = 20 000, M = 16), con datos agrupados: 50 clusters en un subespacio de 8 dimensiones dentro de R⁶⁴, con base y consultas de la misma distribución:

| datos | ef = 10 | ef = 16 | ef = 32 | ef = 64 |
|---|---:|---:|---:|---:|
| uniforme | 0.309 | 0.411 | 0.607 | 0.792 |
| agrupado | **0.966** | 0.990 | 0.998 | 1.000 |

Con estructura, ef = 10 ya da 0.97 de recall. Los **embeddings reales de texto** se parecen al caso agrupado: viven cerca de una variedad de dimensión intrínseca mucho menor que la nominal (temas, idiomas, estilos forman clusters). Por eso en la práctica HNSW con ef ≈ 50–100 suele dar > 95 % de recall, como dice la presentación, aunque con datos aleatorios necesité 256.

## C2 · Un tercer paper: el RAG que explica HNSW

No elegido.

## C3 · App Gradio para los papers con filtro por paper

_Evidencia:_ sección «C3» del notebook 03 · [`evidencias/C3_gradio.png`](evidencias/C3_gradio.png)

La app tiene un `gr.Dropdown` con «Todos», «Attention Is All You Need» y «Retrieval-Augmented Generation». El filtro se aplica **dentro de Qdrant**, pasando `models.Filter(must=[FieldCondition(key="metadata.source", match=MatchValue(...))])` a `similarity_search_with_score`. Así siempre llegan hasta k chunks del paper elegido; si se filtrara después de recuperar, podrían quedar 0. Cada fuente se muestra con título, archivo, chunk y score.

La captura muestra una versión corta de la pregunta, sin las premisas del bonus («What seq2seq model does RAG use as its generator component, and is it based on the Transformer?»), con el filtro «Retrieval-Augmented Generation». Recupera los chunks 10, 11, 12, 8 y 7 (score 0,47–0,56) y responde **«RAG uses BART as its seq2seq generator, and it is based on the Transformer architecture.»**: correcta y completa.

Comparación con la pregunta bonus completa:

| Filtro | Chunks (score) | Respuesta |
|---|---|---|
| Todos | rag#10 (0.493), #12, #13, #86, #11 (0.448) | «…a pre-trained encoder-decoder model (**BART**), which is Transformer-based. The paper initializes the generator from a BART checkpoint…» ✅ |
| Attention Is All You Need | att#8 (0.379), #0, #40, #39, #2 (0.327) | «The RAG paper uses a seq2seq model with a Transformer-based generator.» ⚠️ Los chunks de Attention no hablan de RAG: el modelo respondió con lo que dice la **propia pregunta** (que trae premisas sobre RAG) |
| Retrieval-Augmented Generation | igual que «Todos» | Dice que es Transformer, pero no nombra BART |

- **Sin filtro**, el paper de RAG domina por completo (los 5 chunks), así que «Todos» y «RAG» recuperan lo mismo. Aun así, las respuestas cambiaron entre ejecuciones (una nombró BART y otra no): con temperatura 0,2 el modelo no es determinista.
- **Con el filtro de Attention**, el contexto no tenía la respuesta y el modelo no lo admitió. Usó las premisas incluidas en la pregunta. Un filtro por metadata cambia qué evidencia llega, pero la honestidad sigue dependiendo del prompt.

## C4 · Evaluación automática de la recuperación

_Evidencia:_ [`eval/preguntas.csv`](eval/preguntas.csv) (10 preguntas) · sección «C4» del notebook 03

La función `hit_at_k` cuenta un acierto si algún chunk recuperado viene de `fuente_esperada` **y** contiene `palabra_clave`. La comparación ignora espacios y mayúsculas, porque MarkItDown pega palabras (`21Mdocuments`, `h=8`). Antes verifiqué que las 10 palabras clave aparecen en el texto extraído del paper correcto. Los retrievers van sin umbral, para medir solo el ranking.

| chunk_size | chunks | hit@3 | hit@5 |
|---:|---:|---:|---:|
| 400 | 326 | 70 % | 90 % |
| 900 | 150 | 90 % | 90 % |
| 1500 | 96 | 90 % | **100 %** |

Con chunks más grandes el resultado mejora: cada chunk tiene más contexto alrededor del dato, y hay menos chunks compitiendo por el top-k. Con 900 y k=5 la única pregunta fallida fue la 5 («What regularization techniques…», clave «label smoothing»). Recuperó chunks de entrenamiento (att#29, #25) pero no el de regularización. Es una pregunta conceptual con una respuesta repartida en varios párrafos.

**Limitaciones de la métrica:**
1. **Palabra clave ≠ respuesta.** Un chunk puede contener «28.4» en otro contexto (falso positivo) o la respuesta parafraseada sin la palabra exacta (falso negativo). Además, con chunks grandes es más fácil «acertar» solo porque hay más texto por chunk: parte de la mejora de 1500 puede venir de eso, y el contexto más largo es más ruidoso para el LLM.
2. **Mide la recuperación, no la respuesta.** hit@k = 100 % no garantiza que el LLM use el chunk correcto ni que no invente (B5-2 tuvo el dato en rag#86 a k=5 y aun así respondió de forma incompleta). Tampoco pondera la posición, como sí lo harían MRR o nDCG. Y con 10 preguntas, cada una mueve el resultado 10 puntos.

## C5 · MMR frente a similitud pura

_Evidencia:_ sección «C5» del notebook 03

Compara k=5 por similitud contra MMR (`fetch_k=20`, `lambda_mult=0.5`). La redundancia es el coseno medio entre los pares de chunks recuperados.

| Pregunta | Estrategia | Chunks | Papers distintos | Redundancia |
|---|---|---|:---:|---:|
| Q2 (amplia) | similitud | rag#8, #1, #7, #32, #50 | 1 | 0.672 |
| Q2 (amplia) | MMR | rag#8, #41, #42, #32, #50 | 1 | **0.480** |
| Q5 (dato puntual) | similitud | att#14, #31, #9, #39, #15 | 1 | 0.490 |
| Q5 (dato puntual) | MMR | att#14, #31, #0, #35, #40 | 1 | 0.434 |
| Bonus (dos papers) | similitud | rag#10, #12, #13, #86, #11 | 1 | 0.590 |
| Bonus (dos papers) | MMR | rag#10, **att#0**, rag#40, #86, #25 | **2** | **0.391** |

- **Dónde más ayuda:** en la pregunta Bonus, que cruza papers. Es la única donde aparece un chunk de Attention (att#0, el abstract) y la redundancia baja un 34 %. En Q2 baja un 29 %: la similitud traía rag#1, #7 y #8, tres versiones del abstract y la introducción que dicen lo mismo.
- **Dónde empeora:** en preguntas puntuales como Q5. El dato está en att#14 (que MMR conserva primero), pero MMR cambia chunks cercanos (#9, #15, sobre attention) por otros diversos y menos relevantes (#0 abstract, #35, #40). No pierde la respuesta, pero llena el contexto de ruido.
- **Respuestas al Bonus:** ninguna estrategia hizo que el modelo nombrara BART. La versión MMR sí menciona el paper de Attention como origen del Transformer, que era el objetivo de cruzar documentos.

## C6 · Persistencia del índice en disco

_Evidencia:_ sección «C6» del notebook 03. La segunda celda se ejecutó en un **kernel nuevo**, solo con la celda de configuración, sin descargar PDFs ni indexar.

| Operación | Tiempo | Llamadas a la API de embeddings | Respuesta de control (Q1) |
|---|---:|---|---|
| Indexar (primera vez, `QdrantClient(path="./qdrant_data")`) | **87.2 s** | 150 chunks (+ reintentos por 502) | 28.4 |
| Reabrir tras reiniciar el kernel | **14.0 s** | 0 para documentos (solo la pregunta) | 28.4 |

Reabrir es 6 veces más rápido y no gasta cuota de embeddings. Los 14 s son de cargar los 150 vectores de 2048 dimensiones desde el almacenamiento local de Qdrant (SQLite y deserialización), no de la API.

**Qué sigue faltando para producción:**
- **Concurrencia:** el modo local de Qdrant bloquea la carpeta para un solo proceso. Por eso la celda cierra el cliente anterior antes de abrir otro, y dos workers no podrían compartirla.
- **Seguridad:** no hay control de acceso ni autenticación.
- **Durabilidad:** en Colab, `./qdrant_data` desaparece si se recicla la máquina. Haría falta Google Drive o, mejor, un servidor Qdrant con volumen y **snapshots** de respaldo.
- **Versionado:** la colección no registra con qué modelo de embeddings ni con qué chunking se creó. Si cambia cualquiera de los dos, los vectores guardados quedan inválidos sin que nada avise.
