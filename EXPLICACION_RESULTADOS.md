# Explicación de los resultados

Resumen en lenguaje sencillo de qué pasó en cada tarea y por qué. Los números vienen de los notebooks ejecutados en `notebooks/`. El detalle completo, con tablas, está en [`RESPUESTAS.md`](RESPUESTAS.md).

> Documento preparado con ayuda de Claude (Claude Code) a partir de los resultados del repositorio.

## Teoría (T1–T5)

- **T1 · Capas de HNSW:** cada capa tiene unas 16 veces menos nodos que la de abajo (62 500 → 3 906 → 244). Los pocos nodos de arriba funcionan como «autopista»: con pocos saltos llegas cerca del resultado, y abajo afinas.
- **T2 · ef y k:** `ef` es cuántos candidatos se guardan mientras se busca. Si pides 10 resultados (`k`) con `ef=5`, no se pueden devolver 10. `ef` se cambia en producción sin reconstruir el índice; `ef_construction` no, porque ya quedó en las conexiones del grafo.
- **T3 · Inserción frente a búsqueda:** la inserción sortea una capa y modifica el grafo; la búsqueda solo lo recorre, entrando siempre por la capa más alta.
- **T4 · El invariante de FAISS:** si se borra un documento de la lista pero no del índice, todo se corre una posición. El sistema devuelve textos equivocados sin dar error. Qdrant lo evita porque guarda el texto junto a su vector.
- **T5 · Fuerza bruta:** 7,68×10¹⁰ multiplicaciones por segundo. Por eso se usa búsqueda aproximada (ANN): se pierde un poco de exactitud y se gana mucha velocidad.

## Bloque A · Corpus propio (Tostadora Altamira)

- **A1:** las 4 preguntas de ejemplo salieron bien y la pregunta de vacaciones se rechazó. Siempre llegaban 3 fuentes porque el umbral no filtraba (ver B2); lo que evitó que el modelo inventara fue el prompt.
- **A2 (6 de 7):** falló «¿hasta qué día cancelo la suscripción?». El documento correcto estaba, pero el prompt estricto volvió al modelo demasiado prudente y respondió «no encuentro esa información».
- **A3:** la frase fija para «no sé» fue la regla con más efecto. Antes el modelo respondía cosas confusas como «No, soy un modelo de lenguaje…». La cita entre corchetes la cumplió solo a veces.
- **A4:** las preguntas buenas dan coseno 0,30–0,53 y las de fuera de tema 0,03–0,32. Se solapan, así que ningún umbral las separa perfecto: el umbral quita ruido y el prompt decide.

## Bloque B · Papers

- **B1:** Q1 a Q5 correctas. La bonus fue parcial: no nombró BART, y los 5 fragmentos fueron del paper de RAG, ninguno de Attention.
- **B2 (hallazgo principal):** las 9 configuraciones acertaron 5/5, y con cualquier umbral siempre llegaban 5 fragmentos. La razón es que `langchain-qdrant` compara (1 + coseno) / 2 contra el umbral, no el coseno, así que 0,4 equivale a «casi todo pasa». Para filtrar coseno ≥ 0,30 hay que poner 0,65.
- **B3:** en la pregunta sobre cuántas GPUs y días usó RAG-Token respondió «8 GPUs y 3 días». Las 8 GPUs son ciertas y los 3 días son inventados. Con el prompt nuevo, que exige que cada número aparezca en el texto, ya no inventó.
- **B4:** la pregunta en español y su versión en inglés tienen coseno 0,85–0,97 y recuperan casi los mismos fragmentos (14 de 15). El modelo de embeddings es multilingüe.
- **B5:** la pregunta 1 salió bien. La 2 y la 3 fueron parciales: el dato estaba en un fragmento que no quedó entre los 5 primeros. La pregunta que mezcla los dos papers vuelve a traer solo fragmentos de RAG.

## Retos

- **C1 · FAISS:** con 20 000 vectores la búsqueda exacta es más rápida; con 500 000, HNSW es 15 a 43 veces más rápido, pero pierde recall. Con datos agrupados, que se parecen más a texto real, HNSW casi no pierde.
- **C3 · Filtro por paper:** funciona dentro de Qdrant. Si se filtra a Attention y se pregunta por RAG, el modelo responde con lo que dice la propia pregunta, no con los documentos.
- **C4 · hit@k:** con chunks de 400 caracteres, solo el 70 % de las preguntas tiene la respuesta entre los 3 primeros; con 900 o 1500, el 90 %. Los chunks más grandes conservan mejor el contexto.
- **C5 · MMR:** reduce los fragmentos repetidos (la redundancia baja de 0,59 a 0,39 en la bonus) y trajo un fragmento de Attention. En preguntas de un dato puntual solo agrega ruido.
- **C6 · Persistencia:** indexar tarda 87 s y reabrir el índice desde disco 14 s, sin volver a pagar embeddings.
