# Workshop RAG · SI3003 Inteligencia Artificial

- **Nombre:** …
- **Código estudiantil:** …
- **Usuario de GitHub:** JeronimoCH
- **Fecha:** 21 de octubre de 2026
- **Entrega a:** jipadillab@eafit.edu.co

## Progreso

✅ completa con evidencia en el repo · ⬜ no elegida.

| ID | Tarea | Puntos | Hecha | Evidencia |
|---|---|---:|:---:|---|
| T1 | Cuántos nodos viven en cada capa | 2 | ✅ | RESPUESTAS.md |
| T2 | ef, k y el tamaño de la cola | 2 | ✅ | RESPUESTAS.md |
| T3 | Inserción frente a búsqueda | 2 | ✅ | RESPUESTAS.md |
| T4 | El invariante de FAISSVectorStore | 2 | ✅ | RESPUESTAS.md |
| T5 | El costo de la fuerza bruta | 2 | ✅ | RESPUESTAS.md |
| A1 | Ejecuta la app RAG de punta a punta | 4 | ✅ | notebooks/02b_rag_app_original_A1.ipynb · evidencias/A1_gradio.png |
| A2 | Tu propio corpus de documentos | 6 | ✅ | data/company_documents.json · evidencias/A2_corpus.png |
| A3 | Ingeniería del prompt (antes y después) | 5 | ✅ | notebooks/02b (celda A3) |
| A4 | Muestra el score de similitud en las fuentes | 5 | ✅ | notebooks/02b · evidencias/A4_scores.png |
| B1 | Ejecuta el RAG de papers y evalúa con el ground truth | 6 | ✅ | notebooks/03 (celda B1) |
| B2 | Experimentos con chunk_size, k y score_threshold | 8 | ✅ | notebooks/03 (celdas B2) |
| B3 | Preguntas trampa: ¿el RAG admite que no sabe? | 6 | ✅ | notebooks/03 (celdas B3) |
| B4 | Preguntas en español sobre papers en inglés | 5 | ✅ | notebooks/03 (celdas B4) |
| B5 | Escribe tu propio quiz con ground truth | 5 | ✅ | notebooks/03 (celdas B5) |
| R1 | Reflexión final con tus datos | 10 | ✅ | RESPUESTAS.md#reflexión |
| C1 | HNSW contra fuerza bruta con FAISS (reto) | 15 | ✅ | notebooks/04_challenges.ipynb · evidencias/C1_recall.png |
| C2 | Un tercer paper: el RAG que explica HNSW (reto) | 12 | ⬜ | — |
| C3 | App Gradio para los papers con filtro por paper (reto) | 15 | ✅ | notebooks/03 (C3) · evidencias/C3_gradio.png |
| C4 | Evaluación automática de la recuperación (reto) | 12 | ✅ | eval/preguntas.csv · notebooks/03 (C4) |
| C5 | MMR frente a similitud pura (reto) | 8 | ✅ | notebooks/03 (C5) |
| C6 | Persistencia del índice en disco (reto) | 8 | ✅ | notebooks/03 (C6) |

**Autoevaluación:** 100 / 100 puntos (núcleo 70 + retos 30 de 58 hechos: C1 15, C3 15, C4 12, C5 8, C6 8; el tope es 30).

**Retos elegidos:** C1, C3, C4, C5, C6 (tope 30 puntos).

## Cómo ejecutar mis notebooks

Los notebooks están guardados **con sus salidas**. Para reproducirlos:

1. Abre cada notebook de `notebooks/` en Google Colab (Archivo → Abrir notebook → GitHub).
2. Crea el secreto `NVIDIA_API_KEY` en Colab (panel Secrets) con acceso para el notebook.
3. El corpus y `eval/preguntas.csv` se descargan solos de este repo (`REPO_RAW_BASE`).
4. Ejecuta las celdas en orden. Las celdas de evaluación imprimen tablas en Markdown para `RESPUESTAS.md`.
5. `04_challenges.ipynb` (C1) no necesita API key.

> No incluyas tu API key en ningún archivo de este repositorio.

## Decisiones y aprendizajes

- **El `score_threshold` del notebook no filtraba nada.** `langchain-qdrant` compara (1 + coseno) / 2 contra el umbral, así que 0,25–0,40 equivale a coseno negativo. Lo noté porque en B2 los tres umbrales devolvían siempre 5 chunks, y lo confirmé con vectores de prueba. Para filtrar por coseno ≥ c hay que usar (1 + c) / 2.
- **El prompt decide la honestidad, y hay un equilibrio.** Endurecerlo eliminó una alucinación («8 GPUs y 3 días», B3), pero en otro caso causó un falso «no sé» con el documento correcto en la mano (A2-2).
- **Las preguntas que cruzan documentos son el punto débil.** La búsqueda por similitud llenó el top-5 con un solo paper (Bonus, B5-3); MMR metió el otro paper y bajó la redundancia un 34 % (C5).
- **El embedding es multilingüe:** pregunta EN y su traducción ES tienen coseno 0,85–0,97 y recuperan los mismos chunks (B4).
- **Medir bien importa:** el quiz de 5 preguntas dio 5/5 en las 9 configuraciones; hit@k con 10 preguntas sí distinguió chunk_size 400 (70 %) de 900/1500 (90 %) (C4).
- **HNSW solo compensa a escala:** con 20 000 vectores la búsqueda exacta fue más rápida; con 500 000, HNSW fue 15–43 veces más rápido, perdiendo recall (C1).
- **Detalles prácticos:** Nemotron 3.5 necesita `enable_thinking=False` (si no, devuelve respuestas vacías) y el endpoint de embeddings falló en ~40 % de las llamadas, así que agregué reintentos, timeout corto y caché.
