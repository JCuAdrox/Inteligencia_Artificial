# Workshop RAG · SI3003 Inteligencia Artificial

- **Nombre:** …
- **Código estudiantil:** …
- **Usuario de GitHub:** JeronimoCH
- **Fecha:** 21 de octubre de 2026
- **Entrega a:** jipadillab@eafit.edu.co

## Progreso

✅ completa con evidencia · 🟡 código listo y probado, falta ejecutarlo en Colab con la API key y pegar resultados · ⬜ no elegida.

| ID | Tarea | Puntos | Hecha | Evidencia |
|---|---|---:|:---:|---|
| T1 | Cuántos nodos viven en cada capa | 2 | ✅ | RESPUESTAS.md |
| T2 | ef, k y el tamaño de la cola | 2 | ✅ | RESPUESTAS.md |
| T3 | Inserción frente a búsqueda | 2 | ✅ | RESPUESTAS.md |
| T4 | El invariante de FAISSVectorStore | 2 | ✅ | RESPUESTAS.md |
| T5 | El costo de la fuerza bruta | 2 | ✅ | RESPUESTAS.md |
| A1 | Ejecuta la app RAG de punta a punta | 4 | 🟡 | notebooks/02b · evidencias/A1_gradio.png |
| A2 | Tu propio corpus de documentos | 6 | 🟡 | data/company_documents.json · evidencias/A2_corpus.png |
| A3 | Ingeniería del prompt (antes y después) | 5 | 🟡 | notebooks/02b (celda A3) |
| A4 | Muestra el score de similitud en las fuentes | 5 | 🟡 | notebooks/02b · evidencias/A4_scores.png |
| B1 | Ejecuta el RAG de papers y evalúa con el ground truth | 6 | 🟡 | notebooks/03 (celda B1) |
| B2 | Experimentos con chunk_size, k y score_threshold | 8 | 🟡 | notebooks/03 (celdas B2) |
| B3 | Preguntas trampa: ¿el RAG admite que no sabe? | 6 | 🟡 | notebooks/03 (celdas B3) |
| B4 | Preguntas en español sobre papers en inglés | 5 | 🟡 | notebooks/03 (celdas B4) |
| B5 | Escribe tu propio quiz con ground truth | 5 | 🟡 | notebooks/03 (celdas B5) |
| R1 | Reflexión final con tus datos | 10 | 🟡 | RESPUESTAS.md#reflexión |
| C1 | HNSW contra fuerza bruta con FAISS (reto) | 15 | ✅ | notebooks/04_challenges.ipynb · evidencias/C1_recall.png |
| C2 | Un tercer paper: el RAG que explica HNSW (reto) | 12 | ⬜ | — |
| C3 | App Gradio para los papers con filtro por paper (reto) | 15 | 🟡 | notebooks/03 (C3) · evidencias/C3_gradio.png |
| C4 | Evaluación automática de la recuperación (reto) | 12 | 🟡 | eval/preguntas.csv · notebooks/03 (C4) |
| C5 | MMR frente a similitud pura (reto) | 8 | 🟡 | notebooks/03 (C5) |
| C6 | Persistencia del índice en disco (reto) | 8 | 🟡 | notebooks/03 (C6) |

**Autoevaluación:** … / 100 puntos (actualizar al terminar las tareas 🟡).

**Retos elegidos:** C1, C3, C4, C5, C6 (tope 30 puntos).

## Cómo ejecutar mis notebooks

1. Abre cada notebook de `notebooks/` en Google Colab (Archivo → Abrir notebook → GitHub).
2. Crea el secreto `NVIDIA_API_KEY` en Colab (panel Secrets) con acceso para el notebook.
3. Cambia `REPO_RAW_BASE` por la URL raw de este repo (02b sección 3 y 03 sección C4) para que se descarguen el corpus y `eval/preguntas.csv`.
4. Ejecuta las celdas en orden. Las celdas de evaluación imprimen tablas en Markdown para `RESPUESTAS.md`.
5. `04_challenges.ipynb` (C1) no necesita API key.

> No incluyas tu API key en ningún archivo de este repositorio.

## Decisiones y aprendizajes

Escribe 5 a 8 líneas sobre lo más importante que descubriste hoy.
