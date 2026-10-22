# Workshop RAG · SI3003 Inteligencia Artificial

- **Nombre:** Jerónimo Zapata Cuadros
- **Código estudiantil:** …
- **Usuario de GitHub:** [`JCuAdrox`](https://github.com/JCuAdrox)
- **Fecha:** 22 de octubre de 2026
- **Entrega a:** jipadillab@eafit.edu.co

Las respuestas, tablas y análisis están en [RESPUESTAS.md](RESPUESTAS.md).

## Progreso

✅ hecha con evidencia ejecutada · 🟡 implementada y probada, falta la corrida contra NVIDIA NIM (en el equipo donde se preparó no había `NVIDIA_API_KEY`)

| ID | Tarea | Puntos | Hecha | Evidencia |
|---|---|---:|:---:|---|
| T1 | Cuántos nodos viven en cada capa | 2 | ✅ | [RESPUESTAS T1](RESPUESTAS.md#t1--cuántos-nodos-viven-en-cada-capa) · simulación en [04_challenges](notebooks/04_challenges.ipynb) |
| T2 | ef, k y el tamaño de la cola | 2 | ✅ | [RESPUESTAS T2](RESPUESTAS.md#t2--ef-k-y-el-tamaño-de-la-cola) |
| T3 | Inserción frente a búsqueda | 2 | ✅ | [RESPUESTAS T3](RESPUESTAS.md#t3--inserción-frente-a-búsqueda) |
| T4 | El invariante de FAISSVectorStore | 2 | ✅ | [RESPUESTAS T4](RESPUESTAS.md#t4--el-invariante-de-faissvectorstore) |
| T5 | El costo de la fuerza bruta | 2 | ✅ | [RESPUESTAS T5](RESPUESTAS.md#t5--el-costo-de-la-fuerza-bruta) |
| A1 | Ejecuta la app RAG de punta a punta | 4 | 🟡 | [02b](notebooks/02b_rag_app_nvidia_nim_colab.ipynb) · `evidencias/A1_gradio.png` |
| A2 | Tu propio corpus de documentos | 6 | 🟡 | [company_documents.json](data/company_documents.json) · [02b](notebooks/02b_rag_app_nvidia_nim_colab.ipynb) · `evidencias/A2_corpus.png` |
| A3 | Ingeniería del prompt (antes y después) | 5 | 🟡 | [02b, sección 7](notebooks/02b_rag_app_nvidia_nim_colab.ipynb) |
| A4 | Muestra el score de similitud en las fuentes | 5 | 🟡 | [02b, sección 8](notebooks/02b_rag_app_nvidia_nim_colab.ipynb) · `evidencias/A4_scores.png` · experimento de escala del umbral ✅ en [04](notebooks/04_challenges.ipynb) |
| B1 | Ejecuta el RAG de papers y evalúa con el ground truth | 6 | 🟡 | [03](notebooks/03_rag_workshop_papers_nvidia_nim_colab.ipynb) |
| B2 | Experimentos con chunk_size, k y score_threshold | 8 | 🟡 | [03, B2](notebooks/03_rag_workshop_papers_nvidia_nim_colab.ipynb) · `evidencias/B2_resultados.csv` |
| B3 | Preguntas trampa: ¿el RAG admite que no sabe? | 6 | 🟡 | [03, B3](notebooks/03_rag_workshop_papers_nvidia_nim_colab.ipynb) |
| B4 | Preguntas en español sobre papers en inglés | 5 | 🟡 | [03, B4](notebooks/03_rag_workshop_papers_nvidia_nim_colab.ipynb) |
| B5 | Escribe tu propio quiz con ground truth | 5 | 🟡 | [03, B5](notebooks/03_rag_workshop_papers_nvidia_nim_colab.ipynb) · [RESPUESTAS B5](RESPUESTAS.md#b5--escribe-tu-propio-quiz-con-ground-truth) |
| R1 | Reflexión final con tus datos | 10 | 🟡 | [RESPUESTAS R1](RESPUESTAS.md#r1--reflexión-final-con-tus-datos) (puntos 4–5 completos; 1–3 esperan los datos) |
| C1 | HNSW contra fuerza bruta con FAISS (reto) | 15 | ✅ | [04_challenges](notebooks/04_challenges.ipynb) · [gráfica 1](evidencias/C1_recall_vs_ef_y_latencia.png) · [gráfica 2](evidencias/C1_escalamiento_n.png) · [CSV](evidencias/C1_resultados.csv) |
| C2 | Un tercer paper: el RAG que explica HNSW (reto) | 12 | ⬜ | no lo hice |
| C3 | App Gradio para los papers con filtro por paper (reto) | 15 | 🟡 | [03, C3](notebooks/03_rag_workshop_papers_nvidia_nim_colab.ipynb) · `evidencias/C3_gradio.png` |
| C4 | Evaluación automática de la recuperación (reto) | 12 | 🟡 | [preguntas.csv](eval/preguntas.csv) (palabras clave verificadas ✅) · [04, C4](notebooks/04_challenges.ipynb) |
| C5 | MMR frente a similitud pura (reto) | 8 | 🟡 | [03, C5](notebooks/03_rag_workshop_papers_nvidia_nim_colab.ipynb) |
| C6 | Persistencia del índice en disco (reto) | 8 | 🟡 | [04, C6](notebooks/04_challenges.ipynb) |

**Autoevaluación:** 25 / 100 puntos con evidencia ejecutada hoy (T1–T5 y C1). El resto está implementado; al correr los notebooks con la API, el núcleo suma 70 y los retos llegan al tope de 30.

## Cómo ejecutar mis notebooks

**En Google Colab**

1. Abre cada notebook de `notebooks/` en Colab.
2. Crea el secreto `NVIDIA_API_KEY` (panel *Secrets*) y dale acceso al notebook.
3. Ejecuta las celdas en orden. El 02b descarga `data/company_documents.json` de este repo si no lo encuentra, y el 03 descarga los PDFs de arXiv (o, si falla, del repo del curso).

**En local (con salidas y capturas automáticas)**

```bash
pip install -r requirements.txt
```

```bash
playwright install chromium
```

```bash
python scripts/run_notebooks.py
```

`run_notebooks.py` exige `NVIDIA_API_KEY` en el entorno, ejecuta los tres notebooks con `nbclient`, guarda las salidas, toma las capturas de Gradio con Playwright ([scripts/capture_gradio.py](scripts/capture_gradio.py)) en `evidencias/` y revisa que ninguna salida contenga una API key.

> No hay API keys en ningún archivo de este repositorio: se leen de Colab Secrets o de la variable de entorno.

## Decisiones y aprendizajes

- **El `score_threshold` de `langchain-qdrant` no es un umbral de coseno.** Compara contra `(cos + 1) / 2`, así que 0.25 deja pasar cualquier chunk con coseno ≥ −0.5. Lo comprobé en el código y con vectores de coseno conocido: un chunk ortogonal a la pregunta pasa con relevancia 0.5. Para filtrar de verdad hay que calibrar en coseno y convertir.
- **`QdrantClient(":memory:")` y `path=...` no usan HNSW.** El modo local calcula la distancia contra todos los vectores con numpy. Eso está bien para un taller, pero el HNSW de la teoría solo existe en el servidor de Qdrant.
- **El ef necesario crece con n.** En C1, con M=16, recall ≥ 0.95 pidió ef=256 con 20 000 vectores y ef=512 con 100 000, y con 500 000 no se alcanzó ni con ef=1024. HNSW le gana a la búsqueda exacta con recall alto entre 20k y 100k vectores, y los datos aleatorios lo hacen más difícil (contraste de distancias de 1.78 frente a 9.03 con clústeres).
- **La calidad de la extracción importa tanto como el modelo.** MarkItDown pega palabras en cerca del 7 % de los tokens de los PDFs. Por eso las palabras clave de evaluación son tokens cortos, y aun así aparecen falsos positivos como «to**p100**0».
- **Las verificaciones automáticas también necesitan verificación.** Mi primera comprobación de las preguntas trampa decía que «TPU» estaba en los papers porque `"tpu"` está dentro de `"output"`.
- **Los problemas de red se resuelven con ingeniería, no con suerte:** timeout corto, reintentos con backoff y caché en los embeddings, y `enable_thinking=False` para que Nemotron 3.5 no gaste el presupuesto de tokens razonando.
