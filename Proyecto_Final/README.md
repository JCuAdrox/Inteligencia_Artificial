# Proyecto final IA - Batería en la bolsa de energía de XM con MDP + agente RAG

**SI3003 Inteligencia Artificial - Universidad EAFIT - 2026-2**

Integrantes:
- Jerónimo Castaño Herrera
- Emanuel Gonzalez Florez
- Jerónimo Zapata Cuadros

## De qué se trata

Escogimos la idea de *despacho de energía con RL y asistente sobre regulación*. Una batería pequeña (por defecto 1 MWh) compra energía en la bolsa cuando el precio está bajo y la vende cuando está alto. Con los precios reales horarios de XM:

- **Parte 1 (técnica clásica):** modelamos la operación como un **MDP** y lo resolvemos con **iteración de valor** implementada por nosotros (`src/mdp.py`, sin librerías de RL). Lo comparamos contra una política ingenua (horario fijo), una regla adaptativa (perfil de los últimos 30 días) y un oráculo con información perfecta (cota superior).
- **Parte 2 (IA moderna):** un **agente LLM** con loop manual de tool calling (`src/agente.py`) que tiene 3 herramientas: `consultar_precios`, `simular_bateria` y `buscar_normativa` (RAG con FAISS sobre leyes del sector eléctrico, resoluciones de la CREG y documentos de la UPME sobre baterías).
- **Dashboard** en Streamlit (`app.py`) con parámetros de la batería, visualizaciones y chat con el agente.

Resultados principales (prueba = enero a octubre de 2026, datos que el modelo no vio):

| política | ganancia (millones COP) | % del oráculo |
|---|---|---|
| MDP (iteración de valor) | 41.6 | 74 % |
| ingenua (carga 1-4 a.m., descarga desde 6 p.m.) | 37.0 | 65 % |
| perfil últimos 30 días | 47.7 | 84 % |
| oráculo (conoce el futuro) | 56.5 | 100 % |

El MDP le gana a la regla ingenua (+12,5 %, 9 de 10 meses) pero no a la regla adaptativa, ver el análisis en el notebook 01 y en el informe. El agente llamó la herramienta correcta en 14 de 15 preguntas de prueba y citó la fuente en 6 de 6 preguntas de normativa (notebook 02).

## Estructura

```
Proyecto_Final/
├── app.py                        dashboard (streamlit)
├── src/
│   ├── descargar_datos.py        descarga precios de XM -> data/precio_bolsa.csv
│   ├── descargar_docs.py         descarga leyes, resoluciones CREG y docs UPME -> data/docs/
│   ├── mdp.py                    MDP, iteracion de valor y simulacion de politicas
│   ├── rag.py                    chunks + embeddings + indice FAISS
│   └── agente.py                 loop del agente con tool calling
├── notebooks/
│   ├── 01_datos_y_mdp.ipynb      EDA, entrenamiento, validacion y resultados del MDP
│   └── 02_evaluacion_agente.ipynb  evaluacion del agente con 15 preguntas fijas
├── data/
│   ├── precio_bolsa.csv          precio de bolsa horario 2023-01-01 a 2026-10-31
│   ├── docs/                     corpus del RAG en texto
│   ├── indice.faiss, chunks.json indice del RAG (se regenera si se borra)
│   └── evaluacion_agente.csv     resultados de la evaluacion del agente
├── informe.md                    informe (max 4 paginas)
├── declaracion_uso_IA.md         declaracion de uso de IA generativa
└── requirements.txt
```

## Cómo correrlo

Probado con Python 3.11 en Windows. Todos los comandos se corren desde la carpeta `Proyecto_Final`:

```bash
git clone https://github.com/JCuAdrox/Inteligencia_Artificial.git
```

```bash
cd Inteligencia_Artificial/Proyecto_Final
```

1. Crear un entorno e instalar dependencias

```bash
conda create -n proyecto_ia python=3.11 -y
```

```bash
conda activate proyecto_ia
```

```bash
pip install -r requirements.txt
```

2. Poner la llave del LLM. Copiar `.env.example` como `.env` y llenar `GROQ_API_KEY` (es gratis en https://console.groq.com). Usamos el modelo `openai/gpt-oss-20b` de Groq porque soporta tool calling. Sin la llave el dashboard funciona igual, solo el chat muestra un error.

```bash
cp .env.example .env
```

3. Correr el dashboard

```bash
streamlit run app.py
```

Se abre en http://localhost:8501. La primera vez se demora un poco porque descarga el modelo de embeddings (`intfloat/multilingual-e5-small`, ~470 MB) y entrena el MDP (unos 3 segundos).

4. (Opcional) Los notebooks se corren con Jupyter desde la carpeta `notebooks`. Ya están ejecutados con sus salidas.

### Datos

Los datos ya vienen en `data/`, pero se pueden volver a descargar:

```bash
python src/descargar_datos.py
```

```bash
python src/descargar_docs.py
```

Si se cambian los documentos hay que borrar `data/indice.faiss` y `data/chunks.json` para que el índice se reconstruya.

## Fuentes de datos

- **Precios:** XM S.A. E.S.P., métrica `PrecBolsNaci` (Precio de Bolsa Nacional, por hora, entidad Sistema), API pública https://servapibi.xm.com.co/hourly. Consultada el 5 de noviembre de 2026. Son datos agregados del mercado, no hay datos personales.
- **Corpus RAG:**
  - Ley 143 de 1994 (ley eléctrica), Ley 1715 de 2014 (fuentes no convencionales) y Ley 2099 de 2021 (transición energética), texto del Gestor Normativo de Función Pública: https://www.funcionpublica.gov.co/eva/gestornormativo/norma.php?i=4631 , ?i=57353 y ?i=166326
  - UPME, Circular Externa 030 de 2025 y su Análisis de Impacto Normativo sobre sistemas de almacenamiento con baterías (SAEB): https://docs.upme.gov.co/Normatividad/
  - UPME, respuestas a observaciones de la convocatoria UPME STR 01-2021 (primer SAEB del país, Atlántico)
  - CREG, Resolución 098 de 2019 (SAEB, derogada), Resolución 101 113 de 2026 (SAEB, vigente) y Resolución 174 de 2021 (autogeneración a pequeña escala), del Gestor Normativo de la CREG: https://gestornormativo.creg.gov.co/gestor/entorno/docs/
- Nota técnica: el servidor de la CREG no responde por IPv6, y en algunas redes `requests` se queda esperando. Por eso `descargar_docs.py` fuerza la conexión por IPv4. Los textos ya vienen descargados en `data/docs/`.

## Limitaciones (resumen)

- La batería se supone tomadora de precios (no mueve el precio de bolsa), sin degradación, sin cargos de red ni impuestos, y sin la regulación real de cómo se remunera un SAEB en Colombia.
- El MDP solo ve la hora, el nivel de carga y un rango del precio de la hora anterior; los datos no son estacionarios (el patrón de horas baratas cambió con la entrada de la solar), y por eso una regla adaptativa le gana.
- El RAG usa un modelo de embeddings pequeño y a veces mezcla fragmentos de documentos parecidos; el agente puede citar con imprecisión, hay que revisar la fuente.
- **Esto es un ejercicio académico. Nada de lo que dice la aplicación o el agente es asesoría financiera, legal ni de inversión.**

## Uso de IA generativa

Ver `declaracion_uso_IA.md`.
