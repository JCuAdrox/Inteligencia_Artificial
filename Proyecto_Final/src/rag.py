# RAG sobre las leyes del sector electrico (data/docs).
# Se parte cada ley por articulos, se calculan embeddings con sentence-transformers
# y se indexa con FAISS. Si el indice ya existe se carga del disco.
import os
import re
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

MODELO_EMB = "intfloat/multilingual-e5-small"  # e5 pide los prefijos "query: " y "passage: "
RUTA_INDICE = "data/indice.faiss"
RUTA_CHUNKS = "data/chunks.json"

modelo = SentenceTransformer(MODELO_EMB)

if os.path.exists(RUTA_INDICE):
    indice = faiss.read_index(RUTA_INDICE)
    chunks = json.load(open(RUTA_CHUNKS, encoding="utf-8"))
else:
    chunks = []
    for archivo in sorted(os.listdir("data/docs")):
        texto = open("data/docs/" + archivo, encoding="utf-8").read()
        doc = archivo.replace(".txt", "")
        # las leyes y resoluciones se cortan donde empieza cada "ARTÍCULO 12." / "Artículo 12o."
        # los PDF de la UPME se dejan enteros y se parten por tamaño
        if doc.startswith(("ley", "creg")):
            partes = re.split(r"\n(?=\s*ART[IÍ]CULO\s+\d+)", texto, flags=re.IGNORECASE)
        else:
            partes = [texto]
        for p in partes:
            p = p.strip()
            if len(p) < 80:
                continue
            m = re.match(r"ART[IÍ]CULO\s+(\d+)", p, flags=re.IGNORECASE) if doc.startswith(("ley", "creg")) else None
            # los PDF de la UPME no tienen articulos, ahi se usa el numero de fragmento
            # pedazos de ~1500 caracteres con un poco de traslape
            for j in range(0, len(p), 1500):
                art = "Artículo " + m.group(1) if m else "fragmento %d" % (j // 1500 + 1)
                chunks.append({"doc": doc, "articulo": art, "texto": p[j:j + 1600]})
    emb = modelo.encode(["passage: " + c["texto"] for c in chunks], normalize_embeddings=True)
    indice = faiss.IndexFlatIP(emb.shape[1])
    indice.add(np.array(emb, dtype="float32"))
    faiss.write_index(indice, RUTA_INDICE)
    json.dump(chunks, open(RUTA_CHUNKS, "w", encoding="utf-8"), ensure_ascii=False)


def buscar(pregunta, k=4):
    q = modelo.encode(["query: " + pregunta], normalize_embeddings=True)
    scores, ids = indice.search(np.array(q, dtype="float32"), k)
    res = []
    for s, i in zip(scores[0], ids[0]):
        c = chunks[i]
        res.append({"fuente": c["doc"] + ", " + c["articulo"], "score": round(float(s), 3),
                    "texto": c["texto"]})
    return res
