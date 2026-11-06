# Descarga el texto de las leyes del sector electrico desde el Gestor Normativo
# de Funcion Publica y las guarda como .txt en data/docs (corpus del RAG)
import re
import requests
import truststore
from bs4 import BeautifulSoup

# el servidor de Funcion Publica no manda el certificado intermedio completo,
# con truststore se usa el almacen de certificados del sistema operativo
truststore.inject_into_ssl()

normas = {
    "ley_143_1994": 4631,    # ley electrica: mercado mayorista, bolsa, CREG
    "ley_1715_2014": 57353,  # fuentes no convencionales, autogeneracion
    "ley_2099_2021": 166326, # transicion energetica, almacenamiento
}

for nombre, i in normas.items():
    url = "https://www.funcionpublica.gov.co/eva/gestornormativo/norma.php?i=%d" % i
    html = requests.get(url, timeout=120).text
    sopa = BeautifulSoup(html, "html.parser")
    # el texto de la norma esta en este div, cada parrafo es un <p>
    cont = sopa.find("div", class_="descripcion-contenido")
    parrafos = [p.get_text(" ", strip=True) for p in cont.find_all("p")]
    texto = "\n".join(p for p in parrafos if p)
    texto = re.sub(r"[ \t]+", " ", texto)
    with open("data/docs/%s.txt" % nombre, "w", encoding="utf-8") as f:
        f.write("FUENTE: " + url + "\n\n" + texto)
    print(nombre, len(texto))

# documentos de la UPME sobre sistemas de almacenamiento con baterias (SAEB), en PDF
import io
from pypdf import PdfReader

pdfs = {
    "upme_circular_030_2025": "https://docs.upme.gov.co/Normatividad/Circular_030_2025.pdf",
    "upme_ain_saeb_2025": "https://docs.upme.gov.co/Normatividad/AIN_Final_Circular_030_2025.pdf",
    "upme_str_01_2021_respuestas": "https://docs.upme.gov.co/PromocionSector/InformacionInversionistas/Documents/UPME-STR-01-2021/Avance_2_RespuestaObservacionesInversionista.pdf",
}

for nombre, url in pdfs.items():
    r = requests.get(url, timeout=180)
    lector = PdfReader(io.BytesIO(r.content))
    texto = "\n".join(p.extract_text() or "" for p in lector.pages)
    texto = re.sub(r"[ \t]+", " ", texto)
    with open("data/docs/%s.txt" % nombre, "w", encoding="utf-8") as f:
        f.write("FUENTE: " + url + "\n\n" + texto)
    print(nombre, len(lector.pages), "paginas", len(texto))

# resoluciones de la CREG sobre baterias y autogeneracion
# OJO: el servidor de la CREG no responde por IPv6 y requests se queda colgado,
# por eso se fuerza la conexion por IPv4
import urllib3.util.connection
urllib3.util.connection.HAS_IPV6 = False

resoluciones = {
    "creg_098_2019": "resolucion_creg_0098_2019",      # SAEB (derogada en 2026)
    "creg_101_113_2026": "resolucion_creg_101-113_2026",  # SAEB, la vigente
    "creg_174_2021": "resolucion_creg_0174_2021",      # autogeneracion a pequeña escala
}

for nombre, pagina in resoluciones.items():
    url = "https://gestornormativo.creg.gov.co/gestor/entorno/docs/%s.htm" % pagina
    r = requests.get(url, timeout=120)
    sopa = BeautifulSoup(r.content, "html.parser")
    parrafos = [p.get_text(" ", strip=True) for p in sopa.find_all("p")]
    # quitar el pie de pagina del sitio
    parrafos = [p for p in parrafos if p and "@creg.gov.co" not in p and "Consulte el documento" not in p]
    texto = re.sub(r"[ \t]+", " ", "\n".join(parrafos))
    with open("data/docs/%s.txt" % nombre, "w", encoding="utf-8") as f:
        f.write("FUENTE: " + url + "\n\n" + texto)
    print(nombre, len(texto))
