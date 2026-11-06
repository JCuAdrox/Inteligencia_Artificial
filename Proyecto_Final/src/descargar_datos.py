# Descarga el precio de bolsa nacional horario desde la API publica de XM
# (https://servapibi.xm.com.co) y lo guarda en data/precio_bolsa.csv
# La API solo deja pedir ~30 dias por consulta, por eso se pide mes a mes.
import requests
import pandas as pd

url = "https://servapibi.xm.com.co/hourly"
filas = []
meses = pd.date_range("2023-01-01", "2026-10-01", freq="MS")

for ini in meses:
    fin = ini + pd.offsets.MonthEnd(0)
    body = {"MetricId": "PrecBolsNaci", "StartDate": str(ini.date()),
            "EndDate": str(fin.date()), "Entity": "Sistema"}
    r = requests.post(url, json=body, timeout=60)
    datos = r.json()["Items"]
    for d in datos:
        vals = d["HourlyEntities"][0]["Values"]
        for h in range(1, 25):
            filas.append([d["Date"], h - 1, float(vals["Hour%02d" % h])])
    print(ini.date(), len(datos), "dias")

df = pd.DataFrame(filas, columns=["fecha", "hora", "precio"])
df.to_csv("data/precio_bolsa.csv", index=False)
print(df.shape)
