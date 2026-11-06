# Dashboard: bateria en la bolsa de energia de XM + chat con el agente
# correr con:  streamlit run app.py
import datetime as dt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import mdp

st.set_page_config(page_title="Bateria en bolsa - XM", layout="wide")
st.title("Despacho de una batería con el precio de bolsa (XM)")
st.caption("Proyecto final SI3003 - Jerónimo Castaño Herrera, Emanuel Gonzalez Florez, Jerónimo Zapata Cuadros. "
           "Ejercicio académico, no es asesoría financiera ni legal.")


@st.cache_data
def datos():
    return mdp.cargar_precios()


@st.cache_resource
def modelo(capacidad, potencia, eficiencia):
    train = datos()
    train = train[train["fecha"] < "2026-01-01"]
    return mdp.entrenar(train, capacidad, potencia, eficiencia)


precios = datos()

# ---- parametros del usuario ----
st.sidebar.header("Parámetros")
f_ini = st.sidebar.date_input("Fecha inicio", dt.date(2026, 9, 1),
                              min_value=dt.date(2023, 1, 1), max_value=dt.date(2026, 10, 31))
f_fin = st.sidebar.date_input("Fecha fin", dt.date(2026, 9, 30),
                              min_value=dt.date(2023, 1, 1), max_value=dt.date(2026, 10, 31))
capacidad = st.sidebar.slider("Capacidad (MWh)", 0.5, 4.0, 1.0, 0.5)
horas_llenado = st.sidebar.slider("Horas para llenarla", 1, 8, 4)
eficiencia = st.sidebar.slider("Eficiencia ida y vuelta", 0.6, 1.0, 0.85, 0.01)
politica = st.sidebar.selectbox("Política a mostrar", ["mdp", "ingenua", "perfil", "oraculo"])
st.sidebar.markdown("""
**Políticas**
- *mdp*: iteración de valor (entrenada con 2023-2025)
- *ingenua*: carga 1-4 a.m., descarga desde las 6 p.m.
- *perfil*: horas más baratas/caras de los últimos 30 días
- *oraculo*: conoce los precios futuros (cota superior)
""")
if f_ini < dt.date(2026, 1, 1):
    st.sidebar.warning("Ojo: antes de 2026 son datos de entrenamiento del MDP, el resultado es optimista.")

potencia = capacidad / horas_llenado
m = modelo(capacidad, potencia, eficiencia)
d = precios[(precios["fecha"] >= pd.Timestamp(f_ini)) & (precios["fecha"] <= pd.Timestamp(f_fin))]

tab1, tab2, tab3 = st.tabs(["Simulación", "Datos históricos", "Chat con el agente"])

with tab1:
    if len(d) == 0:
        st.error("No hay datos en ese rango de fechas.")
    else:
        sims = {p: mdp.simular(d, m, p) for p in ["mdp", "ingenua", "perfil", "oraculo"]}
        r = sims[politica]
        r["momento"] = r["fecha"] + pd.to_timedelta(r["hora"], unit="h")

        c = st.columns(4)
        for i, p in enumerate(sims):
            g = sims[p]["ganancia"].sum()
            pct = 100 * g / sims["oraculo"]["ganancia"].sum()
            c[i].metric(p, "$ {:,.0f}".format(g), "{:.0f}% del oráculo".format(pct), delta_color="off")

        # precio vs decisiones
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=r["momento"], y=r["precio"], mode="lines", name="precio",
                                 line=dict(color="gray", width=1)))
        for acc, color in [("cargar", "green"), ("descargar", "red")]:
            x = r[r["accion"] == acc]
            fig.add_trace(go.Scatter(x=x["momento"], y=x["precio"], mode="markers", name=acc,
                                     marker=dict(color=color, size=6)))
        fig.update_layout(title="Precio de bolsa y decisiones de la política '%s'" % politica,
                          yaxis_title="COP/kWh", height=420)
        st.plotly_chart(fig, use_container_width=True)

        # ganancia acumulada de todas las politicas
        acum = pd.DataFrame({p: sims[p]["ganancia_acum"].values for p in sims})
        acum["momento"] = r["momento"].values
        fig2 = px.line(acum, x="momento", y=list(sims), title="Ganancia acumulada (COP)",
                       labels={"value": "COP", "variable": "política"})
        st.plotly_chart(fig2, use_container_width=True)

        col1, col2 = st.columns(2)
        # nivel de carga de la bateria
        fig3 = px.area(r, x="momento", y="nivel", title="Nivel de carga (pasos de %.2f MWh)" % potencia)
        col1.plotly_chart(fig3, use_container_width=True)
        # politica aprendida por el MDP
        pol = pd.DataFrame(m["politica"][:, :, 1], columns=["nivel %d" % s for s in range(m["niveles"])])
        fig4 = px.imshow(pol.T, color_continuous_scale=["green", "lightgray", "red"], zmin=0, zmax=2,
                         labels=dict(x="hora", y="", color="acción"),
                         title="Política MDP con precio 'normal' (verde=cargar, rojo=descargar)")
        col2.plotly_chart(fig4, use_container_width=True)

with tab2:
    p2 = precios.copy()
    p2["mes"] = p2["fecha"].dt.to_period("M").astype(str)
    diario = p2.groupby("fecha")["precio"].mean().reset_index()
    st.plotly_chart(px.line(diario, x="fecha", y="precio", title="Precio de bolsa promedio diario (COP/kWh)"),
                    use_container_width=True)
    tabla = p2.pivot_table(index="hora", columns="mes", values="precio", aggfunc="mean")
    st.plotly_chart(px.imshow(tabla, aspect="auto", color_continuous_scale="Viridis",
                              title="Precio promedio por hora del día y mes"), use_container_width=True)
    st.caption("Fuente: XM, métrica PrecBolsNaci, API pública servapibi.xm.com.co")

with tab3:
    try:
        from src import agente
    except Exception as e:
        st.error("No se pudo cargar el agente (revisar el archivo .env y la conexión): %s" % e)
        st.stop()

    if "historial" not in st.session_state:
        st.session_state.historial = []
        st.session_state.herramientas = []

    for msg in st.session_state.historial:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    pregunta = st.chat_input("Pregúntale algo al agente (precios, simulaciones, normativa)...")
    if pregunta:
        with st.chat_message("user"):
            st.markdown(pregunta)
        with st.chat_message("assistant"):
            with st.spinner("pensando..."):
                try:
                    resp, llamadas = agente.preguntar(pregunta, st.session_state.historial[-6:])
                except Exception as e:
                    resp, llamadas = "Error llamando al modelo: %s" % e, []
            st.markdown(resp)
            if llamadas:
                with st.expander("Herramientas usadas"):
                    st.json(llamadas)
        st.session_state.historial.append({"role": "user", "content": pregunta})
        st.session_state.historial.append({"role": "assistant", "content": resp})
