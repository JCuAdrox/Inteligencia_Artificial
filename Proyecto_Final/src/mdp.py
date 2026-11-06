# MDP de una bateria que compra y vende energia en la bolsa de XM.
# Estado: (hora del dia, nivel de carga, rango del precio relativo de la hora anterior)
# precio relativo = precio / promedio de las 24 horas anteriores (asi no depende del
# nivel general de precios del año, que cambia mucho p.ej. con El Niño en 2024)
# Acciones: 0 = cargar, 1 = esperar, 2 = descargar
# Se resuelve con iteracion de valor (implementada a mano, sin librerias de RL)
import numpy as np
import pandas as pd

ACCIONES = ["cargar", "esperar", "descargar"]
K = 3  # numero de rangos de precio (se escogio con validacion, ver notebook 01)


def cargar_precios(ruta="data/precio_bolsa.csv"):
    df = pd.read_csv(ruta, parse_dates=["fecha"])
    df = df.sort_values(["fecha", "hora"]).reset_index(drop=True)
    media24 = df["precio"].shift(1).rolling(24, min_periods=1).mean()
    df["relativo"] = (df["precio"] / media24).fillna(1.0)
    return df


def entrenar(train, capacidad=1.0, potencia=0.25, eficiencia=0.85, gamma=0.999):
    """capacidad en MWh, potencia en MWh por hora. Devuelve un dict con la politica."""
    niveles = int(round(capacidad / potencia)) + 1
    precios = train["relativo"].values
    horas = train["hora"].values

    # rangos del precio relativo por cuantiles del entrenamiento
    cortes = np.quantile(precios, np.linspace(0, 1, K + 1)[1:-1])
    rango = np.digitize(precios, cortes)

    # precio relativo esperado de la hora h dado el rango de la hora anterior: E[p_h | k_{h-1}]
    # y matriz de transicion del rango: T[h, k_anterior, k_actual]
    suma = np.zeros((24, K))
    cuenta = np.zeros((24, K))
    T = np.ones((24, K, K))  # se arranca en 1 para que no haya filas en cero
    for t in range(1, len(precios)):
        h, k_ant, k = horas[t], rango[t - 1], rango[t]
        suma[h, k_ant] += precios[t]
        cuenta[h, k_ant] += 1
        T[h, k_ant, k] += 1
    precio_esp = suma / np.maximum(cuenta, 1)
    # si alguna combinacion no aparece se usa el promedio de esa hora
    prom_hora = train.groupby("hora")["precio"].mean().values
    prom_rel = train.groupby("hora")["relativo"].mean().values
    for h in range(24):
        for k in range(K):
            if cuenta[h, k] == 0:
                precio_esp[h, k] = prom_rel[h]
    T = T / T.sum(axis=2, keepdims=True)

    # recompensa en "pesos a precio promedio": relativo * 1000 * MWh
    E = potencia * 1000
    V = np.zeros((24, niveles, K))
    Q = np.zeros((24, niveles, K, 3))
    for it in range(20000):
        V_nuevo = np.zeros_like(V)
        for h in range(24):
            h2 = (h + 1) % 24
            # valor esperado del siguiente estado: EV[s2, k] = sum_k' T[h,k,k'] V[h2,s2,k']
            # (el "precio anterior" del siguiente estado es el de esta hora)
            EV = V[h2] @ T[h].T
            p = precio_esp[h]  # vector por rango k
            Q[h, :, :, 1] = gamma * EV                          # esperar
            Q[h, :-1, :, 0] = -p * E + gamma * EV[1:]           # cargar: s -> s+1
            Q[h, -1, :, 0] = -1e18                              # llena, no puede cargar
            Q[h, 1:, :, 2] = p * E * eficiencia + gamma * EV[:-1]  # descargar: s -> s-1
            Q[h, 0, :, 2] = -1e18                               # vacia, no puede descargar
            V_nuevo[h] = Q[h].max(axis=2)
        dif = V_nuevo - V
        V = V_nuevo
        # con gamma casi 1 se para cuando el cambio es casi igual en todos los estados
        # (span pequeño), en ese punto la politica ya no cambia
        if dif.max() - dif.min() < 0.01:
            break

    return {"politica": Q.argmax(axis=3), "V": V, "cortes": cortes, "niveles": niveles,
            "potencia": potencia, "eficiencia": eficiencia, "iteraciones": it + 1,
            "precio_esp": precio_esp, "prom_hora": prom_hora}


def simular(test, modelo, politica="mdp"):
    """Simula la bateria hora a hora sobre los precios reales de test.
    politica: 'mdp', 'ingenua' (horario fijo), 'perfil' (horas mas baratas/caras de los
    ultimos 30 dias) u 'oraculo' (conoce todos los precios de antemano, cota superior)."""
    niveles = modelo["niveles"]
    E = modelo["potencia"] * 1000
    ef = modelo["eficiencia"]
    precios = test["precio"].values
    horas = test["hora"].values
    n = niveles - 1  # horas que toma llenar la bateria

    # horas para las reglas simples
    if politica == "ingenua":
        h_cargar = list(range(1, 1 + n))     # madrugada
        h_descargar = list(range(18, 18 + n))  # pico de la noche

    # el oraculo resuelve todo el periodo hacia atras conociendo los precios
    if politica == "oraculo":
        N = len(precios)
        W = np.zeros((N + 1, niveles))
        mejor = np.zeros((N, niveles), dtype=int)
        for t in range(N - 1, -1, -1):
            for s in range(niveles):
                opciones = [-1e18, W[t + 1, s], -1e18]
                if s < niveles - 1:
                    opciones[0] = -precios[t] * E + W[t + 1, s + 1]
                if s > 0:
                    opciones[2] = precios[t] * E * ef + W[t + 1, s - 1]
                mejor[t, s] = int(np.argmax(opciones))
                W[t, s] = max(opciones)

    relativos = test["relativo"].values
    s = 0
    filas = []
    for t in range(len(precios)):
        h = horas[t]
        if politica == "mdp":
            # solo se conoce el precio de la hora anterior
            rel_prev = relativos[t - 1] if t > 0 else 1.0
            k = int(np.digitize(rel_prev, modelo["cortes"]))
            a = modelo["politica"][h, s, k]
        elif politica == "oraculo":
            a = mejor[t, s]
        else:
            # perfil: cada medianoche se recalculan las horas con el promedio de los
            # ultimos 30 dias (al inicio, si hay menos de una semana, el del entrenamiento)
            if politica == "perfil" and h == 0:
                if t >= 24 * 7:
                    prom = pd.Series(precios[max(0, t - 720):t]).groupby(horas[max(0, t - 720):t]).mean()
                    orden = list(prom.sort_values().index)
                else:
                    orden = list(np.argsort(modelo["prom_hora"]))
                h_cargar = orden[:n]
                h_descargar = orden[-n:]
            a = 1
            if h in h_cargar and s < niveles - 1:
                a = 0
            if h in h_descargar and s > 0:
                a = 2

        if a == 0:
            ganancia = -precios[t] * E
            s += 1
        elif a == 2:
            ganancia = precios[t] * E * ef
            s -= 1
        else:
            ganancia = 0
        filas.append([h, precios[t], ACCIONES[a], s, ganancia])

    res = pd.DataFrame(filas, columns=["hora", "precio", "accion", "nivel", "ganancia"])
    res.insert(0, "fecha", test["fecha"].values)
    res["ganancia_acum"] = res["ganancia"].cumsum()
    return res
