
from pathlib import Path
import sys
import importlib.util
import pickle
import io
import tempfile
from datetime import datetime

import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image, ImageFilter

import matplotlib.pyplot as plt
import matplotlib.dates as mdates


import zipfile
import pandas as pd
import io

import re

import json
def construir_dataset_modelo_cestas(df_cestas):

    if df_cestas is None or df_cestas.empty:
        return pd.DataFrame()

    df = df_cestas.copy()

    # Target
    df["Velocidad experimental"] = df["Velocidad"]

    # 🔥 One-hot encoding de especies
    especies_split = df["Especies"].str.split(", ")

    especies_unicas = sorted(set(
        e for sub in especies_split for e in sub
    ))

    for esp in especies_unicas:
        df[f"ESP_{esp}"] = especies_split.apply(
            lambda x: 1 if esp in x else 0
        )

    return df
def analisis_desviacion_por_cesta(
    df_cestas,
    detalle_crudos
):

    resultados = []

    for _, cesta in df_cestas.iterrows():

        fi = pd.to_datetime(cesta["Fecha_ini"])
        ff = pd.to_datetime(cesta["Fecha_fin"])

        estado = cesta.get("Estado", "NA")
        especies = cesta["Especies"]

        sub = detalle_crudos[
            (detalle_crudos["Fecha"] >= fi) &
            (detalle_crudos["Fecha"] <= ff)
        ]

        if sub.empty:
            continue

        suma = (
            sub.groupby("Especie")["Porcentaje"]
            .sum()
        )

        total = suma.sum()

        if total == 0:
            continue

        pct = (suma / total * 100)

        for crudo, val in pct.items():

            resultados.append({
                "Cesta": ", ".join(especies),
                "Estado": estado,
                "Crudo": crudo,
                "% promedio": val
            })

    df = pd.DataFrame(resultados)

    if df.empty:
        return df

    # 🔥 AGRUPACIÓN FINAL
    df_resumen = (
        df.groupby(["Cesta", "Crudo", "Estado"])
        .agg(
            veces=("Crudo", "count"),
            pct_medio=("% promedio", "mean")
        )
        .reset_index()
    )

    # pivot → ver claro ENCIMA / DEBAJO
    df_pivot = df_resumen.pivot_table(
        index=["Cesta", "Crudo"],
        columns="Estado",
        values="veces",
        fill_value=0
    ).reset_index()

    return df_pivot
def exportar_configuracion_json():

    config = {
        "parametros_globales": {
            "umbral": st.session_state.get("umbral"),
            "umbral_factor": st.session_state.get("umbral_factor"),
            "min_dias_seg": st.session_state.get("min_dias_seg")
        },
        "sondas": []
    }

    for key, data in st.session_state.get("processed_sheets", {}).items():

        sonda_data = {
            "source_name": data["source_name"],
            "hoja": data["hoja"],
            "segmentos_validos": [],
            "descartados": data.get("descartados", []),
            "saved": data.get("saved", False),
            "manually_modified": data.get("manually_modified", False)
        }

        for seg in data.get("segmentos_validos", []):

            seg_json = seg.copy()

            # Convertir fechas
            for f in ["fecha_ini", "fecha_fin"]:
                if isinstance(seg_json.get(f), pd.Timestamp):
                    seg_json[f] = seg_json[f].isoformat()

            # Convertir medias
            medias = seg_json.get("medias")
            if isinstance(medias, pd.Series):
                seg_json["medias"] = medias.to_dict()

            sonda_data["segmentos_validos"].append(seg_json)

        config["sondas"].append(sonda_data)

    config_safe = json_safe(config)

    return json.dumps(config_safe, indent=4)
from sklearn.linear_model import LinearRegression

def dividir_todos_segmentos(
        df_filtrado,
        segmentos,
        df_proc,
        vars_proceso,
        offset,
        min_dias=5
):

    nuevos = []

    for seg in segmentos:

        partes = dividir_segmento_por_intervalo(
            df_filtrado,
            seg,
            df_proc,
            vars_proceso,
            offset,
            min_dias=min_dias
        )

        if partes:
            nuevos.extend(partes)
        else:
            nuevos.append(seg)

    return sorted(nuevos, key=lambda x: x["fecha_ini"])

import plotly.graph_objects as go
def entrenar_modelos_ml(df, vars_proceso):

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score

    resultados = {}

    df = df.copy()

    # variables válidas
    vars_validas = [v for v in vars_proceso if v in df.columns]

    if not vars_validas:
        return {}

    X = df[vars_validas].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(df["Velocidad experimental"], errors="coerce")
    
    mask = (~X.isna().any(axis=1)) & (~y.isna())
    X = X[mask]
    y = y[mask]

    if len(X) < 10:
        return {}

    # =========================
    # RANDOM FOREST
    # =========================
    try:
        rf = RandomForestRegressor(n_estimators=300, random_state=42)
        rf.fit(X, y)
        pred_rf = rf.predict(X)
        resultados["Random Forest"] = {
            "modelo": rf,
            "pred": pred_rf,
            "r2": r2_score(y, pred_rf),
            "importancias": dict(zip(X.columns, rf.feature_importances_))
        }
    except:
        pass

    return resultados, y

def clasificar_por_tolerancia(y_real, y_pred, tol):

    df = pd.DataFrame({
        "real": y_real,
        "pred": y_pred
    }).dropna()

    df["delta"] = df["real"] - df["pred"]

    def clasificar(x):
        if x > tol:
            return "ENCIMA"
        elif x < -tol:
            return "DEBAJO"
        else:
            return "DENTRO"

    df["estado"] = df["delta"].apply(clasificar)

    return df["estado"].values

def grafica_modelo_vs_real(y_real, y_pred, titulo, tolerancia):

    import plotly.graph_objects as go

    df = pd.DataFrame({
        "real": y_real,
        "pred": y_pred
    }).dropna()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["real"],
        y=df["pred"],
        mode="markers"
    ))

    max_val = max(df["real"].max(), df["pred"].max())

    # diagonal
    fig.add_trace(go.Scatter(
        x=[0, max_val],
        y=[0, max_val],
        mode="lines",
        line=dict(color="red"),
        name="y=x"
    ))

    # tolerancia
    fig.add_trace(go.Scatter(
        x=[0, max_val],
        y=[tolerancia, max_val + tolerancia],
        mode="lines",
        line=dict(dash="dash"),
        name="tolerancia +"
    ))

    fig.add_trace(go.Scatter(
        x=[0, max_val],
        y=[-tolerancia, max_val - tolerancia],
        mode="lines",
        line=dict(dash="dash"),
        name="tolerancia -"
    ))

    fig.update_layout(
        title=titulo,
        xaxis_title="Real",
        yaxis_title="Predicción"
    )

    return fig
def importancia_por_subset(df, vars_proceso, target):

    resultados = []

    for var in vars_proceso:

        if var not in df.columns:
            continue

        sub = df[[var, target]].dropna()

        if len(sub) < 3:
            continue

        x = sub[var]
        y = sub[target]

        if x.std() == 0:
            continue

        corr = np.corrcoef(x, y)[0,1]

        resultados.append({
            "Variable": var,
            "Importancia": abs(corr)
        })

    if not resultados:
        return pd.DataFrame()

    return pd.DataFrame(resultados).sort_values(
        "Importancia",
        ascending=False
    )
def importancia_mpa(df):

    vars_mpa = []

    if "T" in df.columns:
        vars_mpa.append("T")

    if "TAN" in df.columns:
        vars_mpa.append("TAN")

    resultados = []

    for var in vars_mpa:

        sub = df[[var, "Velocidad esperada"]].dropna()

        if len(sub) < 3:
            continue

        x = sub[var]
        y = sub["Velocidad esperada"]

        if x.std() == 0:
            continue

        corr = np.corrcoef(x, y)[0,1]

        resultados.append({
            "Variable": var,
            "Importancia": abs(corr)
        })

    return pd.DataFrame(resultados)
def graficar_especie_vs_corrosion(df_result, especie):

    if df_result.empty:
        return None

    df_plot = df_result.dropna(subset=["% especie", "Velocidad"])

    if df_plot.empty:
        return None

    fig = go.Figure()

    # puntos
    fig.add_trace(go.Scatter(
        x=df_plot["% especie"],
        y=df_plot["Velocidad"],
        mode="markers",
        name=especie,
        marker=dict(size=10)
    ))

    # regresión
    x_reg, y_reg, r2 = calcular_regresion(
        df_plot["% especie"],
        df_plot["Velocidad"]
    )

    if x_reg is not None:

        fig.add_trace(go.Scatter(
            x=x_reg,
            y=y_reg,
            mode="lines",
            name="Tendencia"
        ))

        if r2 is not None:
            fig.add_annotation(
                x=0.05,
                y=0.95,
                xref="paper",
                yref="paper",
                text=f"R² = {r2:.3f}",
                showarrow=False
            )

    fig.update_layout(
        title=f"{especie}: % vs velocidad de corrosión",
        xaxis_title="% en la cesta",
        yaxis_title="Velocidad corrosión (mm/año)",
        height=500
    )

    return fig

def construir_cestas_crudo(detalle_crudos):

    if detalle_crudos.empty:
        return pd.DataFrame()

    # Agrupar por día → conjunto de especies
    df_day = (
        detalle_crudos
        .groupby("Fecha")["Especie"]
        .apply(lambda x: tuple(sorted(set(x))))
        .reset_index()
    )

    # Detectar cambios de cesta
    df_day["cambio"] = df_day["Especie"] != df_day["Especie"].shift()
    df_day["grupo"] = df_day["cambio"].cumsum()

    # Agrupar en cestas
    cestas = (
        df_day.groupby("grupo")
        .agg(
            Fecha_ini=("Fecha", "min"),
            Fecha_fin=("Fecha", "max"),
            Especies=("Especie", "first"),
            Dias=("Fecha", "count")
        )
        .reset_index(drop=True)
    )

    return cestas

def analizar_cestas(
    cestas,
    df_validos,
    df_proc
):

    resultados = []

    for i, cesta in cestas.iterrows():

        fi = pd.to_datetime(cesta["Fecha_ini"])
        ff = pd.to_datetime(cesta["Fecha_fin"])

        # 🔎 Buscar segmento donde cae
        seg_match = df_validos[
            (df_validos["Inicio"] <= ff) &
            (df_validos["Fin"] >= fi)
        ]

        if seg_match.empty:
            continue

        seg = seg_match.iloc[0]

        # 🔎 medias de proceso en ese intervalo
        if df_proc is not None and not df_proc.empty:

            sub_proc = df_proc[
                (df_proc["Fecha"] >= fi) &
                (df_proc["Fecha"] <= ff)
            ]

            medias = sub_proc.mean(numeric_only=True)

        else:
            medias = pd.Series()

        fila = {
            "Cesta_id": i + 1,
            "Fecha_ini": fi,
            "Fecha_fin": ff,
            "Dias": cesta["Dias"],
            "Especies": ", ".join(cesta["Especies"]),
            "Segmento": seg["Segmento"],
            "Estado": seg["estado_diag"],
            "Velocidad": seg["Velocidad experimental"]
        }

        # añadir variables proceso
        if isinstance(medias, pd.Series):
            for k, v in medias.items():
                fila[k] = v

        resultados.append(fila)

    if resultados:
        return pd.DataFrame(resultados)

    return pd.DataFrame()

def ranking_cestas(df_cestas):

    if df_cestas.empty:
        return pd.DataFrame()

    df_rank = (
        df_cestas.groupby("Especies")
        .agg(
            num_veces=("Cesta_id", "count"),
            dias_totales=("Dias", "sum"),
            vel_media=("Velocidad", "mean")
        )
        .sort_values("num_veces", ascending=False)
        .reset_index()
    )

    return df_rank


def calcular_calidad_segmento(df_filtrado, seg):

    try:

        i = int(seg["ini"])
        f = int(seg["fin"])

        sub = df_filtrado.iloc[i:f]

        if len(sub) < 3:
            return None

        fechas = pd.to_datetime(sub["Sent Time"])
        ut = sub["UT measurement (mm)"].values

        # convertir tiempo a días desde inicio
        t = (fechas - fechas.iloc[0]).dt.days.values.reshape(-1,1)

        model = LinearRegression().fit(t, ut)

        r2 = model.score(t, ut)

        return round(r2, 4)

    except:
        return None
def clasificar_calidad(r2):

    if r2 is None:
        return "Sin datos"

    if r2 >= 0.80:
        return "Excelente"
    elif r2 >= 0.65:
        return "Muy buena"
    elif r2 >= 0.50:
        return "Buena"
    elif r2 >= 0.30:
        return "Aceptable"
    else:
        return "Baja"

def color_calidad(val):
    colores = {
        "Excelente": "background-color: #4CAF50",
        "Muy buena": "background-color: #8BC34A",
        "Buena": "background-color: #CDDC39",
        "Aceptable": "background-color: #FFC107",
        "Baja": "background-color: #F44336"
    }
    return colores.get(val, "")
def analizar_importancia_proceso(df, vars_proceso):

    if df.empty or "Velocidad_corr" not in df.columns:
        return pd.DataFrame()

    resultados = []

    for var in vars_proceso:

        if var not in df.columns:
            continue

        sub = df[[var, "Velocidad_corr"]].dropna()

        if len(sub) < 3:
            continue

        x = sub[var]
        y = sub["Velocidad_corr"]

        if x.std() == 0:
            continue

        corr = np.corrcoef(x, y)[0,1]

        resultados.append({
            "Variable proceso": var,
            "Correlación con corrosión": corr,
            "Valor absoluto": abs(corr)
        })

    if not resultados:
        return pd.DataFrame()

    return pd.DataFrame(resultados).sort_values(
        "Valor absoluto",
        ascending=False
    )

def construir_tabla_segmentos_comparativa(processed_sheets, df_mpa, material):

    processed = {
        k: v for k, v in processed_sheets.items()
        if v.get("saved")
    }

    if not processed:
        return pd.DataFrame()

    primera = list(processed.values())[0]
    segmentos_base = primera["segmentos_validos"]

    filas = []

    for i, seg_base in enumerate(segmentos_base, start=1):

        fila = {
            "Segmento": f"Seg {i}",
            "Inicio": seg_base.get("fecha_ini"),
            "Fin": seg_base.get("fecha_fin")
        }

        fi_ref = pd.to_datetime(seg_base["fecha_ini"])
        ff_ref = pd.to_datetime(seg_base["fecha_fin"])

        velocidades = []

        # =========================================
        # COLUMNAS POR SONDA (Vel + Calidad)
        # =========================================

        for key, data in processed.items():

            nombre_sonda = f"{data['source_name']} | {data['hoja']}"

            vel = None
            calidad = None

            for seg in data["segmentos_validos"]:

                fi = pd.to_datetime(seg["fecha_ini"])
                ff = pd.to_datetime(seg["fecha_fin"])

                if fi == fi_ref and ff == ff_ref:

                    vel = seg.get("vel_abs")

                    calidad = calcular_calidad_segmento(
                        data["df_filtrado"],
                        seg
                    )
                    texto_calidad = clasificar_calidad(calidad)
                    break

            fila[f"{nombre_sonda} Velocidad"] = vel
            fila[f"{nombre_sonda} Calidad R2"] = calidad
            fila[f"{nombre_sonda} Calidad"] = texto_calidad
            fila["Días segmento"] = seg.get("delta_dias")
            if vel is not None:
                velocidades.append(vel)

        # =========================================
        # ESTADÍSTICAS ENTRE SONDAS
        # =========================================

        if velocidades:
            media = np.mean(velocidades)
            std = np.std(velocidades, ddof=1) if len(velocidades) > 1 else 0
            cv = (std / media) * 100 if media != 0 else None
        else:
            media, std, cv = None, None, None

        fila["Media velocidades"] = media
        fila["Desviación estándar"] = std
        fila["Coef Variación (%)"] = cv

        # =========================================
        # VELOCIDAD ESPERADA MPA
        # =========================================

        vel_esperada = None

        medias_proc = seg_base.get("medias")

        if df_mpa is not None and isinstance(medias_proc, (dict, pd.Series)):

            md = dict(medias_proc)

            temp = md.get("T")
            tan = md.get("TAN")

            vel_esperada = buscar_velocidad_mas_cercana(
                df_mpa,
                temp,
                tan,
                material
            )

        fila["Velocidad esperada"] = vel_esperada

        if vel_esperada is not None and media is not None:
            fila["Dif Real vs Esperada"] = media - vel_esperada
            fila["Dif absoluta"] = abs(fila["Dif Real vs Esperada"])

        # =========================================
        # VARIABLES PROCESO (una sola vez)
        # =========================================

        if isinstance(medias_proc, (dict, pd.Series)):
            for k, v in dict(medias_proc).items():
                fila[k] = v

        filas.append(fila)

    return pd.DataFrame(filas)

def json_safe(obj):

    # pandas Timestamp
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()

    # numpy numbers
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)

    if isinstance(obj, (np.floating, np.float64)):
        return float(obj)

    # pandas Series
    if isinstance(obj, pd.Series):
        return {k: json_safe(v) for k, v in obj.to_dict().items()}

    # numpy arrays
    if isinstance(obj, np.ndarray):
        return [json_safe(x) for x in obj.tolist()]

    # dict
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}

    # list
    if isinstance(obj, list):
        return [json_safe(x) for x in obj]

    return obj


def importar_configuracion_json(uploaded_json):

    config = json.load(uploaded_json)

    # Parámetros globales
    params = config.get("parametros_globales", {})

    st.session_state["umbral"] = params.get("umbral")
    st.session_state["umbral_factor"] = params.get("umbral_factor")
    st.session_state["min_dias_seg"] = params.get("min_dias_seg")

    # Sondas
    for sonda in config.get("sondas", []):

        key = f"proc|{sonda['source_name']}|{sonda['hoja']}"

        segmentos = []

        for seg in sonda.get("segmentos_validos", []):

            # Reconstruir fechas
            for f in ["fecha_ini","fecha_fin"]:
                if seg.get(f):
                    seg[f] = pd.to_datetime(seg[f])

            # Reconstruir medias
            if isinstance(seg.get("medias"), dict):
                seg["medias"] = pd.Series(seg["medias"])

            segmentos.append(seg)

        st.session_state["processed_sheets"][key] = {
            "source_name": sonda["source_name"],
            "hoja": sonda["hoja"],
            "segmentos_validos": segmentos,
            "descartados": sonda.get("descartados", []),
            "saved": sonda.get("saved", True),
            "manually_modified": sonda.get("manually_modified", True)
        }


def procesar_crudos(df):

    df = df.copy()

    # ======================================================
    # DETECTAR COLUMNA FECHA AUTOMÁTICAMENTE
    # ======================================================
    
    fecha_col = None
    
    # 1️⃣ Buscar nombres típicos
    for c in df.columns:
        cl = str(c).lower()
        if any(k in cl for k in ["fecha", "date", "time", "timestamp", "dia"]):
            fecha_col = c
            break
    
    # 2️⃣ Buscar columnas Unnamed
    if fecha_col is None:
        unnamed_cols = [c for c in df.columns if "unnamed" in str(c).lower()]
    
        for c in unnamed_cols:
            try:
                test = pd.to_datetime(df[c].dropna().iloc[:10], errors="coerce")
                if test.notna().sum() >= 3:
                    fecha_col = c
                    break
            except:
                pass
    
    # 3️⃣ Buscar cualquier columna convertible a fecha
    if fecha_col is None:
        for c in df.columns:
            try:
                test = pd.to_datetime(df[c].dropna().iloc[:10], errors="coerce")
                if test.notna().sum() >= 3:
                    fecha_col = c
                    break
            except:
                pass
    
    if fecha_col is None:
        raise ValueError("No se pudo detectar columna de fechas")
    
    df["Fecha"] = pd.to_datetime(df[fecha_col], errors="coerce")


    # -----------------------------
    # Detectar columnas COMP y %
    # -----------------------------
    def es_carga_comb(col):
        c = col.lower()
        return (
            "carga" in c and
            ("comb" in c or "combustible" in c)
        )

    comp_cols = [
        c for c in df.columns
        if "comp" in c.lower()
        and "porc" not in c.lower()
        and es_carga_comb(c)
    ]
    
    porc_cols = [
        c for c in df.columns
        if "porccomp" in c.lower()
        and es_carga_comb(c)
    ]
    comp_cols = sorted(comp_cols)
    porc_cols = sorted(porc_cols)
    
    if not comp_cols or not porc_cols:
        raise ValueError("No se detectaron columnas COMP / PORCCOMP")

    # Convertir porcentajes
    df[porc_cols] = df[porc_cols].apply(pd.to_numeric, errors="coerce")

    # -----------------------------
    # Crear SLOP si no suma 100
    # -----------------------------
    df["SUMA_COMP"] = df[porc_cols].sum(axis=1)

    df["SLOP"] = np.where(
        df["SUMA_COMP"] < 100,
        100 - df["SUMA_COMP"],
        0
    )

    # -----------------------------
    # Crear tabla diaria
    # -----------------------------
    registros = []

    for i in range(min(len(comp_cols), len(porc_cols))):

        tmp = df[["Fecha", porc_cols[i], comp_cols[i]]].copy()
        tmp.columns = ["Fecha", "Porcentaje", "Especie"]
        tmp["COMP"] = f"COMP{i+1}"

        registros.append(tmp)

    detalle = pd.concat(registros, ignore_index=True)

    detalle["Porcentaje"] = pd.to_numeric(detalle["Porcentaje"], errors="coerce")
    detalle = detalle.dropna()

    detalle = detalle[detalle["Especie"] != "-"]

    # -----------------------------
    # Añadir SLOP
    # -----------------------------
    slop_df = df[["Fecha", "SLOP"]].copy()
    slop_df = slop_df[slop_df["SLOP"] > 0]
    slop_df["COMP"] = "SLOP"
    slop_df["Especie"] = "SLOP"
    slop_df.columns = ["Fecha", "Porcentaje", "COMP", "Especie"]

    detalle = pd.concat([detalle, slop_df], ignore_index=True)

    return detalle

def añadir_proceso_a_dias_crudo(df_dias_crudo, df_proc):

    if df_proc is None or df_proc.empty:
        return df_dias_crudo

    df_proc = df_proc.copy()

    df_proc["Fecha"] = pd.to_datetime(df_proc["Fecha"])
    df_dias_crudo["Fecha"] = pd.to_datetime(df_dias_crudo["Fecha"])

    # Merge por fecha
    df_merge = pd.merge(
        df_dias_crudo,
        df_proc,
        on="Fecha",
        how="left"
    )

    return df_merge



def asignar_crudos_a_segmentos(detalle_crudos, processed_sheets):

    resultados = []

    for key, data in processed_sheets.items():

        if not data.get("saved"):
            continue

        sonda = data["source_name"]
        hoja = data["hoja"]

        for i, seg in enumerate(data["segmentos_validos"], start=1):

            fi = pd.to_datetime(seg["fecha_ini"])
            ff = pd.to_datetime(seg["fecha_fin"])

            sub = detalle_crudos[
                (detalle_crudos["Fecha"] >= fi) &
                (detalle_crudos["Fecha"] <= ff)
            ]

            if sub.empty:
                continue

            suma = (
                sub.groupby("Especie")["Porcentaje"]
                .sum()
                .reset_index()
            )
            
            total = suma["Porcentaje"].sum()
            
            suma["Porcentaje"] = (
                suma["Porcentaje"] / total * 100
            )
            
            resumen = suma

            resumen["Segmento"] = i
            resumen["Sonda"] = sonda
            resumen["Hoja"] = hoja
            resumen["Fecha_inicio"] = fi
            resumen["Fecha_fin"] = ff
            resumen["Velocidad"] = seg.get("vel_abs")

            resultados.append(resumen)

    if resultados:
        return pd.concat(resultados, ignore_index=True)

    return pd.DataFrame()

def construir_dataset_crudos_segmentos(detalle_crudos, processed_sheets):

    filas = []

    for key, data in processed_sheets.items():

        if not data.get("saved"):
            continue

        sonda = data["source_name"]
        hoja = data["hoja"]

        for i, seg in enumerate(data["segmentos_validos"], start=1):

            fi = pd.to_datetime(seg["fecha_ini"])
            ff = pd.to_datetime(seg["fecha_fin"])

            sub = detalle_crudos[
                (detalle_crudos["Fecha"] >= fi) &
                (detalle_crudos["Fecha"] <= ff)
            ]

            if sub.empty:
                continue

            suma = (
                sub.groupby("Especie")["Porcentaje"]
                .sum()
                .reset_index()
            )
            
            total = suma["Porcentaje"].sum()
            
            suma["Porcentaje"] = (
                suma["Porcentaje"] / total * 100
            )
            
            resumen = suma

            medias_proc = seg.get("medias", {})

            for _, row in resumen.iterrows():

                fila = {
                    "Sonda": sonda,
                    "Hoja": hoja,
                    "Segmento": i,
                    "Crudo": row["Especie"],
                    "Porcentaje_promedio": row["Porcentaje"],
                    "Velocidad_corr": seg.get("vel_abs"),
                    "Fecha_inicio": fi,
                    "Fecha_fin": ff
                }

                if isinstance(medias_proc, (dict, pd.Series)):
                    for k,v in medias_proc.items():
                        fila[k] = v

                filas.append(fila)

    if filas:
        return pd.DataFrame(filas)

    return pd.DataFrame()
def analizar_crudos_agresividad(df_master):

    if df_master.empty:
        return pd.DataFrame()

    resultados = []

    for crudo in df_master["Crudo"].unique():

        sub = df_master[df_master["Crudo"] == crudo].dropna()

        if len(sub) < 3:
            continue

        x = sub["Porcentaje_promedio"]
        y = sub["Velocidad_corr"]

        if x.std() == 0:
            continue

        corr = np.corrcoef(x, y)[0,1]

        # 🔥 Velocidad media ponderada por porcentaje
        vel_pond = np.average(y, weights=x)

        resultados.append({
            "Crudo": crudo,
            "Correlación % vs corrosión": corr,
            "% medio en segmentos": x.mean(),
            "Velocidad media simple": y.mean(),
            "Velocidad media ponderada": vel_pond,
            "Score agresividad": abs(corr) * vel_pond
        })

    if not resultados:
        return pd.DataFrame()

    return pd.DataFrame(resultados).sort_values(
        "Score agresividad",
        ascending=False
    )
def ranking_cestas_por_estado(df_cestas):

    if df_cestas.empty:
        return pd.DataFrame()

    df_rank = (
        df_cestas
        .groupby(["Especies", "Estado"])
        .agg(
            num_veces=("Cesta_id", "count"),
            dias_totales=("Dias", "sum"),
            vel_media=("Velocidad", "mean")
        )
        .reset_index()
    )

    # Pivot para verlo tipo matriz
    df_pivot = df_rank.pivot_table(
        index="Especies",
        columns="Estado",
        values="num_veces",
        fill_value=0
    )

    df_pivot["TOTAL"] = df_pivot.sum(axis=1)

    return df_pivot.sort_values("TOTAL", ascending=False)
from sklearn.ensemble import RandomForestRegressor

def analizar_ml_por_cesta(df_cestas, vars_proceso):

    resultados = []

    if df_cestas.empty:
        return pd.DataFrame()

    cestas_unicas = df_cestas["Especies"].unique()

    for cesta in cestas_unicas:

        sub = df_cestas[df_cestas["Especies"] == cesta].copy()

        if len(sub) < 5:
            continue

        # variables disponibles
        vars_validas = [v for v in vars_proceso if v in sub.columns]

        if not vars_validas:
            continue

        X = sub[vars_validas].copy()
        y = sub["Velocidad"].copy()

        # limpiar
        X = X.apply(pd.to_numeric, errors="coerce")
        y = pd.to_numeric(y, errors="coerce")

        mask = (~X.isna().any(axis=1)) & (~y.isna())
        X = X[mask]
        y = y[mask]

        if len(X) < 5:
            continue

        try:
            model = RandomForestRegressor(
                n_estimators=200,
                max_depth=5,
                random_state=42
            )

            model.fit(X, y)

            importancias = model.feature_importances_

            for var, imp in zip(vars_validas, importancias):

                resultados.append({
                    "Cesta": cesta,
                    "Variable": var,
                    "Importancia": imp
                })

        except Exception:
            continue

    if not resultados:
        return pd.DataFrame()

    df_imp = pd.DataFrame(resultados)

    return df_imp.sort_values(
        ["Cesta", "Importancia"],
        ascending=[True, False]
    )

def resumen_top_variables(df_imp, top_n=3):

    if df_imp.empty:
        return pd.DataFrame()

    return (
        df_imp
        .groupby("Cesta")
        .head(top_n)
        .reset_index(drop=True)
    )
def cargar_proceso_primera_hoja_limpio(path_excel):

    # Leer primera hoja completa
    df_raw = pd.read_excel(path_excel, sheet_name=0, header=None)
    
    # 🔧 LIMPIEZA BRUTAL PARA EVITAR ERRORES DEL EXCELç
    # 🔧 Limpieza total antes de cualquier operación
    def limpiar_celda(x):
        if isinstance(x, (int, float, str)) or pd.isna(x):
            return x
        return np.nan  # listas/arrays/objetos → NaN

    df_raw = df_raw.applymap(limpiar_celda)

    df_raw = df_raw.replace(
        ["nan", "NaN", "None", "<NA>", "N/A", "NA", "", " "],
        np.nan
    )


    # Buscar la primera fila que contenga al menos un número (datos reales)
    # Buscar fila donde aparezca columna tipo Fecha / Time / Timestamp
    fila_inicio = None
    
    for i in range(len(df_raw)):
        fila = df_raw.iloc[i].astype(str).str.lower()
    
        if any(
            any(k in celda for k in ["fecha", "time", "timestamp", "date"])
            for celda in fila
        ):
            fila_inicio = i
            break
    
    if fila_inicio is None:
        fila_inicio = 0


    if fila_inicio is None:
        raise ValueError("No se encontraron filas con datos numéricos en el archivo de proceso.")

    # Usamos esa fila como cabecera
    df_raw.columns = [str(c).strip() for c in df_raw.iloc[fila_inicio]]
    df = df_raw.iloc[fila_inicio+1:].reset_index(drop=True)

    # Reemplazar columnas vacías por nombres seguros
    df.columns = [f"Var_{i}" if c == "" or c.lower().startswith("unnamed") else c for i,c in enumerate(df.columns)]

    # Crear columna fecha artificial si no existe
    if "Fecha" not in df.columns:
        df["Fecha"] = pd.date_range(start="2000-01-01", periods=len(df), freq="D")
        df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
        df = df.dropna(subset=["Fecha"])


    # Convertir todo lo posible a numérico
    for c in df.columns:
        if c != "Fecha":
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df[c] = (
                df[c]
                .astype(str)
                .str.replace(",", ".")
            )
            
            df[c] = pd.to_numeric(df[c], errors="coerce")


    # Quitar columnas completamente vacías
    df = df.dropna(axis=1, how="all")

    # Variables de proceso (todas menos fecha)
    vars_proceso = [c for c in df.columns if c != "Fecha"]

    return df, vars_proceso

def dividir_segmento_por_intervalo(
        df_filtrado,
        segmento,
        df_proc,
        vars_proceso,
        intervalo_offset,
        min_dias=5
):
    nuevos_segmentos = []

    fi = pd.to_datetime(segmento["fecha_ini"])
    ff = pd.to_datetime(segmento["fecha_fin"])

    fechas = pd.date_range(start=fi, end=ff, freq=intervalo_offset)

    fechas = list(fechas)
    if fechas[-1] != ff:
        fechas.append(ff)

    for i in range(len(fechas)-1):

        sub_ini = fechas[i]
        sub_fin = fechas[i+1]

        sub_df = df_filtrado[
            (df_filtrado["Sent Time"] >= sub_ini) &
            (df_filtrado["Sent Time"] <= sub_fin)
        ]

        if len(sub_df) < 3:
            continue

        y = sub_df["UT measurement (mm)"].values
        delta = (sub_fin - sub_ini).days

        if delta < min_dias:
            continue

        velocidad = (y[-1] - y[0]) / (delta / 365.25)

        # medias proceso
        medias = {}
        if df_proc is not None:
            sub_proc = df_proc[
                (df_proc["Fecha"] >= sub_ini) &
                (df_proc["Fecha"] <= sub_fin)
            ]
        
            medias = sub_proc.mean(numeric_only=True)
        
            # ⭐ FALLBACK SI NO HAY DATOS EN EL SEGMENTO
            if medias.empty:
                medias = df_proc.mean(numeric_only=True)


        nuevos_segmentos.append({
            "ini": sub_df.index.min(),
            "fin": sub_df.index.max(),
            "fecha_ini": sub_ini,
            "fecha_fin": sub_fin,
            "delta_dias": delta,
            "velocidad": velocidad,
            "vel_abs": abs(velocidad),
            "medias": medias,
            "estado": "valido"
        })

    return nuevos_segmentos
def analizar_variables_en_crudo(df_master, crudo_objetivo, vars_proceso):

    sub = df_master[df_master["Crudo"] == crudo_objetivo]

    if sub.empty:
        return pd.DataFrame()

    resultados = []

    for var in vars_proceso:

        if var not in sub.columns:
            continue

        sub2 = sub[[var, "Velocidad_corr"]].dropna()

        if len(sub2) < 3:
            continue

        x = sub2[var]
        y = sub2["Velocidad_corr"]

        if x.std() == 0:
            continue

        corr = np.corrcoef(x, y)[0,1]

        resultados.append({
            "Variable proceso": var,
            "Correlación con corrosión": corr,
            "Valor absoluto": abs(corr)
        })

    if not resultados:
        return pd.DataFrame()

    return pd.DataFrame(resultados).sort_values(
        "Valor absoluto",
        ascending=False
    )

def aplicar_segmentacion_referencia(
        df_filtrado,
        segmentos_ref,
        df_proc,
        vars_proceso,
        segmentos_validos_previos=None,
        segmentos_descartados_previos=None,
        min_dias=5
):

    nuevos_segmentos = []

    if not segmentos_validos_previos:
        return []

    for ref in segmentos_ref:

        fi_ref = pd.to_datetime(ref["fecha_ini"])
        ff_ref = pd.to_datetime(ref["fecha_fin"])

        # ====================================
        # 1️⃣ INTERSECCIÓN CON ZONAS VÁLIDAS
        # ====================================

        for seg_prev in segmentos_validos_previos:

            fi_prev = pd.to_datetime(seg_prev["fecha_ini"])
            ff_prev = pd.to_datetime(seg_prev["fecha_fin"])

            fi = max(fi_ref, fi_prev)
            ff = min(ff_ref, ff_prev)

            if fi >= ff:
                continue

            # ====================================
            # 2️⃣ BLOQUEAR SI SOLAPA CON DESCARTADOS
            # ====================================

            if segmentos_descartados_previos:

                solapa_gris = False

                for desc in segmentos_descartados_previos:

                    # 🔒 Si no tiene fechas, saltarlo
                    if "fecha_ini" not in desc or "fecha_fin" not in desc:
                        continue
                
                    fi_desc = pd.to_datetime(desc["fecha_ini"])
                    ff_desc = pd.to_datetime(desc["fecha_fin"])
                
                    if not (ff <= fi_desc or fi >= ff_desc):
                        solapa_gris = True
                        break


                if solapa_gris:
                    continue

            # ====================================
            # 3️⃣ CREAR SEGMENTO
            # ====================================

            sub_df = df_filtrado[
                (df_filtrado["Sent Time"] >= fi) &
                (df_filtrado["Sent Time"] <= ff)
            ]

            if sub_df.empty:
                continue

            delta = (ff - fi).days

            if delta < min_dias:
                continue

            y = sub_df["UT measurement (mm)"].values

            velocidad = (y[-1] - y[0]) / (delta / 365.25)

            medias = {}

            if df_proc is not None:
                sub_proc = df_proc[
                    (df_proc["Fecha"] >= fi) &
                    (df_proc["Fecha"] <= ff)
                ]

                medias = sub_proc.mean(numeric_only=True)

                if medias.empty:
                    medias = df_proc.mean(numeric_only=True)

            nuevos_segmentos.append({
                "ini": sub_df.index.min(),
                "fin": sub_df.index.max(),
                "fecha_ini": fi,
                "fecha_fin": ff,
                "delta_dias": delta,
                "velocidad": velocidad,
                "vel_abs": abs(velocidad),
                "medias": medias,
                "estado": "valido"
            })

    return sorted(nuevos_segmentos, key=lambda x: x["fecha_ini"])
def analisis_porcentaje_crudos_top_cestas(df_cestas, detalle_crudos):

    resultados = []

    for estado in ["ENCIMA", "DEBAJO", "DENTRO"]:

        sub_estado = df_cestas[df_cestas["Estado"] == estado]

        if sub_estado.empty:
            continue

        # Top 5 cestas
        top_cestas = (
            sub_estado["Especies"]
            .value_counts()
            .head(5)
            .index
        )

        for cesta in top_cestas:

            sub_cesta = sub_estado[sub_estado["Especies"] == cesta]

            porcentajes = []

            for _, row in sub_cesta.iterrows():

                fi = row["Fecha_ini"]
                ff = row["Fecha_fin"]

                sub = detalle_crudos[
                    (detalle_crudos["Fecha"] >= fi) &
                    (detalle_crudos["Fecha"] <= ff)
                ]

                if sub.empty:
                    continue

                suma = (
                    sub.groupby("Especie")["Porcentaje"]
                    .sum()
                )

                total = suma.sum()

                if total == 0:
                    continue

                pct = (suma / total * 100).to_dict()

                porcentajes.append(pct)

            # promedio de porcentajes
            if porcentajes:

                df_pct = pd.DataFrame(porcentajes).fillna(0)

                media_pct = df_pct.mean()

                for crudo, val in media_pct.items():

                    resultados.append({
                        "Estado": estado,
                        "Cesta": cesta,
                        "Crudo": crudo,
                        "% promedio": val
                    })

    return pd.DataFrame(resultados)
def enriquecer_cestas_con_proceso(df_cestas, vars_proceso):

    if df_cestas.empty:
        return pd.DataFrame()

    filas = []

    for cesta in df_cestas["Especies"].unique():

        sub = df_cestas[df_cestas["Especies"] == cesta]

        fila = {
            "Cesta": cesta,
            "num_veces": len(sub),
            "dias_totales": sub["Dias"].sum(),
            "vel_media": sub["Velocidad"].mean()
        }

        for var in vars_proceso:

            if var not in sub.columns:
                continue

            vals = pd.to_numeric(sub[var], errors="coerce").dropna()

            if len(vals) > 0:
                fila[f"{var}_mean"] = vals.mean()
                fila[f"{var}_std"] = vals.std()

        filas.append(fila)

    return pd.DataFrame(filas)

from scipy.stats import spearmanr

def analizar_variables_por_cesta_simple(df_cestas, vars_proceso):

    resultados = []

    if df_cestas.empty:
        return pd.DataFrame()

    for cesta in df_cestas["Especies"].unique():

        sub = df_cestas[df_cestas["Especies"] == cesta]

        if len(sub) < 3:
            continue

        for var in vars_proceso:

            if var not in sub.columns:
                continue

            x = pd.to_numeric(sub[var], errors="coerce")
            y = pd.to_numeric(sub["Velocidad"], errors="coerce")

            mask = (~x.isna()) & (~y.isna())

            x = x[mask]
            y = y[mask]

            if len(x) < 3 or x.std() == 0:
                continue

            try:
                corr, _ = spearmanr(x, y)

                resultados.append({
                    "Cesta": cesta,
                    "Variable": var,
                    "Importancia": abs(corr),
                    "Correlacion": corr,
                    "n_puntos": len(x)
                })

            except:
                continue

    if not resultados:
        return pd.DataFrame()

    return pd.DataFrame(resultados).sort_values(
        ["Cesta", "Importancia"],
        ascending=[True, False]
    )

def make_safe_name(text: str) -> str:
    import re, unicodedata
    text = (text or "").strip()

    # Normaliza unicode (quita caracteres ocultos y acentos raros)
    text = unicodedata.normalize("NFKD", text)

    # Elimina todos los caracteres no permitidos en nombres de carpeta
    text = re.sub(r'[\/\\:\*\?"<>\|\n\r\t]+', '', text)

    # Reemplaza espacios por guiones bajos
    text = text.replace(' ', '_')

    # Evitar que quede vacío o demasiado largo
    return text[:120] or "sin_nombre"

def make_safe_slug(text: str, max_len: int = 120) -> str:
    """
    Convierte un nombre arbitrario en uno seguro para rutas/archivos:
    - Minúsculas
    - Sustituye espacios por '_'
    - Elimina caracteres problemáticos (/ \ : * ? " < > |)
    - Compacta guiones bajos repetidos
    """
    t = (text or "").strip().lower()
    t = t.replace(" ", "_")
    t = re.sub(r'[\/\\:\*\?"<>\|]+', '', t)  # quita caracteres no válidos
    t = re.sub(r'__+', '_', t)               # compacta underscores
    return t[:max_len] or "sin_nombre"

def leer_archivo(uploaded_file):
    hojas_dict = {}

    if uploaded_file.name.endswith(".xlsx"):
        # Leer Excel
        xls = pd.ExcelFile(uploaded_file)
        for sheet in xls.sheet_names:
            df = pd.read_excel(uploaded_file, sheet_name=sheet)
            df.columns = [str(c).strip() for c in df.columns]
            hojas_dict[sheet] = df

    elif uploaded_file.name.endswith(".zip"):
        # Leer ZIP con CSVs
        import zipfile
        with zipfile.ZipFile(uploaded_file) as z:
            for fname in z.namelist():
                if fname.lower().endswith(".csv"):
                    with z.open(fname) as f:
                        df = pd.read_csv(f, sep=",")
                        df.columns = [str(c).strip() for c in df.columns]
                        hojas_dict[fname.replace(".csv", "")] = df

    else:
        st.error("Formato no soportado. Usa .xlsx o .zip con CSVs.")

    return hojas_dict

# Configuración básica y estilo
st.set_page_config(page_title="Analizador de corrosión", layout="wide")
st.markdown("<h1 class='darkblue-title'>Análisis de corrosión</h1>", unsafe_allow_html=True)

st.markdown("""
<style>

/* =========================================================
   0. FONDO GENERAL → BLANCO
   ========================================================= */
html, body, .block-container, [class*="stApp"] {
    background-color: #FFFFFF !important;  /* BLANCO */
    color: #333333 !important;             /* texto gris oscuro */
}

/* =========================================================
   1. TITULOS GRANDES → POR DEFECTO NARANJA
   ========================================================= */
h1, h2, h3, h4, h5, h6 {
    color: #D98B3B !important;     /* naranja Repsol */
    font-weight: 800 !important;
}

/* =========================================================
   2. TITULOS AZUL OSCURO (solo si tú lo marcas con clase)
   ========================================================= */
.darkblue-title {
    color: #0B1A33 !important;     /* azul oscuro */
    font-weight: 800 !important;
}

/* =========================================================
   3. WIDGETS → letra gris oscuro
   ========================================================= */
.stSelectbox label,
.stMultiSelect label,
.stNumberInput label,
.stSlider label,
.stTextInput label {
    color: #333333 !important;
}

.css-16idsys, .css-1pndypt, .css-1offfwp, .css-1kyxreq {
    color: #333333 !important;
}

.stSelectbox div[data-baseweb="select"],
.stMultiSelect div[data-baseweb="select"] {
    background-color: white !important;
    color: #333333 !important;
}

/* =========================================================
   4. TABS → gris / ROJO seleccionada
   ========================================================= */
.stTabs [data-baseweb="tab"] p {
    color: #666666 !important;   /* gris */
    font-weight: 600 !important;
}

.stTabs [aria-selected="true"] p {
    color: red !important;       /* ROJO al seleccionar */
    font-weight: 700 !important;
}

/* Fondo de tabs */
.stTabs [data-baseweb="tab"] {
    background-color: #FFFFFF !important; /* fondo blanco */
}

/* =========================================================
   5. Botones → NARANJAS
   ========================================================= */
.stButton>button {
    background-color: #D98B3B !important;
    color: white !important;
    border-radius: 8px;
}
.stButton>button:hover {
    background-color: #b57830 !important;
    color: white !important;
}

</style>
""", unsafe_allow_html=True)

# intentar cargar logo
try:
    logo_original = Image.open("logo_repsol.png").convert("RGBA")
    blur_radius = 20
    padding = blur_radius * 5
    new_size = (logo_original.width + padding, logo_original.height + padding)
    final_logo = Image.new("RGBA", new_size, (0, 0, 0, 0))
    center_x = (new_size[0] - logo_original.width) // 2
    center_y = (new_size[1] - logo_original.height) // 2
    final_logo.paste(logo_original, (center_x, center_y), logo_original)
    mask = final_logo.split()[3]
    white_halo = Image.new("RGBA", final_logo.size, (255, 255, 255, 0))
    white_halo.putalpha(mask.filter(ImageFilter.GaussianBlur(blur_radius)))
    final_logo = Image.alpha_composite(white_halo, final_logo)
    st.image(final_logo, width=200)
except Exception:
    st.write("⚠️ No se encontró 'logo_repsol.png' o no se pudo procesarlo.")

HERE = Path.cwd()

# -------------------- Intentar cargar script del usuario (si existe) --------------------
def load_user_module_from_folder(folder: Path):
    py_files = list(folder.glob("*.py"))
    if not py_files:
        return None, None
    candidates = [f for f in py_files if "intento" in f.stem.lower() or "interfaz" in f.stem.lower()]
    if not candidates:
        candidates = sorted(py_files, key=lambda x: x.stat().st_size, reverse=True)
    chosen = candidates[0]
    try:
        spec = importlib.util.spec_from_file_location("user_script", str(chosen))
        module = importlib.util.module_from_spec(spec)
        sys.modules["user_script"] = module
        spec.loader.exec_module(module)
        return module, chosen
    except Exception:
        return None, chosen

user_module, user_module_path = load_user_module_from_folder(HERE)

def safe_get(fn_name):
    if user_module is None:
        return None
    return getattr(user_module, fn_name, None)

# -------------------- Barra lateral: entradas y estado --------------------
st.sidebar.header("Entradas y parámetros")
uploaded_corr = st.sidebar.file_uploader("Archivo de corrosión (.xlsx)", type=None, key="file_uploader_corr")
uploaded_proc = st.sidebar.file_uploader("Archivo de proceso (.xlsx) — opcional", type=None, key="file_uploader_proc")
uploaded_crudos = st.sidebar.file_uploader(
    "Archivo de crudos de petróleo (.xlsx) — opcional",
    type=None,
    key="file_uploader_crudos"
)
uploaded_mpa = st.sidebar.file_uploader(
    "Archivo curvas corrosión MPA (.xlsx)",
    type=["xlsx"],
    key="file_uploader_mpa"
)
st.sidebar.markdown("---")
st.sidebar.header("División global de segmentos")

tipo_intervalo_global = st.sidebar.selectbox(
    "Tipo intervalo global",
    ["Días", "Meses", "Años"],
    key="tipo_intervalo_global"
)

valor_intervalo_global = st.sidebar.number_input(
    "Cantidad intervalo global",
    min_value=1,
    value=30,
    key="valor_intervalo_global"
)

btn_dividir_global = st.sidebar.button("Dividir TODA la gráfica")

def cargar_y_limpiar_mpa(uploaded_file):

    df = pd.read_excel(uploaded_file)

    df.columns = [str(c).strip() for c in df.columns]

    # eliminar fila unidades
    df = df.iloc[1:].reset_index(drop=True)

    # corregir nombre 5 Cr
    if "5 Cr " in df.columns:
        df.rename(columns={"5 Cr ": "5 Cr"}, inplace=True)

    df["Temperature"] = pd.to_numeric(df["Temperature"], errors="coerce")
    df["Acid Measurement"] = pd.to_numeric(df["Acid Measurement"], errors="coerce")

    for col in ["Carbon Steel", "5 Cr"]:

        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", ".")
            .str.replace("<", "")
            .str.strip()
        )

        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Temperature", "Acid Measurement"])

    return df

if uploaded_mpa is not None:

    try:
        df_mpa = cargar_y_limpiar_mpa(uploaded_mpa)

        st.session_state["df_mpa"] = df_mpa
        st.sidebar.success("Curvas MPA cargadas")

    except Exception as e:
        st.sidebar.error(f"Error leyendo MPA: {e}")

st.sidebar.markdown("---")

umbral_factor = st.sidebar.slider(
    "Umbral factor",
    min_value=1.0000,
    max_value=1.1000,
    value=1.0200,
    step=0.0001,
    format="%.4f",
    key="umbral_factor"
)

umbral = st.sidebar.number_input(
    "Umbral (ej: 0.0005)",
    min_value=1e-9,
    value=0.0005,
    step=0.0001,
    format="%.6f",
    key="umbral"
)
umbral_error_segmento = st.sidebar.slider(
    "Umbral error segmentos (%)",
    min_value=0.0,
    max_value=200.0,
    value=30.0,
    step=1.0,
    key="umbral_error_segmento"
)
min_dias_seg = st.sidebar.number_input("Mínimo días por segmento", min_value=1, max_value=3650, value=10, key="min_dias_seg")
fig_w = st.sidebar.slider("Ancho figura", 6, 20, 14, key="fig_w")
fig_h = st.sidebar.slider("Alto figura", 4, 16, 10, key="fig_h")

st.sidebar.markdown("---")
st.sidebar.header("Estado del script")
if user_module is not None:
    st.sidebar.success(f"Módulo cargado: {user_module_path.name}")
    funcs = ["detectar_segmentos","extraer_segmentos_validos","dibujar_grafica_completa","recalcular_segmento_local","guardar_resultados"]
    exist = [f for f in funcs if getattr(user_module, f, None) is not None]
    miss = [f for f in funcs if getattr(user_module, f, None) is None]
    st.sidebar.write("Funciones detectadas:")
    if exist:
        st.sidebar.write("✅ " + ", ".join(exist))
    if miss:
        st.sidebar.write("⚠️ Faltan (se usarán fallbacks): " + ", ".join(miss))
else:
    st.sidebar.info("No se encontró script de usuario en la carpeta (se usarán fallbacks).")

# -------------------- Caching lectura Excel --------------------
@st.cache_data(show_spinner=False)
def cached_read_excel_sheets(uploaded_file):
    if uploaded_file is None:
        return []

    try:
        if uploaded_file.name.endswith(".xlsx"):
            xls = pd.ExcelFile(uploaded_file)
            return xls.sheet_names

        elif uploaded_file.name.endswith(".zip"):
            import zipfile
            with zipfile.ZipFile(uploaded_file) as z:
                csv_files = [fname.replace(".csv", "") for fname in z.namelist() if fname.lower().endswith(".csv")]
            return csv_files

        else:
            return []

    except Exception:
        return []

@st.cache_data(show_spinner=False)
def cached_read_excel_sheet_df(uploaded_file, sheet_name):
    if uploaded_file is None:
        return pd.DataFrame()

    try:
        if uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
            df.columns = [str(c).strip() for c in df.columns]
            return df

        elif uploaded_file.name.endswith(".zip"):
            import zipfile
            with zipfile.ZipFile(uploaded_file) as z:
                # Buscar el archivo CSV que coincide con sheet_name
                fname = next((f for f in z.namelist() if f.lower().endswith(".csv") and sheet_name in f), None)
                if fname:
                    with z.open(fname) as f:
                        df = pd.read_csv(f, sep=",")
                        df.columns = [str(c).strip() for c in df.columns]
                        return df

        return pd.DataFrame()

    except Exception:
        return pd.DataFrame()


# -------------------- Funciones fallback --------------------
def detect_columns_fallback(df):
    col_fecha = None
    col_espesor = None
    for c in df.columns:
        cl = str(c).lower()
        if any(k in cl for k in ["sent time", "sent_time", "senttime", "sent", "timestamp"]):
            col_fecha = c
            break
    if col_fecha is None:
        for c in df.columns:
            cl = str(c).lower()
            if any(k in cl for k in ["fecha", "date", "time"]):
                col_fecha = c
                break
    for c in df.columns:
        cl = str(c).lower()
        if any(k in cl for k in ["ut measurement", "ut", "measurement", "mm", "espesor", "thickness"]):
            col_espesor = c
            break
    if col_espesor is None:
        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]):
                col_espesor = c
                break
    if col_fecha is None or col_espesor is None:
        for c in df.columns:
            try:
                sample = df[c].dropna().iloc[:5]
                parsed = False
                for v in sample:
                    try:
                        pd.to_datetime(v)
                        parsed = True
                        break
                    except Exception:
                        parsed = False
                if parsed and col_fecha is None:
                    col_fecha = c
                    break
            except Exception:
                continue
        for c in df.columns:
            if col_espesor is None and pd.api.types.is_numeric_dtype(df[c]):
                col_espesor = c
                break
    return col_fecha, col_espesor
def analizar_importancia_variables(df, vars_proceso):

    resultados = []

    for var in vars_proceso:

        if var not in df.columns:
            continue

        sub = df[[var, "Velocidad experimental"]].dropna()

        if len(sub) < 3:
            continue

        x = sub[var]
        y = sub["Velocidad experimental"]

        if x.std() == 0:
            continue

        corr = np.corrcoef(x, y)[0,1]

        resultados.append({
            "Variable proceso": var,
            "Correlación": corr,
            "Importancia": abs(corr)
        })

    if not resultados:
        return pd.DataFrame()

    return pd.DataFrame(resultados).sort_values(
        "Importancia",
        ascending=False
    )
def detectar_segmentos_fallback(df_original, umbral_factor=1.02, umbral=0.0005, min_dias=10, wl_max=51, wl_min=5):
    df = df_original.copy()
    try:
        col_fecha, col_espesor = detect_columns_fallback(df)
    except Exception:
        return None, None, [], []
    df["Sent Time"] = pd.to_datetime(df[col_fecha], errors="coerce")
    df["UT measurement (mm)"] = pd.to_numeric(df[col_espesor], errors="coerce")
    df = df.sort_values("Sent Time").reset_index(drop=True)
    df = df.dropna(subset=["Sent Time", "UT measurement (mm)"]).reset_index(drop=True)
    if len(df) < 5:
        return df, None, [], []
    n_ref = min(10, len(df))
    grosor_ref = df["UT measurement (mm)"].iloc[:n_ref].mean()
    df_filtrado = df[df["UT measurement (mm)"] <= grosor_ref * umbral_factor].reset_index(drop=True)
    if len(df_filtrado) < 5:
        return df_filtrado, None, [], []
    y = df_filtrado["UT measurement (mm)"].values
    wl = min(21, len(y) - 1)
    wl = max(wl_min, wl)
    if wl % 2 == 0:
        wl += 1
    try:
        from scipy.signal import savgol_filter
        y_suave = savgol_filter(y, wl, 3)
    except Exception:
        y_suave = y.copy()
    pendiente = np.gradient(y_suave)
    cambios = [0]
    for i in range(1, len(pendiente)):
        if abs(pendiente[i] - pendiente[i - 1]) > umbral:
            cambios.append(i)
    cambios.append(len(y_suave) - 1)
    segmentos_raw = []
    for k in range(len(cambios) - 1):
        ini, fin = cambios[k], cambios[k + 1]
        if ini < 0 or fin <= ini or fin > len(df_filtrado):
            continue
        fecha_ini = pd.to_datetime(df_filtrado["Sent Time"].iloc[ini], errors="coerce")
        fecha_fin = pd.to_datetime(df_filtrado["Sent Time"].iloc[fin - 1], errors="coerce")
        delta_dias = (fecha_fin - fecha_ini).days if (pd.notna(fecha_ini) and pd.notna(fecha_fin)) else 0
        velocidad = np.nan
        if delta_dias > 0:
            try:
                velocidad = (y_suave[fin - 1] - y_suave[ini]) / (delta_dias / 365.25)
            except Exception:
                velocidad = np.nan
        segmentos_raw.append({"ini": ini, "fin": fin, "fecha_ini": fecha_ini, "fecha_fin": fecha_fin, "delta_dias": delta_dias, "velocidad": velocidad})
    return df_filtrado, np.asarray(y_suave), cambios, segmentos_raw

def extraer_segmentos_validos_fallback(df_filtrado, y_suave, segmentos_raw, df_proc=None, vars_proceso=None, min_dias=10):
    segmentos_validos = []
    descartados = []

    fecha_col = None
    if df_proc is not None and not df_proc.empty:
        for c in df_proc.columns:
            if any(k in str(c).lower() for k in ["fecha", "date", "time", "sent"]):
                fecha_col = c
                break
        if fecha_col is None:
            fecha_col = df_proc.columns[0]
        try:
            df_proc[fecha_col] = pd.to_datetime(df_proc[fecha_col], errors="coerce")
        except Exception:
            pass

    for seg in segmentos_raw:
        ini, fin = seg["ini"], seg["fin"]
        fecha_ini, fecha_fin = seg["fecha_ini"], seg["fecha_fin"]
        delta_dias = seg["delta_dias"]
        velocidad = seg["velocidad"]

        if pd.isna(fecha_ini) or pd.isna(fecha_fin):
            seg2 = dict(seg); seg2.update({"motivo": "Fechas inválidas", "estado": "descartado"})
            descartados.append(seg2)
            continue
        if delta_dias <= 0 or delta_dias < min_dias:
            seg2 = dict(seg); seg2.update({"motivo": f"Duración < {min_dias} días", "estado": "descartado"})
            descartados.append(seg2)
            continue
        if velocidad is None or (not np.isfinite(velocidad)) or velocidad >= 0:
            seg2 = dict(seg); seg2.update({"motivo": "Velocidad no negativa o NaN", "estado": "descartado"})
            descartados.append(seg2)
            continue

        medias = pd.Series(dtype=float)
        if df_proc is not None and not df_proc.empty and fecha_col in df_proc.columns:
            try:
                sub = df_proc[
                    (df_proc[fecha_col] >= fecha_ini - pd.Timedelta(days=1))
                    & (df_proc[fecha_col] <= fecha_fin + pd.Timedelta(days=1))
                ]
                
                # --- 🔧 Limpieza robusta ANTES de calcular medias ---
                
                # 1. Convertir a numérico todo lo que debe ser numérico
                for col in sub.columns:
                    if col != "Fecha":
                        sub[col] = pd.to_numeric(sub[col], errors="coerce")
                
                # 2. Quedarse solo con columnas numéricas + Fecha
                cols_num = [c for c in sub.columns if c != "Fecha" and pd.api.types.is_numeric_dtype(sub[c])]
                sub = sub[ ["Fecha"] + cols_num ]
                
                # 3. Eliminar cualquier celda que sea lista/array/dict/objeto raro
                sub = sub.applymap(
                    lambda x: x if isinstance(x, (int, float)) or pd.isna(x) else np.nan
                )
                
                # Finalmente, calcular medias
                medias = sub.mean(numeric_only=True)
                if medias.empty and df_proc is not None:
                    medias = df_proc.mean(numeric_only=True)
                

            except Exception:
                medias = pd.Series(dtype=float)

        dur_days = delta_dias
        anios = dur_days // 365
        meses = (dur_days % 365) // 30
        if anios == 0 and meses == 0 and dur_days > 0:
            meses = 1

        segmentos_validos.append({
            "ini": ini,
            "fin": fin,
            "fecha_ini": fecha_ini,
            "fecha_fin": fecha_fin,
            "delta_dias": delta_dias,
            "velocidad": velocidad,
            "vel_abs": abs(velocidad),
            "medias": medias,
            "anios": anios,
            "meses": meses,
            "estado": "valido",
            "num_segmento_valido": None
        })

    return segmentos_validos, descartados
def construir_tabla_corregida(processed_sheets, df_mpa, material, sondas_activas):

    processed_sheets = {
        k: v for k, v in processed_sheets.items()
        if k in sondas_activas
    }

    df_comp = construir_tabla_segmentos_comparativa(
        processed_sheets,
        df_mpa,
        material
    )

    if df_comp.empty:
        return df_comp

    df_comp["Velocidad experimental"] = df_comp["Media velocidades"]
    df_comp["Velocidad teórica"] = df_comp["Velocidad esperada"]

    return df_comp

def buscar_velocidad_mpa(df_mpa, temp, tan, material):

    if df_mpa is None or pd.isna(temp) or pd.isna(tan):
        return None

    col_temp = "Temperature"
    col_tan = "Acid Measurement"
    col_cs = "Carbon Steel"
    col_5cr = "5 Cr"

    df_mpa["dist"] = (
        (df_mpa[col_temp] - temp)**2 +
        (df_mpa[col_tan] - tan)**2
    )

    fila = df_mpa.loc[df_mpa["dist"].idxmin()]

    if material == "Carbon Steel":
        return fila.get(col_cs)
    else:
        return fila.get(col_5cr)
def obtener_cv_seguro(df):

    if df is None or df.empty:
        return None

    if "Coef Variación (%)" not in df.columns:
        return None

    return pd.to_numeric(df["Coef Variación (%)"], errors="coerce")
def buscar_velocidad_mas_cercana(df_mpa, temp, tan, material):

    if df_mpa is None or pd.isna(temp) or pd.isna(tan):
        return None

    df_tmp = df_mpa.copy()

    df_tmp["dist"] = (
        (df_tmp["Temperature"] - temp)**2 +
        (df_tmp["Acid Measurement"] - tan)**2
    )

    fila = df_tmp.loc[df_tmp["dist"].idxmin()]

    return fila.get(material)

def aplicar_umbral_error_segmentos(processed_sheets, df_comp, umbral_cv):

    if df_comp is None or df_comp.empty:
        return processed_sheets

    # segmentos válidos según CV
    cv = obtener_cv_seguro(df_comp)

    if cv is not None:
    
        df_validos = df_comp[
            cv.isna() | (cv <= umbral_cv)
        ].copy()
    
    else:
    
        df_validos = df_comp.copy()

    segmentos_validos = set(
        zip(
            pd.to_datetime(df_validos["Inicio"]),
            pd.to_datetime(df_validos["Fin"])
        )
    )

    nuevo = {}

    for key, data in processed_sheets.items():

        segs = data.get("segmentos_validos", [])

        filtrados = []

        for seg in segs:

            ini = pd.to_datetime(seg["fecha_ini"])
            fin = pd.to_datetime(seg["fecha_fin"])

            if (ini, fin) in segmentos_validos:
                filtrados.append(seg)

        data_copy = data.copy()
        data_copy["segmentos_validos"] = filtrados

        nuevo[key] = data_copy

    return nuevo
def calcular_segmentos_crudo(df):

    resumen = (
        df.groupby("Especie")
        .agg(
            num_segmentos=("Segmento", "nunique"),
            segmentos=("Segmento", lambda x: sorted(set(x)))
        )
        .reset_index()
    )

    resumen["segmentos"] = resumen["segmentos"].apply(
        lambda x: ", ".join(map(str, x))
    )

    return resumen
def calcular_perfil_teorico_por_segmentos(df_filtrado, segmentos, df_mpa, material):

    if df_mpa is None:
        return None

    df_teo = df_filtrado[["Sent Time", "UT measurement (mm)"]].copy()
    df_teo.rename(columns={"Sent Time": "Fecha"}, inplace=True)

    df_teo["Vel_teorica"] = np.nan

    # asignar velocidad teórica a cada fecha según segmento
    for seg in segmentos:

        if seg.get("estado") != "valido":
            continue

        medias = seg.get("medias")

        if medias is None:
            continue

        medias_dict = dict(medias)

        temp = medias_dict.get("T")
        tan = medias_dict.get("TAN")

        if pd.isna(temp) or pd.isna(tan):
            continue

        vel = buscar_velocidad_mas_cercana(
            df_mpa,
            float(temp),
            float(tan),
            material
        )

        fi = pd.to_datetime(seg["fecha_ini"])
        ff = pd.to_datetime(seg["fecha_fin"])

        mask = (df_teo["Fecha"] >= fi) & (df_teo["Fecha"] <= ff)

        df_teo.loc[mask, "Vel_teorica"] = vel

    # integrar velocidades → espesor teórico
    espesor_teo = [df_teo["UT measurement (mm)"].iloc[0]]

    fechas = pd.to_datetime(df_teo["Fecha"])

    for i in range(1, len(df_teo)):

        vel = df_teo["Vel_teorica"].iloc[i]

        if pd.isna(vel):
            espesor_teo.append(espesor_teo[-1])
            continue

        delta_dias = (fechas.iloc[i] - fechas.iloc[i-1]).days

        perdida = vel * (delta_dias / 365.25)

        espesor_teo.append(espesor_teo[-1] - perdida)

    df_teo["Espesor_teorico"] = espesor_teo

    return df_teo

def dibujar_grafica_completa_fallback(df_filtrado, y_suave, segmentos_validos, descartados, segmentos_eliminados_idx, titulo="Velocidad de corrosión", figsize=(14,10), show=False):
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white"); ax.grid(True, alpha=0.35)
    try:
        ax.plot(pd.to_datetime(df_filtrado["Sent Time"]), df_filtrado["UT measurement (mm)"].values, alpha=0.25, linewidth=1.2, label="Mediciones")
    except Exception:
        pass
    if y_suave is None:
        y_suave = np.asarray(df_filtrado["UT measurement (mm)"].values) if "UT measurement (mm)" in df_filtrado.columns else np.zeros(len(df_filtrado))
    ymax, ymin = float(np.max(y_suave)), float(np.min(y_suave)); altura = ymax - ymin if (ymax - ymin) != 0 else max(abs(ymax), 1.0)
    ax.set_ylim(ymin - 0.05 * altura, ymax + 0.2 * altura)
    gris_alpha = 0.35
    for d in descartados:
        i, f = d.get("ini",0), d.get("fin",0)
        if i < 0 or f <= i or f > len(y_suave): continue
        try:
            ax.plot(pd.to_datetime(df_filtrado["Sent Time"].iloc[i:f]), y_suave[i:f], alpha=gris_alpha, linewidth=2)
            ax.fill_between(pd.to_datetime(df_filtrado["Sent Time"].iloc[i:f]), y_suave[i:f], ymin, alpha=gris_alpha)
        except Exception:
            continue
    for (i,f) in segmentos_eliminados_idx:
        if i < 0 or f <= i or f > len(y_suave): continue
        try:
            ax.plot(pd.to_datetime(df_filtrado["Sent Time"].iloc[i:f]), y_suave[i:f], alpha=gris_alpha, linewidth=2)
            ax.fill_between(pd.to_datetime(df_filtrado["Sent Time"].iloc[i:f]), y_suave[i:f], ymin, alpha=gris_alpha)
        except Exception:
            continue
    validos = [s for s in segmentos_validos if s.get("estado","valido") == "valido"]
    colormap = plt.cm.get_cmap("turbo", max(2, len(validos)))
    contador = 0
    for s in sorted(segmentos_validos, key=lambda x: x.get("fecha_ini") or pd.Timestamp.max):
        if s.get("estado","valido") != "valido": continue
        contador += 1; s["num_segmento_valido"] = contador
        i, f = int(s["ini"]), int(s["fin"])
        color = colormap((contador - 1) % max(1, colormap.N))
        try:
            ax.plot(pd.to_datetime(df_filtrado["Sent Time"].iloc[i:f]), y_suave[i:f], color=color, linewidth=2.6, label=f"Segmento {contador}")
            ax.fill_between(pd.to_datetime(df_filtrado["Sent Time"].iloc[i:f]), y_suave[i:f], ymin, color=color, alpha=0.25)
            for fecha in [s["fecha_ini"], s["fecha_fin"]]:
                ax.axvline(fecha, color="black", linestyle=":", alpha=0.5, zorder=0)
                ax.text(fecha, ymax + 0.07 * altura, fecha.strftime("%Y-%m-%d"), ha="center", va="bottom", rotation=90, fontsize=8, color="black", bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85, lw=0))
            centro_idx = min((i + f) // 2, len(df_filtrado) - 1)
            x_centro = pd.to_datetime(df_filtrado["Sent Time"].iloc[centro_idx])
            y_centro = ymin + 0.45 * altura
            ax.text(x_centro, y_centro, f"{s['vel_abs']:.4f} mm/año", ha="center", va="center", rotation=90, fontsize=10, fontweight="bold", color=color, bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.9, lw=0))
        except Exception:
            continue
    ax.xaxis.set_major_locator(mdates.AutoDateLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center", fontsize=9)
    ax.set_title(titulo, fontsize=14, fontweight="bold"); ax.set_xlabel("Fecha", fontsize=12); ax.set_ylabel("UT measurement (mm)", fontsize=12)
    try:
        leg = ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=9, title="Segmentos", borderaxespad=0.)
        for text in leg.get_texts(): text.set_multialignment('left')
    except Exception:
        pass
    plt.tight_layout()
    return fig, ax

# Wrappers prefieren funciones del usuario
def detectar_segmentos_wrapper(df, umbral_factor_val, umbral_val):
    fn = safe_get("detectar_segmentos")
    if fn is not None:
        try:
            return fn(df, umbral_factor_val, umbral_val)
        except Exception:
            pass
    return detectar_segmentos_fallback(df, umbral_factor_val, umbral_val)
def graficar_segmentos(df, titulo):

    if df.empty:
        st.info("No hay segmentos")
        return

    import plotly.graph_objects as go

    sondas = [c for c in df.columns if c not in ["segmento","promedio","std","CV (%)"]]

    fig = go.Figure()

    for s in sondas:
        fig.add_trace(
            go.Scatter(
                x=df["segmento"],
                y=df[s],
                mode="lines+markers",
                name=s
            )
        )

    fig.add_trace(
        go.Scatter(
            x=df["segmento"],
            y=df["promedio"],
            mode="lines+markers",
            name="Promedio segmento",
            line=dict(color="black", width=5)
        )
    )

    fig.update_layout(
        title=titulo,
        xaxis_title="Segmento",
        yaxis_title="Velocidad corrosión",
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)
    
def resumen_especie(df_result):

    if df_result.empty:
        return pd.DataFrame()

    resumen = (
        df_result.groupby("Cesta")
        .agg(
            num_veces=("Cesta", "count"),
            dias_totales=("Dias", "sum"),
            pct_medio=("% especie", "mean"),
            vel_media=("Velocidad", "mean")
        )
        .sort_values("num_veces", ascending=False)
        .reset_index()
    )

    return resumen

def extraer_segmentos_validos_wrapper(df_filtrado, y_suave, segmentos_raw, df_proc, vars_proceso, min_dias_val):
    fn = safe_get("extraer_segmentos_validos")
    if fn is not None:
        try:
            return fn(df_filtrado, y_suave, segmentos_raw, df_proc, vars_proceso, min_dias=min_dias_val)
        except Exception:
            pass
    return extraer_segmentos_validos_fallback(df_filtrado, y_suave, segmentos_raw, df_proc, vars_proceso, min_dias=min_dias_val)

def buscar_especie_en_cestas(df_cestas, detalle_crudos, especie):

    resultados = []

    for _, cesta in df_cestas.iterrows():

        especies_cesta = cesta["Especies"]

        # comprobar si está la especie
        if especie not in especies_cesta:
            continue

        fi = pd.to_datetime(cesta["Fecha_ini"])
        ff = pd.to_datetime(cesta["Fecha_fin"])

        # datos crudos en ese intervalo
        sub = detalle_crudos[
            (detalle_crudos["Fecha"] >= fi) &
            (detalle_crudos["Fecha"] <= ff)
        ]

        if sub.empty:
            continue

        suma = (
            sub.groupby("Especie")["Porcentaje"]
            .sum()
        )

        total = suma.sum()

        if total == 0:
            continue

        pct = (suma / total * 100)

        pct_especie = pct.get(especie, 0)

        fila = {
            "Cesta": ", ".join(especies_cesta),
            "Fecha_ini": fi,
            "Fecha_fin": ff,
            "Dias": cesta["Dias"],
            "Estado": cesta["Estado"],
            "% especie": pct_especie,
            "Velocidad": cesta["Velocidad"]
        }

        # añadir variables de proceso si existen
        for col in cesta.index:
            if col not in fila:
                fila[col] = cesta[col]

        resultados.append(fila)

    if resultados:
        return pd.DataFrame(resultados)

    return pd.DataFrame()
def dibujar_grafica_completa_wrapper(df_filtrado, y_suave, segmentos_validos, descartados, segmentos_eliminados_idx, titulo, figsize, show=False):
    fn = safe_get("dibujar_grafica_completa")
    if fn is not None:
        try:
            return fn(df_filtrado, y_suave, segmentos_validos, descartados, segmentos_eliminados_idx, titulo=titulo, figsize=figsize, show=show)
        except Exception:
            pass
    return dibujar_grafica_completa_fallback(df_filtrado, y_suave, segmentos_validos, descartados, segmentos_eliminados_idx, titulo=titulo, figsize=figsize, show=show)
def mapear_crudos_a_segmentos(df_master):

    if df_master is None or df_master.empty:
        return {}

    mapa = {}

    for _, row in df_master.iterrows():

        key = (
            row["Sonda"],
            row["Hoja"],
            row["Segmento"]
        )

        texto = f"{row['Crudo']} ({row['Porcentaje_promedio']:.1f}%)"

        if key not in mapa:
            mapa[key] = []

        mapa[key].append(texto)

    # convertir listas a string
    for k in mapa:
        mapa[k] = " | ".join(mapa[k])

    return mapa
def recalcular_segmento_local_wrapper(df_filtrado, y_suave, segmento, df_proc, vars_proceso, nuevo_umbral, nuevo_umbral_factor=None, min_dias=10):
    fn = safe_get("recalcular_segmento_local")
    if fn is not None:
        try:
            return fn(df_filtrado, y_suave, segmento, df_proc, vars_proceso, nuevo_umbral, nuevo_umbral_factor, min_dias=min_dias)
        except Exception:
            pass
    return recalcular_segmento_local_fallback(df_filtrado, y_suave, segmento, df_proc, vars_proceso, nuevo_umbral, nuevo_umbral_factor, min_dias)
import numpy as np

def calcular_regresion(x, y):

    import numpy as np
    import pandas as pd

    # Convertir a Series
    x = pd.Series(x)
    y = pd.Series(y)

    # Forzar numérico
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")

    # Quitar NaN
    mask = (~x.isna()) & (~y.isna())

    x = x[mask].values
    y = y[mask].values

    # Si no hay suficientes datos
    if len(x) < 2:
        return None, None, None

    try:

        coef = np.polyfit(x, y, 1)

        pendiente = coef[0]
        intercepto = coef[1]

        y_pred = pendiente * x + intercepto

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        if ss_tot == 0:
            r2 = None
        else:
            r2 = 1 - (ss_res / ss_tot)

        return x, y_pred, r2

    except Exception:
        return None, None, None

def recalcular_segmento_local_fallback(df_filtrado, y_suave, segmento, df_proc, vars_proceso,
                                       nuevo_umbral, nuevo_umbral_factor=None, min_dias=10,
                                       wl_max=51, wl_min=5):
    ini_g, fin_g = int(segmento.get("ini", 0)), int(segmento.get("fin", 0))
    df_local = df_filtrado.iloc[ini_g:fin_g].reset_index(drop=True)
    if df_local.empty or len(df_local) < 5:
        return [], [{"ini": ini_g, "fin": fin_g, "motivo": "Datos insuficientes local", "estado": "descartado"}]

    if nuevo_umbral_factor is not None:
        n_ref_local = min(10, len(df_local))
        grosor_ref_local = df_local["UT measurement (mm)"].iloc[:n_ref_local].mean()
        mask = df_local["UT measurement (mm)"] <= grosor_ref_local * nuevo_umbral_factor
        df_local = df_local[mask].reset_index(drop=True)
        if df_local.empty or len(df_local) < 5:
            return [], [{"ini": ini_g, "fin": fin_g, "motivo": "Filtro local eliminó casi todo", "estado": "descartado"}]

    y_local = df_local["UT measurement (mm)"].values
    wl = min(wl_max, (len(y_local) - 1) if (len(y_local) % 2 == 0) else len(y_local))
    wl = max(wl_min, wl)
    if wl % 2 == 0:
        wl += 1
    try:
        from scipy.signal import savgol_filter
        y_suave_local = savgol_filter(y_local, wl, 3)
    except Exception:
        y_suave_local = y_local.copy()

    pendiente_local = np.gradient(y_suave_local)
    cambios_local = [0]
    for i in range(1, len(pendiente_local)):
        if abs(pendiente_local[i] - pendiente_local[i - 1]) > nuevo_umbral:
            cambios_local.append(i)
    cambios_local.append(len(y_suave_local) - 1)

    segmentos_raw_local = []
    for k in range(len(cambios_local) - 1):
        a, b = cambios_local[k], cambios_local[k + 1]
        if a < 0 or b <= a or b > len(df_local):
            continue
        fecha_a = pd.to_datetime(df_local["Sent Time"].iloc[a], errors="coerce")
        fecha_b = pd.to_datetime(df_local["Sent Time"].iloc[b - 1], errors="coerce")
        delta_dias = (fecha_b - fecha_a).days if (pd.notna(fecha_a) and pd.notna(fecha_b)) else 0
        velocidad = np.nan
        if delta_dias > 0:
            try:
                velocidad = (y_suave_local[b - 1] - y_suave_local[a]) / (delta_dias / 365.25)
            except Exception:
                velocidad = np.nan
        segmentos_raw_local.append({
            "ini": a, "fin": b,
            "fecha_ini": fecha_a, "fecha_fin": fecha_b,
            "delta_dias": delta_dias, "velocidad": velocidad
        })

    nuevos_validos_global = []
    nuevos_descartados_global = []

    fecha_col = None
    if df_proc is not None and not df_proc.empty:
        for c in df_proc.columns:
            if str(c).strip().lower().startswith("fecha"):
                fecha_col = c
                break
        if fecha_col is None:
            for c in df_proc.columns:
                try:
                    sample = pd.to_datetime(df_proc[c].dropna().iloc[:5], errors="coerce")
                    if sample.notna().any():
                        fecha_col = c
                        break
                except Exception:
                    continue

    for s in segmentos_raw_local:
        if pd.isna(s["fecha_ini"]) or pd.isna(s["fecha_fin"]):
            nuevos_descartados_global.append({
                "ini": ini_g + s.get("ini", 0),
                "fin": ini_g + s.get("fin", 0),
                "motivo": "Fechas inválidas local",
                "estado": "descartado"
            })
            continue
        if s["delta_dias"] <= 0 or s["delta_dias"] < min_dias:
            nuevos_descartados_global.append({
                "ini": ini_g + s.get("ini", 0),
                "fin": ini_g + s.get("fin", 0),
                "motivo": f"Duración < {min_dias} días (local)",
                "estado": "descartado"
            })
            continue
        if s["velocidad"] is None or (not np.isfinite(s["velocidad"])) or s["velocidad"] >= 0:
            nuevos_descartados_global.append({
                "ini": ini_g + s.get("ini", 0),
                "fin": ini_g + s.get("fin", 0),
                "motivo": "Velocidad no negativa o NaN local",
                "estado": "descartado"
            })
            continue

        medias = pd.Series(dtype=float)
        if df_proc is not None and not df_proc.empty and fecha_col is not None:
            try:
                df_proc[fecha_col] = pd.to_datetime(df_proc[fecha_col], errors="coerce")
                sub = df_proc[
                    (df_proc[fecha_col] >= s["fecha_ini"] - pd.Timedelta(days=1))
                    & (df_proc[fecha_col] <= s["fecha_fin"] + pd.Timedelta(days=1))
                ]
                medias = sub.mean(numeric_only=True)
            except Exception:
                medias = pd.Series(dtype=float)

        rd_days = s["delta_dias"]
        anios = rd_days // 365
        meses = (rd_days % 365) // 30
        if anios == 0 and meses == 0 and rd_days > 0:
            meses = 1
        if meses == 12:
            anios += 1
            meses = 0

        nuevos_validos_global.append({
            "ini": ini_g + s["ini"], "fin": ini_g + s["fin"],
            "fecha_ini": s["fecha_ini"], "fecha_fin": s["fecha_fin"],
            "delta_dias": s["delta_dias"], "velocidad": s["velocidad"],
            "vel_abs": abs(s["velocidad"]), "medias": medias,
            "anios": anios, "meses": meses,
            "estado": "valido", "num_segmento_valido": None
        })

    return nuevos_validos_global, nuevos_descartados_global

# -------------------- Session storage --------------------
if "processed_sheets" not in st.session_state:
    st.session_state["processed_sheets"] = {}
    
# =========================================
# CREAR DATASET GLOBAL DE CRUDOS
# =========================================

if uploaded_crudos is not None:

    try:

        hojas_crudos = leer_archivo(uploaded_crudos)

        if hojas_crudos:

            hoja_crudos = list(hojas_crudos.keys())[0]

            df_crudos = hojas_crudos[hoja_crudos]

            detalle_crudos = procesar_crudos(df_crudos)

            df_master = construir_dataset_crudos_segmentos(
                detalle_crudos,
                st.session_state.get("processed_sheets", {})
            )

            if not df_master.empty:

                df_master["Segmento"] = "Seg " + df_master["Segmento"].astype(str)

                st.session_state["df_master_global"] = df_master

    except Exception as e:

        st.warning(f"No se pudo generar dataset de crudos: {e}")
# -------------------- Pestañas UI --------------------
tabs = st.tabs([
    "Procesar hoja",
    "Combinar hojas",
    "Revisión / Guardado",
    "Tabla corregida",
    "Modelo predictivo",
    "Crudos"
])


# -------------------- Cargar y preparar datos de proceso --------------------
df_proc = None
vars_proceso = []

if uploaded_proc is not None:
    cargar_datos_proceso_fn = safe_get("cargar_datos_proceso")
    try:
        # Guardar archivo subido temporalmente
        if hasattr(uploaded_proc, "name"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp_proc:
                tmp_proc.write(uploaded_proc.getbuffer())
                tmp_proc_path = tmp_proc.name
        else:
            tmp_proc_path = uploaded_proc

        if cargar_datos_proceso_fn is not None:
            df_proc, vars_proceso = cargar_datos_proceso_fn(tmp_proc_path)
        else:
            df_proc, vars_proceso = cargar_proceso_primera_hoja_limpio(tmp_proc_path)
            # -------------------------------------------------------
            # 🔧 LIMPIEZA GLOBAL PARA EVITAR EL ERROR DEL EXCEL
            # -------------------------------------------------------
            
            def limpiar_celda(x):
                # Dejar pasar valores normales
                if isinstance(x, (int, float, str)) or pd.isna(x):
                    return x
            
                # Si es lista, tuple, dict, array, objeto extraño → eliminarlo
                try:
                    if hasattr(x, "__iter__") and not isinstance(x, (bytes, str)):
                        return np.nan
                except:
                    pass
            
                return np.nan
            
            # Aplicar limpieza a todas las celdas
            df_proc = df_proc.applymap(limpiar_celda)
            
            # Reemplazar strings vacíos o representaciones de NaN por NaN real
            df_proc = df_proc.replace(
                ["nan", "NaN", "None", "<NA>", "N/A", "NA", "", " "],
                np.nan
            )

        fecha_col = None
        for c in df_proc.columns:
            if any(k in str(c).lower() for k in ["fecha", "date", "time", "sent"]):
                fecha_col = c
                break
        if fecha_col is None:
            fecha_col = df_proc.columns[0]
        if fecha_col != "Fecha":
            df_proc.rename(columns={fecha_col: "Fecha"}, inplace=True)

        df_proc["Fecha"] = pd.to_datetime(df_proc["Fecha"], errors="coerce")
        df_proc = df_proc.dropna(subset=["Fecha"]).sort_values("Fecha").reset_index(drop=True)

        for c in df_proc.columns:
            if c != "Fecha":
                df_proc[c] = pd.to_numeric(df_proc[c], errors="coerce")

        vars_proceso = [c for c in df_proc.columns if c != "Fecha"]

        st.session_state["df_proc"] = df_proc
        st.session_state["vars_proceso"] = vars_proceso

        st.sidebar.success(f"Archivo de proceso cargado: {len(df_proc)} filas, {len(vars_proceso)} variables.")

    except Exception as e:
        st.sidebar.error(f"Error al leer archivo de proceso: {e}")
else:
    st.sidebar.info("Sube un archivo de datos de proceso (.xlsx) para calcular medias.")

# -------------------- TAB 1: Procesar hoja --------------------
with tabs[0]:
    st.header("Procesamiento de hoja")

    if uploaded_corr is None:
        st.info("Sube el archivo de corrosión en la barra lateral para comenzar.")
    else:
        # ============================================================
# BLOQUE ÚNICO Y CORRECTO PARA LEER EL EXCEL DE CORROSIÓN
# ============================================================
        
        import tempfile
        
        corr_path = None
        
        # Crear archivo temporal con el Excel subido
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
                tmp.write(uploaded_corr.getbuffer())
                corr_path = tmp.name
        except Exception as e:
            st.error(f"No se pudo crear archivo temporal: {e}")
            corr_path = None

        # Leer las hojas del archivo
        hojas = []
        if uploaded_corr is not None:
            try:
                hojas_dict = leer_archivo(uploaded_corr)
                hojas = list(hojas_dict.keys())
        
                if not hojas:
                    st.warning("No se encontraron hojas en el archivo subido.")
                else:
                    hoja_sel = st.selectbox("Selecciona hoja", options=hojas, key=f"selectbox_corr_{uploaded_corr.name}")
                    df_original = hojas_dict[hoja_sel]
                    st.success(f"Hoja cargada: {hoja_sel} — filas: {len(df_original)}")
        
            except Exception as e:
                st.error(f"No se pudieron leer las hojas del archivo: {e}")
                hojas = []
        
        if df_original is not None and not df_original.empty:
            st.write("Los parámetros que cambies a continuación recalcularán automáticamente la gráfica y segmentos.")
            col1, col2 = st.columns([3,1])
            with col1:
                st.markdown("**Parámetros activos**")
                st.write(f"umbral_factor = {umbral_factor}, umbral = {umbral}, min_dias = {min_dias_seg}")
            with col2:
                st.markdown("Guardar/Exportar")
                save_auto = st.checkbox("Salvar automáticamente al guardar procesado", value=False, key="chk_save_auto")


            # si tienes archivo de proceso subido, cargarlo (solo sheet 0)
            if uploaded_proc is not None and st.session_state.get("df_proc") is not None:
                df_proc = st.session_state.get("df_proc")
                vars_proceso = st.session_state.get("vars_proceso", [])
                st.sidebar.success("Archivo de proceso listo para usar.")
            else:
                df_proc = None
                vars_proceso = []

            with st.spinner("Procesando y detectando segmentos..."):
                df_filtrado, y_suave, cambios, segmentos_raw = detectar_segmentos_wrapper(
                    df_original, umbral_factor, umbral
                )
                if df_filtrado is None or y_suave is None:
                    st.error("No se pudieron detectar segmentos. Revisa las columnas (fecha/espesor) o ajusta umbrales.")
                else:
                    df_proc = st.session_state.get("df_proc", None)
                    vars_proceso = st.session_state.get("vars_proceso", [])
                    segmentos_validos, descartados = extraer_segmentos_validos_wrapper(
                        df_filtrado, y_suave, segmentos_raw, df_proc, vars_proceso, min_dias_seg
                    )
                    key = f"proc|{uploaded_corr.name}|{hoja_sel}"
                    # =========================
                    # DIVISIÓN GLOBAL
                    # =========================
                    
                    if btn_dividir_global and key in st.session_state["processed_sheets"]:
                    
                        data = st.session_state["processed_sheets"][key]
                    
                        if tipo_intervalo_global == "Días":
                            offset = pd.DateOffset(days=valor_intervalo_global)
                        elif tipo_intervalo_global == "Meses":
                            offset = pd.DateOffset(months=valor_intervalo_global)
                        else:
                            offset = pd.DateOffset(years=valor_intervalo_global)
                    
                        nuevos = dividir_todos_segmentos(
                            data["df_filtrado"],
                            data["segmentos_validos"],
                            df_proc,
                            vars_proceso,
                            offset,
                            min_dias=min_dias_seg
                        )
                    
                        nuevos = sorted(nuevos, key=lambda x: x["fecha_ini"])
                    
                        # historial
                        if "historial_segmentos" not in data:
                            data["historial_segmentos"] = []
                    
                        data["historial_segmentos"].append(
                            data["segmentos_validos"].copy()
                        )
                    
                        st.session_state["processed_sheets"][key]["segmentos_validos"] = nuevos
                        st.session_state["processed_sheets"][key]["manually_modified"] = True
                    
                        st.success("División global aplicada")
                        st.rerun()

                    if key not in st.session_state["processed_sheets"]:

                        st.session_state["processed_sheets"][key] = {
                            "df_original": df_original,
                            "df_filtrado": df_filtrado,
                            "y_suave": y_suave,
                            "segmentos_validos": segmentos_validos,
                            "descartados": descartados,
                            "hoja": hoja_sel,
                            "source_name": uploaded_corr.name,
                            "saved": False,
                            "manually_modified": False
                        }
                    else:
                        existing = st.session_state["processed_sheets"][key]
                        existing.update({
                            "df_original": df_original,
                            "df_filtrado": df_filtrado,
                            "y_suave": y_suave,
                            "hoja": hoja_sel,
                            "source_name": uploaded_corr.name
                        })
                        if not existing.get("manually_modified", False):
                            existing["segmentos_validos"] = segmentos_validos
                            existing["descartados"] = descartados
                        st.session_state["processed_sheets"][key] = existing

                    try:
                        stored = st.session_state["processed_sheets"][key]
                        fig, ax = dibujar_grafica_completa_wrapper(
                            stored["df_filtrado"], stored["y_suave"],
                            stored["segmentos_validos"], stored["descartados"], [],
                            titulo=f"Segmentación — {hoja_sel}", figsize=(fig_w, fig_h), show=False
                        )

                        st.pyplot(fig)
                    except Exception as e:
                        st.error(f"Error dibujando gráfica: {e}")

                    st.markdown("### Editar segmentos (eliminar / recalcular)")
                    # ===============================
                    # RECUPERAR SEGMENTOS DESCARTADOS
                    # ===============================
                    
                    desc_list = []
                    descartados = st.session_state["processed_sheets"][key]["descartados"]
                    
                    if descartados:
                    
                        st.markdown("### Recuperar segmentos descartados")
                    
                        desc_list = [
                            f"{i+1}: {d.get('fecha_ini')} → {d.get('fecha_fin')} | Motivo: {d.get('motivo')}"
                            for i,d in enumerate(descartados)
                        ]
                    
                        sel_desc = st.selectbox(
                            "Selecciona segmento descartado",
                            options=list(range(1, len(desc_list)+1)),
                            format_func=lambda x: desc_list[x-1],
                            key=f"sel_desc_{key}"
                        )
                    
                        if st.button("Recuperar segmento descartado", key=f"recuperar_{key}"):
                    
                            idx0 = sel_desc - 1
                    
                            if 0 <= idx0 < len(descartados):
                    
                                seg = st.session_state["processed_sheets"][key]["descartados"].pop(idx0)
                    
                                seg["estado"] = "valido"
                                seg["vel_abs"] = abs(seg.get("velocidad", 0))
                    
                                st.session_state["processed_sheets"][key]["segmentos_validos"].append(seg)
                    
                                st.session_state["processed_sheets"][key]["segmentos_validos"] = sorted(
                                    st.session_state["processed_sheets"][key]["segmentos_validos"],
                                    key=lambda x: x.get("fecha_ini")
                                )
                    
                                st.session_state["processed_sheets"][key]["manually_modified"] = True
                    
                                st.success("Segmento recuperado")
                                st.rerun()

                    seg_list = []
                    try:
                        seg_list = [f"{i+1}: {s.get('fecha_ini')} → {s.get('fecha_fin')}  | Vel: {s.get('vel_abs')}" for i,s in enumerate(st.session_state["processed_sheets"][key]["segmentos_validos"])]
                    except Exception:
                        seg_list = []

                    if seg_list:
                        sel_idx = st.selectbox("Selecciona segmento", options=list(range(1, len(seg_list)+1)), format_func=lambda x: seg_list[x-1], key=f"selseg_{key}")
                        # ===============================
                        # APLICAR SEGMENTACIÓN REFERENCIA
                        # ===============================
                        
                        if "segmentacion_referencia" in st.session_state:
                        
                            if st.button("Aplicar segmentación de referencia"):
                        
                                ref = st.session_state["segmentacion_referencia"]
                        
                                nuevos = aplicar_segmentacion_referencia(
                                    st.session_state["processed_sheets"][key]["df_filtrado"],
                                    ref,
                                    df_proc,
                                    vars_proceso,
                                    segmentos_validos_previos=st.session_state["processed_sheets"][key]["segmentos_validos"],
                                    segmentos_descartados_previos=st.session_state["processed_sheets"][key]["descartados"],
                                    min_dias=min_dias_seg
                                )
                                st.session_state["processed_sheets"][key]["segmentos_validos"] = nuevos
                                st.session_state["processed_sheets"][key]["manually_modified"] = True
                        
                                st.success("Segmentación de referencia aplicada.")
                                st.rerun()

                        colA, colB, colC, colD = st.columns(4)

                        with colA:
                            if st.button("Eliminar segmento (sesión)", key=f"del_{key}"):
                                idx0 = sel_idx - 1
                                segmentos = st.session_state["processed_sheets"][key]["segmentos_validos"]
                                if 0 <= idx0 < len(segmentos):
                                    s = segmentos.pop(idx0)
                                    st.session_state["processed_sheets"][key]["descartados"].append({
                                        "ini": s.get('ini'),
                                        "fin": s.get('fin'),
                                        "motivo": "eliminado_manual",
                                        "estado": "descartado"
                                    })
                                    st.session_state["processed_sheets"][key]["manually_modified"] = True
                                    st.success("✅ Segmento eliminado de la sesión.")
                                    st.rerun()
                                else:
                                    st.error("Índice de segmento no válido.")

                        with colB:
                            st.markdown("**Recalcular segmento local**")
                            new_umbral_local = st.number_input(
                                "Nuevo umbral local",
                                min_value=1e-12,
                                value=float(umbral),
                                step=0.0001,
                                format="%.6f",
                                key=f"umbral_local_{key}"
                            )
                            new_umbral_factor_local = st.number_input(
                                "Nuevo umbral_factor local",
                                min_value=1.0,
                                max_value=2.0,
                                value=float(umbral_factor),
                                step=0.0001,
                                format="%.4f",
                                key=f"umbral_factor_local_{key}"
                            )

                            if st.button("Recalcular segmento", key=f"recalc_{key}"):
                                idx0 = sel_idx - 1
                                segmentos = st.session_state["processed_sheets"][key]["segmentos_validos"]
                                if 0 <= idx0 < len(segmentos):
                                    seg = segmentos[idx0]
                                    try:
                                        nuevos_validos, nuevos_descartados = recalcular_segmento_local_wrapper(
                                            st.session_state["processed_sheets"][key]["df_filtrado"],
                                            st.session_state["processed_sheets"][key]["y_suave"],
                                            seg, df_proc, vars_proceso, new_umbral_local, new_umbral_factor_local, min_dias=min_dias_seg
                                        )
                                        st.session_state["processed_sheets"][key]["manually_modified"] = True
                                        try:
                                            st.session_state["processed_sheets"][key]["segmentos_validos"].pop(idx0)
                                        except Exception:
                                            pass
                                        for nd in nuevos_descartados:
                                            st.session_state["processed_sheets"][key]["descartados"].append(nd)
                                        for nv in nuevos_validos:
                                            st.session_state["processed_sheets"][key]["segmentos_validos"].append(nv)
                                        st.session_state["processed_sheets"][key]["segmentos_validos"] = sorted(
                                            st.session_state["processed_sheets"][key]["segmentos_validos"],
                                            key=lambda x: x.get("fecha_ini") or pd.Timestamp.max
                                        )
                                        st.success(f"Recalculado: añadidos {len(nuevos_validos)} segmentos (si los hubo). Actualizando vista...")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Error recalculando: {e}")
                                else:
                                    st.error("Índice de segmento no válido.")

                        with colC:
                            if st.button("Guardar procesado (pickle + imagen)", key=f"save_{key}"):
                                data_now = st.session_state["processed_sheets"][key]
                                out_dir = Path.cwd() / "procesados_finales"
                                out_dir.mkdir(exist_ok=True)
                                
                                import re
                                
                                def make_safe_name(text: str) -> str:
                                    text = text.strip()
                                    text = re.sub(r'[\/\\:\*\?"<>\|]+', '', text)  # quita caracteres no válidos
                                    text = text.replace(' ', '_')
                                    return text
                                
                                # Nombres seguros
                                safe_source = make_safe_name(data_now['source_name'])
                                safe_sheet = make_safe_name(data_now['hoja'])
                                folder_name = f"{safe_source}_{safe_sheet}"
                                
                                # Carpeta específica
                                out_dir = Path.cwd() / "procesados_finales" / folder_name
                                out_dir.mkdir(parents=True, exist_ok=True)
                                
                                # Rutas finales
                                pkl_path = out_dir / f"{folder_name}_procesado.pkl"
                                figpath = out_dir / f"{folder_name}_grafica.png"
                                
                                try:
                                    datos_guardar = {
                                        "df_filtrado": data_now['df_filtrado'],
                                        "y_suave": data_now['y_suave'],
                                        "segmentos_validos": data_now['segmentos_validos'],
                                        "descartados": data_now['descartados'],
                                        "segmentos_eliminados_idx": []
                                    }
                                    with open(pkl_path, "wb") as f:
                                        pickle.dump(datos_guardar, f)

                                    try:
                                        fig_save, ax_save = dibujar_grafica_completa_wrapper(
                                            data_now['df_filtrado'], data_now['y_suave'],
                                            data_now['segmentos_validos'], data_now['descartados'], [],
                                            titulo=f"{data_now['hoja']}", figsize=(fig_w, fig_h), show=False
                                        )
                                        fig_save.savefig(figpath, dpi=200, bbox_inches="tight")
                                        plt.close(fig_save)
                                    except Exception:
                                        pass
                                    st.session_state["processed_sheets"][key]["saved"] = True
                                    st.success(f"Procesado guardado: {pkl_path}. Actualizando vista...")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"No se pudo guardar: {e}")
                                    
                        with colD:
                            st.markdown("**Dividir segmento por intervalo**")
                        
                            tipo_intervalo = st.selectbox(
                                "Tipo intervalo",
                                ["Días", "Meses", "Años"],
                                key=f"tipo_intervalo_{key}"
                            )
                        
                            valor_intervalo = st.number_input(
                                "Cantidad",
                                min_value=1,
                                value=15,
                                key=f"valor_intervalo_{key}"
                            )
                        
                            if st.button("Dividir segmento", key=f"dividir_{key}"):
                        
                                idx0 = sel_idx - 1
                                segmentos = st.session_state["processed_sheets"][key]["segmentos_validos"]
                        
                                if 0 <= idx0 < len(segmentos):
                        
                                    seg = segmentos[idx0]
                        
                                    # Crear offset
                                    if tipo_intervalo == "Días":
                                        offset = pd.DateOffset(days=valor_intervalo)
                                    elif tipo_intervalo == "Meses":
                                        offset = pd.DateOffset(months=valor_intervalo)
                                    else:
                                        offset = pd.DateOffset(years=valor_intervalo)
                        
                                    nuevos_segmentos = dividir_segmento_por_intervalo(
                                        st.session_state["processed_sheets"][key]["df_filtrado"],
                                        seg,
                                        df_proc,
                                        vars_proceso,
                                        offset,
                                        min_dias=min_dias_seg
                                    )
                            
                                    # Guardar backup para poder deshacer
                                    if "historial_segmentos" not in st.session_state["processed_sheets"][key]:
                                        st.session_state["processed_sheets"][key]["historial_segmentos"] = []
                                    
                                    st.session_state["processed_sheets"][key]["historial_segmentos"].append(
                                        st.session_state["processed_sheets"][key]["segmentos_validos"].copy()
                                    )

                                    # eliminar segmento original
                                    segmentos.pop(idx0)
                        
                                    # añadir nuevos
                                    for n in nuevos_segmentos:
                                        segmentos.append(n)
                        
                                    st.session_state["processed_sheets"][key]["segmentos_validos"] = sorted(
                                        segmentos,
                                        key=lambda x: x.get("fecha_ini")
                                    )
                        
                                    st.session_state["processed_sheets"][key]["manually_modified"] = True
                        
                                    st.success("Segmento dividido correctamente")
                                    st.rerun()
                            # =========================
                            # BOTÓN DESHACER DIVISIÓN
                            # =========================
                            if st.button("↩️ Deshacer última división", key=f"undo_div_{key}"):
                            
                                hist = st.session_state["processed_sheets"][key].get("historial_segmentos", [])
                            
                                if hist:
                                    st.session_state["processed_sheets"][key]["segmentos_validos"] = hist.pop()
                                    st.success("División deshecha")
                                    st.rerun()
                                else:
                                    st.warning("No hay historial para deshacer")


# -------------------- TAB 2: Combinar hojas --------------------
with tabs[1]:
    st.header("Combinar hojas (curvas desplazadas y selección por intervalo)")
    saved_keys = [k for k,v in st.session_state.get("processed_sheets", {}).items() if v.get("saved")]
    if not saved_keys:
        st.info("No hay procesados guardados en sesión. Guarda desde la pestaña 'Procesar hoja'.")
    else:
        choices = {k: f"{v['source_name']} | {v['hoja']}" for k,v in st.session_state['processed_sheets'].items() if v.get('saved')}
        sel = st.multiselect("Selecciona hojas guardadas para combinar", options=list(choices.keys()), format_func=lambda x: choices[x], default=list(choices.keys()))
        if sel:
            offsets = {}
            current_offset = 0.0
            downsample_threshold = 5000
            for k in sel:
                d = st.session_state['processed_sheets'][k]
                y = np.asarray(d['y_suave'])
                ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))
                rango = ymax - ymin if (ymax - ymin) != 0 else 0.1
                gap = max(0.6, rango * 1.1)
                offsets[k] = current_offset
                current_offset += gap
            import plotly.graph_objects as go
            fig = go.Figure()
            for k in sel:
                d = st.session_state['processed_sheets'][k]
                df_f = d['df_filtrado']
                y = np.asarray(d['y_suave'])
                off = offsets[k]
                x = pd.to_datetime(df_f['Sent Time'])
                yoff = y + off
                if len(x) > downsample_threshold:
                    idxs = np.linspace(0, len(x)-1, downsample_threshold, dtype=int)
                    x_plot = x.iloc[idxs]
                    y_plot = yoff[idxs]
                else:
                    x_plot = x
                    y_plot = yoff
                fig.add_trace(go.Scatter(x=x_plot, y=y_plot, mode='lines', name=f"{d['hoja']}"))
                for s in d['segmentos_validos']:
                    if s.get('estado','valido') != 'valido': continue
                    i, f = int(s['ini']), int(s['fin'])
                    xs = pd.to_datetime(df_f['Sent Time'].iloc[i:f])
                    ys = np.asarray(d['y_suave'])[i:f] + off
                    if len(xs) > 1:
                        fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines', line=dict(width=6), name=f"{d['hoja']} seg", showlegend=False, opacity=0.5))
            fig.update_layout(height=600, title="Curvas combinadas (desplazadas)")
            all_dates = []
            for k in sel:
                df_f = st.session_state['processed_sheets'][k]['df_filtrado']
                all_dates.extend(pd.to_datetime(df_f['Sent Time']).tolist())
            all_dates = sorted(set(all_dates))
            if all_dates:
                min_date, max_date = min(all_dates), max(all_dates)
                date_range = st.slider("Intervalo (fecha)", min_value=min_date.date(), max_value=max_date.date(), value=(min_date.date(), max_date.date()), key="slider_date_range_comb")
                fi = pd.to_datetime(date_range[0])
                ff = pd.to_datetime(date_range[1])
                fig.add_vrect(x0=fi, x1=ff, fillcolor="LightSalmon", opacity=0.3, layer="below", line_width=0)
                st.plotly_chart(fig, use_container_width=True)
                # ======================================
                # CONSTRUIR TABLA DE SEGMENTOS ENTRE SONDAS
                # ======================================
                
                segmentos_rows = []
                
                for k in sel:
                    d = st.session_state['processed_sheets'][k]
                
                    for i, s in enumerate(d['segmentos_validos']):
                        if s.get('estado','valido') != 'valido':
                            continue
                
                        segmentos_rows.append({
                            "sonda": d['hoja'],
                            "segmento": i,
                            "vel": s.get("vel_abs")
                        })
                
                df_seg = pd.DataFrame(segmentos_rows)
                
                if not df_seg.empty:
                
                    df_comp = df_seg.pivot_table(
                        index="segmento",
                        columns="sonda",
                        values="vel"
                    ).reset_index()
                
                    sondas = [c for c in df_comp.columns if c != "segmento"]
                
                    df_comp["promedio"] = df_comp[sondas].mean(axis=1)
                
                    df_comp["std"] = df_comp[sondas].std(axis=1)
                
                    df_comp["CV (%)"] = df_comp["std"] / df_comp["promedio"] * 100
                    # ======================================
                    # FILTRAR SEGÚN UMBRAL EXISTENTE
                    # ======================================
                    
                    df_validos = df_comp[
                        df_comp["CV (%)"].isna() | (df_comp["CV (%)"] <= umbral_error_segmento)
                    ]
                    
                    df_invalidos = df_comp[
                        df_comp["CV (%)"] > umbral_error_segmento
                    ]
                st.subheader("Segmentos válidos")
                
                graficar_segmentos(
                    df_validos,
                    f"Segmentos válidos (CV ≤ {umbral_error_segmento}%)"
                )
                
                st.subheader("Segmentos no válidos")
                
                graficar_segmentos(
                    df_invalidos,
                    f"Segmentos no válidos (CV > {umbral_error_segmento}%)"
                )
                if st.button("Extraer segmentos en intervalo seleccionado"):
                    extracted = []
                    for k in sel:
                        d = st.session_state['processed_sheets'][k]
                        for s in d['segmentos_validos']:
                            s_fi = pd.to_datetime(s.get('fecha_ini'))
                            s_ff = pd.to_datetime(s.get('fecha_fin'))
                            if not (s_ff < fi or s_fi > ff):
                                row = {
                                    'origen': f"{d['source_name']}|{d['hoja']}",
                                    'segmento_ini': s_fi,
                                    'segmento_fin': s_ff,
                                    'vel_mm_yr': s.get('vel_abs')
                                }
                                medias = s.get('medias')
                                if medias is not None and isinstance(medias, (pd.Series, dict)):
                                    try:
                                        for var, val in (medias.items() if isinstance(medias, dict) else medias.items()):
                                            row[var] = val
                                    except Exception:
                                        pass
                                extracted.append(row)
                    if extracted:
                        df_ex = pd.DataFrame(extracted)
                        st.write(f"Segmentos extraídos: {len(df_ex)}")
                        st.dataframe(df_ex)
                        buf = io.BytesIO()
                        df_ex.to_excel(buf, index=False, engine='openpyxl')
                        buf.seek(0)
                        st.download_button(
                            "Descargar segmentos extraídos (Excel)",
                            data=buf,
                            file_name=f"segmentos_extraidos_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                        )
                    else:
                        st.info("No se encontraron segmentos que se solapen con el intervalo seleccionado.")

# -------------------- TAB 3: Revisión / Guardado --------------------
with tabs[2]:
    st.header("Revisión y guardado")

    # ======================================
    # TABLA COMPARATIVA ENTRE SONDAS
    # ======================================
    st.caption("Si una sonda no tiene un intervalo, aparecerá vacío.")
    st.markdown("## Comparativa entre sondas guardadas")
    material_sel = st.radio(
        "Material velocidad esperada",
        ["Carbon Steel", "5 Cr"],
        horizontal=True
    )

    st.markdown("## Comparativa global por segmento")

    df_comp = construir_tabla_segmentos_comparativa(
        st.session_state.get("processed_sheets", {}),
        st.session_state.get("df_mpa"),
        material_sel
    )
    # ======================================
    # ANALISIS IMPORTANCIA VARIABLES PROCESO
    # ======================================
    
    st.markdown("## 📊 Análisis matemático — Variables que más influyen en la corrosión")
    
    df_vars = df_comp.copy()
    
    if not df_vars.empty:
    
        # renombrar media velocidad como variable objetivo
        if "Media velocidades" in df_vars.columns:
            df_vars["Velocidad_corr"] = df_vars["Media velocidades"]
    
        df_importancia = analizar_importancia_proceso(
            df_vars,
            st.session_state.get("vars_proceso", [])
        )

    
        if not df_importancia.empty:

            st.dataframe(df_importancia)
        
            var_top = df_importancia.iloc[0]["Variable proceso"]
        
            st.success(
                f"La variable de proceso más relacionada con la velocidad de corrosión es: **{var_top}**"
            )

    
        else:
            st.info("No hay suficientes datos para análisis multivariable.")


    if df_comp is None or df_comp.empty:
        st.info("No hay sondas guardadas aún.")
    else:
        cols_calidad = [col for col in df_comp.columns if "Calidad" in col]
    
        if cols_calidad:
            st.dataframe(
                df_comp.style.map(color_calidad, subset=cols_calidad)
            )
        else:
            st.dataframe(df_comp)

        buffer = io.BytesIO()
        df_comp.to_excel(buffer, index=False)
        buffer.seek(0)
    
        st.download_button(
            "Descargar comparativa completa",
            buffer,
            file_name="comparativa_global_segmentos.xlsx"
        )

    saved_list = [k for k,v in st.session_state.get("processed_sheets", {}).items() if v.get("saved")]
    if not saved_list:
        st.info("No hay procesados guardados en sesión.")
    else:
        choice = st.selectbox("Selecciona procesado guardado", options=saved_list, format_func=lambda x: f"{st.session_state['processed_sheets'][x]['source_name']} | {st.session_state['processed_sheets'][x]['hoja']}")
        data = st.session_state['processed_sheets'][choice]
        st.subheader(f"{data['source_name']} | {data['hoja']}")
        # ===============================
        # MARCAR COMO AJUSTE DE REFERENCIA
        # ===============================
        
        if st.button("⭐ Usar este ajuste como referencia"):
        
            st.session_state["segmentacion_referencia"] = data["segmentos_validos"]
        
            st.success("Este ajuste ahora es la referencia temporal para otras sondas.")
        # =====================================
        # APLICAR REFERENCIA A TODAS LAS SONDAS
        # =====================================
        
        if "segmentacion_referencia" in st.session_state:
        
            if st.button("Aplicar segmentación referencia a TODAS las sondas"):
        
                ref = st.session_state["segmentacion_referencia"]
        
                for key, data in st.session_state["processed_sheets"].items():
        
                    df_filtrado = data["df_filtrado"]
        
                    nuevos_segmentos = aplicar_segmentacion_referencia(
                        df_filtrado,
                        ref,
                        st.session_state.get("df_proc"),
                        st.session_state.get("vars_proceso"),
                        min_dias=st.session_state.get("min_dias_seg", 5)
                    )
        
                    if nuevos_segmentos:
        
                        st.session_state["processed_sheets"][key]["segmentos_validos"] = nuevos_segmentos
                        st.session_state["processed_sheets"][key]["manually_modified"] = True
        
                st.success("Referencia aplicada a todas las sondas")
                st.rerun()

        img_dir = Path.cwd() / "graficos_exportados"
        img_file = img_dir / f"{data['source_name']}_{data['hoja']}_grafica.png"
        col1, col2 = st.columns([2,1])
        with col1:
            if img_file.exists():
                st.image(str(img_file), caption="Gráfica guardada (definitiva)")
            else:
                try:
                    fig, ax = dibujar_grafica_completa_wrapper(
                        data['df_filtrado'],
                        data['y_suave'],
                        data['segmentos_validos'],
                        data['descartados'],
                        [],
                        titulo=f"{data['hoja']}",
                        figsize=(fig_w, fig_h),
                        show=False
                    )
                    # Quitar leyenda solo en esta pestaña
                    if ax.get_legend() is not None:
                        ax.get_legend().remove()
                    df_teo = calcular_perfil_teorico_por_segmentos(
                        data['df_filtrado'],
                        data['segmentos_validos'],
                        st.session_state.get("df_mpa"),
                        material_sel
                    )
                    
                    if df_teo is not None:
                    
                        ax.plot(
                            df_teo["Fecha"],
                            df_teo["Espesor_teorico"],
                            linestyle="--",
                            linewidth=2,
                            label="Perfil teórico MPA"
                        )
                    
                     # Quitar leyenda solo en esta pestaña
                    if ax.get_legend() is not None:
                        ax.get_legend().remove()

                    
                    st.pyplot(fig)

                except Exception as e:
                    st.error(f"No se pudo mostrar gráfica: {e}")
        with col2:
            st.markdown("### Resumen y acciones")
            st.write(f"Segmentos válidos: {len(data['segmentos_validos'])} — Descartados: {len(data['descartados'])}")
            try:
                try:
                    try:
                        ut_vals = data['df_filtrado']['UT measurement (mm)']
                    
                        # 🔹 Pérdida REAL
                        perdida_real = ut_vals.iloc[0] - ut_vals.iloc[-1]
                    
                        st.metric(
                            label="Pérdida total REAL (mm)",
                            value=f"{perdida_real:.4f}"
                        )
                    
                        # 🔹 Pérdida TEÓRICA
                        df_teo = calcular_perfil_teorico_por_segmentos(
                            data['df_filtrado'],
                            data['segmentos_validos'],
                            st.session_state.get("df_mpa"),
                            material_sel
                        )
                    
                        if df_teo is not None:
                    
                            perdida_teorica = (
                                df_teo["Espesor_teorico"].iloc[0]
                                - df_teo["Espesor_teorico"].iloc[-1]
                            )
                    
                            st.metric(
                                label="Pérdida total TEÓRICA MPA (mm)",
                                value=f"{perdida_teorica:.4f}"
                            )
                    
                            # 🔹 Diferencia modelo
                            diferencia = perdida_real - perdida_teorica
                    
                            st.metric(
                                label="Diferencia REAL - TEÓRICA (mm)",
                                value=f"{diferencia:.4f}"
                            )
                    
                    except Exception:
                        st.write("No se pudo calcular pérdidas de grosor")

                
                    st.metric(
                        label="Pérdida total de grosor (mm)",
                        value=f"{perdida_grosor:.4f}"
                    )
                
                except Exception:
                    st.write("No se pudo calcular pérdida de grosor")

            except Exception:
                st.write("No se pudo calcular la media UT (datos faltantes).")

            if st.button("Exportar media y resumen a Excel"):
                rows = []
                for idx,s in enumerate(data['segmentos_validos'], start=1):
                    row = {'Segmento': idx, 'Inicio': s.get('fecha_ini'), 'Fin': s.get('fecha_fin'), 'Días': s.get('delta_dias'), 'Vel (mm/año)': s.get('vel_abs')}
                    medias = s.get('medias')
                    if medias is not None and isinstance(medias, (pd.Series, dict)):
                        try:
                            for var, val in (medias.items() if isinstance(medias, dict) else medias.items()):
                                row[var] = val
                        except Exception:
                            pass
                    rows.append(row)
                df_rows = pd.DataFrame(rows)
                df_summary = pd.DataFrame([{'Media UT (mm)': mean_ut, 'Hoja': data['hoja']}])
                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine='openpyxl') as writer:
                    df_summary.to_excel(writer, sheet_name='Resumen', index=False)
                    df_rows.to_excel(writer, sheet_name='Segmentos', index=False)
                buf.seek(0)
                st.download_button("Descargar Excel (media + segmentos)", data=buf, file_name=f"media_segmentos_{data['hoja']}.xlsx")
            st.markdown("### Exportar configuración completa")

            if st.button("Descargar configuración JSON"):
            
                json_data = exportar_configuracion_json()
            
                st.download_button(
                    "Descargar JSON configuración",
                    json_data,
                    file_name="configuracion_analisis.json",
                    mime="application/json"
                )
            uploaded_config = st.sidebar.file_uploader(
                "Cargar configuración JSON",
                type=["json"]
            )
            
            if uploaded_config is not None:
                importar_configuracion_json(uploaded_config)
                st.sidebar.success("Configuración cargada")

            if st.button("Borrar procesado seleccionado (sesión + archivos)"):
                pkl_path = Path.cwd() / "procesados_finales" / f"{data['source_name']}_{data['hoja']}_procesado.pkl"
                figpath = Path.cwd() / "graficos_exportados" / f"{data['source_name']}_{data['hoja']}_grafica.png"
                removed = []
                for f in [pkl_path, figpath]:
                    try:
                        if f.exists():
                            f.unlink()
                            removed.append(str(f))
                    except Exception:
                        pass
                st.session_state['processed_sheets'].pop(choice, None)
                st.success(f"Procesado eliminado. Archivos borrados: {len(removed)} (si existían).")
                st.rerun()
                
            if safe_get("guardar_resultados") is not None and st.button("Ejecutar guardar_resultados del script original"):
                try:
                    guardar_fn = safe_get("guardar_resultados")
                    guardar_fn(data['segmentos_validos'], data['df_filtrado'], data['y_suave'], data['descartados'], [], pd.DataFrame(), [], data['hoja'])
                    st.success("guardar_resultados ejecutado desde el script original (revisa carpeta de salida).")
                except Exception as e:
                    st.error(f"Error ejecutando guardar_resultados: {e}")

st.markdown("---")
st.subheader("Exportación masiva")

if st.button("📦 Exportar TODOS los ajustes (gráficas + excels + collages)"):

    from PIL import Image, ImageDraw
    import zipfile
    import math

    export_dir = Path.cwd() / "export_todo"
    export_dir.mkdir(exist_ok=True)

    zip_path = export_dir / "export_completo.zip"
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w") as z:

        saved_items = {k: v for k, v in st.session_state["processed_sheets"].items() if v.get("saved")}
        for key, data in saved_items.items():

            safe_source = make_safe_name(data["source_name"])
            safe_sheet = make_safe_name(data["hoja"])
            nombre_base = f"{safe_source}_{safe_sheet}"
            
            carpeta = export_dir / nombre_base
            carpeta.mkdir(parents=True, exist_ok=True)

            # ==========================
            # 1) Exportar GRÁFICA GLOBAL
            # ==========================
            fig, ax = dibujar_grafica_completa_wrapper(
                data['df_filtrado'], data['y_suave'],
                data['segmentos_validos'], data['descartados'], [],
                titulo=f"{data['hoja']}", figsize=(14,10)
            )
            

            img_global_path = carpeta / f"{nombre_base}_grafica.png"
            fig.savefig(img_global_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            z.write(img_global_path, arcname=f"{nombre_base}/{img_global_path.name}")

            # ==========================================
            # 2) Excel por hoja con segmentos y variables
            # ==========================================
            rows = []
            for idx,s in enumerate(data['segmentos_validos'], start=1):
                row = {
                    'Segmento': idx,
                    'Inicio': s.get('fecha_ini'),
                    'Fin': s.get('fecha_fin'),
                    'Días': s.get('delta_dias'),
                    'Vel (mm/año)': s.get('vel_abs')
                }
                medias = s.get('medias')
                if medias is not None and isinstance(medias, (pd.Series, dict)):
                    for var,val in (medias.items() if isinstance(medias, dict) else medias.items()):
                        row[var] = val
                rows.append(row)

            df_x = pd.DataFrame(rows)
            excel_path = carpeta / f"{nombre_base}_segmentos.xlsx"
            df_x.to_excel(excel_path, index=False)
            z.write(excel_path, arcname=f"{nombre_base}/{excel_path.name}")

            # ================================
            # 3) Collage de segmentos por hoja
            # ================================
            imagenes_segmentos = []
            for idx,s in enumerate(data['segmentos_validos'], start=1):
                i, f = int(s["ini"]), int(s["fin"])
                fig_seg, ax_seg = plt.subplots(figsize=(6,4))
                x = pd.to_datetime(data['df_filtrado']["Sent Time"].iloc[i:f])
                y = data['y_suave'][i:f]
                ax_seg.plot(x, y)
                ax_seg.set_title(f"Segmento {idx} – {s['vel_abs']:.4f} mm/año")
                ax_seg.tick_params(axis='x', rotation=90)
                seg_path = carpeta / f"seg_{idx}.png"
                fig_seg.savefig(seg_path, dpi=150, bbox_inches="tight")
                plt.close(fig_seg)

                try:
                    imagenes_segmentos.append(Image.open(seg_path))
                except:
                    pass

            if imagenes_segmentos:
                cols = 2
                filas = math.ceil(len(imagenes_segmentos) / cols)
                w, h = imagenes_segmentos[0].size
                collage = Image.new("RGB", (cols*w, filas*h), "white")

                for n,img in enumerate(imagenes_segmentos):
                    fila = n // cols
                    col = n % cols
                    collage.paste(img, (col*w, fila*h))

                collage_path = carpeta / f"{nombre_base}_collage.png"
                collage.save(collage_path)
                z.write(collage_path, arcname=f"{nombre_base}/{collage_path.name}")
                
            
            # ==========================================
            # 4) Gráficas de variables de proceso vs velocidad
            # ==========================================
            df_medias = pd.DataFrame([
                {"Segmento": i+1, "Velocidad (mm/año)": s.get("vel_abs"), **(s.get("medias", {}))}
                for i, s in enumerate(data["segmentos_validos"]) if s.get("estado") == "valido"
            ])
            
            if not df_medias.empty:
                columnas_vars = [c for c in df_medias.columns if c not in ["Segmento", "Velocidad (mm/año)"]]
            
                # ✅ Crear subcarpeta 'variables'
                carpeta_variables = carpeta / "variables"
                carpeta_variables.mkdir(exist_ok=True)
            
                imagenes_proceso = []
            
                for var in columnas_vars:
                    fig_proc, ax_proc = plt.subplots(figsize=(6, 4))
                    ax_proc.scatter(df_medias["Velocidad (mm/año)"], df_medias[var], alpha=0.7)
                    ax_proc.set_xlabel("Velocidad de corrosión (mm/año)")
                    ax_proc.set_ylabel(var)
                    ax_proc.grid(True, alpha=0.4)
                    ax_proc.set_title(f"{var} vs Velocidad")
                    
                    proc_path = carpeta_variables / f"{var}_vs_velocidad.png"
                    fig_proc.savefig(proc_path, dpi=150, bbox_inches="tight")
                    plt.close(fig_proc)
            
                    try:
                        imagenes_proceso.append(Image.open(proc_path))
                    except:
                        pass
            
                    # Añadir al ZIP con la ruta dentro de la carpeta 'variables'
                    z.write(proc_path, arcname=f"{nombre_base}/variables/{proc_path.name}")
            
                # ✅ Collage de todas las gráficas de variables
                if imagenes_proceso:
                    cols = 2
                    filas = math.ceil(len(imagenes_proceso) / cols)
                    w, h = imagenes_proceso[0].size
                    collage_proc = Image.new("RGB", (cols*w, filas*h), "white")
            
                    for n, img in enumerate(imagenes_proceso):
                        fila = n // cols
                        col = n % cols
                        collage_proc.paste(img, (col*w, fila*h))
            
                    collage_proc_path = carpeta / f"{nombre_base}_collage_variables.png"
                    collage_proc.save(collage_proc_path)
                    z.write(collage_proc_path, arcname=f"{nombre_base}/{collage_proc_path.name}")


            # ==========================================
            # 5) Collage de todas las gráficas de proceso vs velocidad
            # ==========================================
            imagenes_proceso = []
            
            if not df_medias.empty:
                columnas_vars = [c for c in df_medias.columns if c not in ["Segmento", "Velocidad (mm/año)"]]
                for var in columnas_vars:
                    fig_proc, ax_proc = plt.subplots(figsize=(6, 4))
                    ax_proc.scatter(df_medias["Velocidad (mm/año)"], df_medias[var], alpha=0.7)
                    ax_proc.set_xlabel("Velocidad de corrosión (mm/año)")
                    ax_proc.set_ylabel(var)
                    ax_proc.grid(True, alpha=0.4)
                    ax_proc.set_title(f"{var} vs Velocidad")
                    proc_path = carpeta / f"{nombre_base}_{var}_vs_velocidad.png"
                    fig_proc.savefig(proc_path, dpi=150, bbox_inches="tight")
                    plt.close(fig_proc)
                    try:
                        imagenes_proceso.append(Image.open(proc_path))
                    except:
                        pass
            
            # Crear collage si hay imágenes
            if imagenes_proceso:
                cols = 2
                filas = math.ceil(len(imagenes_proceso) / cols)
                w, h = imagenes_proceso[0].size
                collage_proc = Image.new("RGB", (cols*w, filas*h), "white")
            
                for n, img in enumerate(imagenes_proceso):
                    fila = n // cols
                    col = n % cols
                    collage_proc.paste(img, (col*w, fila*h))
            
                collage_proc_path = carpeta / f"{nombre_base}_collage_proceso.png"
                collage_proc.save(collage_proc_path)
                z.write(collage_proc_path, arcname=f"{nombre_base}/{collage_proc_path.name}")

    zip_buffer.seek(0)

    st.download_button(
        "⬇️ Descargar ZIP completo",
        data=zip_buffer,
        file_name="export_completo.zip",
        mime="application/zip"
    )

    st.success("Exportación completa generada.")
    
with tabs[3]:

    st.header("Tabla corregida y control avanzado")
    st.subheader("Sondas activas para el análisis")

    processed = st.session_state.get("processed_sheets", {})

    # SOLO sondas guardadas
    processed = {
        k: v for k, v in processed.items()
        if v.get("saved")
    }
    
    sondas_activas_disponibles = sorted(processed.keys())
    if "sondas_seleccionadas" not in st.session_state:
        st.session_state["sondas_seleccionadas"] = sondas_activas_disponibles
    
    sondas_seleccionadas = st.multiselect(
        "Selecciona las sondas a utilizar",
        sondas_activas_disponibles,
        default=st.session_state["sondas_seleccionadas"]
    )
    
    st.session_state["sondas_seleccionadas"] = sondas_seleccionadas
    processed_filtrado = {
        k: v for k, v in processed.items()
        if k in sondas_seleccionadas and v.get("saved")
    }
    umbral_diag = st.slider(
        "Tolerancia respecto a la diagonal (mm/año)",
        min_value=0.0,
        max_value=1.0,
        value=0.05,
        step=0.01
    )
    # =========================================
    # TABLA BASE ENTRE SONDAS
    # =========================================
    
    df_comp = construir_tabla_segmentos_comparativa(
        processed_filtrado,
        st.session_state.get("df_mpa"),
        material_sel
    )
    
    # aplicar filtro de error
    processed_filtrado = aplicar_umbral_error_segmentos(
        processed_filtrado,
        df_comp,
        umbral_error_segmento
    )
    
    # 🔥 RECONSTRUIR tabla ya filtrada
    df_comp = construir_tabla_segmentos_comparativa(
        processed_filtrado,
        st.session_state.get("df_mpa"),
        material_sel
    )
    # =========================================
    # TABLA CORREGIDA FINAL
    # =========================================
    
    df_corr = df_comp.copy()
    
    # Crear columnas solo si existen las originales
    if "Media velocidades" in df_corr.columns:
        df_corr["Velocidad experimental"] = df_corr["Media velocidades"]
    else:
        df_corr["Velocidad experimental"] = np.nan
    
    if "Velocidad esperada" in df_corr.columns:
        df_corr["Velocidad teórica"] = df_corr["Velocidad esperada"]
    else:
        df_corr["Velocidad teórica"] = np.nan
   
    if df_corr.empty:
        st.info("No hay datos suficientes.")
        st.stop()

    df_validos = df_corr.copy()
    # =========================================
    # CLASIFICACIÓN RESPECTO A LA DIAGONAL
    # =========================================
    df_validos["delta_diag"] = (
        df_validos["Velocidad experimental"]
        - df_validos["Velocidad teórica"]
    )
    
    df_validos["estado_diag"] = df_validos["delta_diag"].apply(
        lambda x:
            "ENCIMA" if x > umbral_diag
            else "DEBAJO" if x < -umbral_diag
            else "DENTRO"
    )
    # =========================================
    # AÑADIR CRUDOS A LA TABLA
    # =========================================
    
    if "df_master_global" in st.session_state:
    
        df_master = st.session_state["df_master_global"]
    
        mapa_crudos = (
            df_master.groupby("Segmento")["Crudo"]
            .apply(lambda x: ", ".join(sorted(x.unique())))
            .to_dict()
        )
    
        df_validos["Crudos"] = df_validos["Segmento"].map(mapa_crudos)
    # ===============================
    # AÑADIR CRUDOS A DESCARTADOS
    # ===============================
    
    df_master = None
    
    if "df_master_global" in st.session_state:
        df_master = st.session_state["df_master_global"]
    
    if df_master is not None and not df_master.empty:
    
        mapa_crudos = mapear_crudos_a_segmentos(df_master)
    
        def obtener_crudos(row):
            sonda_full = row["Sonda"]
            try:
                sonda, hoja = sonda_full.split(" | ")
            except:
                return None
    
            key = (sonda, hoja, row["Segmento"])
            return mapa_crudos.get(key)
        
        st.session_state["df_master_global"] = df_master
    # ===============================
    # TABLA PRINCIPAL
    # ===============================

    st.subheader("Segmentos válidos (filtrados)")
    st.dataframe(df_validos)

    # ===============================
    # TABLA DESCARTADOS
    # ===============================

    # ===============================
    # GRÁFICO DINÁMICO — REGRESIÓN FORZADA AL ORIGEN
    # ===============================

    import plotly.graph_objects as go
    import numpy as np

    st.subheader("Experimental vs Teórica (regresión forzada al origen)")

    df_plot = df_validos.dropna(
        subset=["Velocidad experimental", "Velocidad teórica"]
    )

    if df_plot.empty:
        st.warning("No hay datos para graficar.")
        st.stop()
        
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
    
        x=df_validos["Velocidad experimental"],
        y=df_validos["Velocidad teórica"],
    
        mode="markers",
    
        marker=dict(size=10)
    
    ))
    
    max_val = max(
        df_validos["Velocidad teórica"].max(),
        df_validos["Velocidad experimental"].max()
    )
    
    # Línea diagonal ideal
    fig.add_trace(go.Scatter(
        x=[0,max_val],
        y=[0,max_val],
        mode="lines",
        line=dict(color="red"),
        name="y = x"
    ))
    fig.add_trace(go.Scatter(
        x=[0, max_val],
        y=[0 + umbral_diag, max_val + umbral_diag],
        mode="lines",
        line=dict(color="green", dash="dash"),
        name="Límite superior tolerancia"
    ))
    
    fig.add_trace(go.Scatter(
        x=[0, max_val],
        y=[0 - umbral_diag, max_val - umbral_diag],
        mode="lines",
        line=dict(color="green", dash="dash"),
        name="Límite inferior tolerancia"
    ))
    df_encima = df_validos[df_validos["estado_diag"]=="ENCIMA"]
    df_debajo = df_validos[df_validos["estado_diag"]=="DEBAJO"]
    df_dentro = df_validos[df_validos["estado_diag"]=="DENTRO"]
    if "df_master_global" in st.session_state:
    
        df_master = st.session_state["df_master_global"]
    
        st.subheader("Crudos involucrados")
    
        seg_encima = df_encima["Segmento"].unique()
        seg_debajo = df_debajo["Segmento"].unique()
    
        crudos_encima = df_master[df_master["Segmento"].isin(seg_encima)]
        crudos_debajo = df_master[df_master["Segmento"].isin(seg_debajo)]
    
        st.markdown("### Crudos en segmentos ENCIMA")
        st.dataframe(
            crudos_encima.groupby("Crudo")["Porcentaje_promedio"]
            .mean()
            .sort_values(ascending=False)
        )
    
        st.markdown("### Crudos en segmentos DEBAJO")
        st.dataframe(
            crudos_debajo.groupby("Crudo")["Porcentaje_promedio"]
            .mean()
            .sort_values(ascending=False)
        )  
    st.subheader("Impacto de los parámetros del crudo sobre las velocidades experimentales")

    vars_proc = st.session_state.get("vars_proceso", [])
    
    # ENCIMA
    st.markdown("### Velocidades medidas subestimadas")
    
    imp_encima = analizar_importancia_variables(df_encima, vars_proc)
    
    if not imp_encima.empty:
        st.dataframe(imp_encima)
    else:
        st.info("No hay suficientes datos")
    
    # DEBAJO
    st.markdown("### Velocidades medidas sobreestimadas")
    
    imp_debajo = analizar_importancia_variables(df_debajo, vars_proc)
    
    if not imp_debajo.empty:
        st.dataframe(imp_debajo)
    else:
        st.info("No hay suficientes datos")
    # Zonas interpretativas
    
    fig.add_annotation(
        x=max_val*0.25,
        y=max_val*0.8,
        text="Sobreestimación<br>(predicción conservadora)",
        showarrow=False
    )
    
    fig.add_annotation(
        x=max_val*0.75,
        y=max_val*0.3,
        text="Subestimación<br>(predicción ambiciosa)",
        showarrow=False
    )
    
    fig.update_layout(

        xaxis_title="Velocidad experimental (mm/año)",
        yaxis_title="Velocidad teórica MPA (mm/año)",)
    
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Segmentos dentro del umbral")
    st.dataframe(df_dentro)
    
    st.subheader("Valores por debajo de la diagonal")
    st.dataframe(df_encima)
    
    st.subheader("Valores por encima de la diagonal")
    st.dataframe(df_debajo)
    st.subheader("Relación con variables de proceso")
    # =========================================
    # ANALISIS DE CRUDOS POR ZONA
    # =========================================
    
    if "df_master_global" in st.session_state:
    
        df_master = st.session_state["df_master_global"]
    
        st.subheader("Crudos involucrados en desviaciones del modelo")
    
        seg_encima = df_encima["Segmento"].unique()
        seg_debajo = df_debajo["Segmento"].unique()
        seg_dentro = df_dentro["Segmento"].unique()
    
        crudos_encima = df_master[df_master["Segmento"].isin(seg_encima)]
        crudos_debajo = df_master[df_master["Segmento"].isin(seg_debajo)]
        crudos_dentro = df_master[df_master["Segmento"].isin(seg_dentro)]
    
        col1, col2, col3 = st.columns(3)
    
        with col1:
            st.markdown("### Crudos en segmentos Subestimados ")
        
            if not crudos_encima.empty:
        
                tabla = (
                    crudos_encima
                    .groupby("Crudo")
                    .agg(
                        Porcentaje_promedio=("Porcentaje_promedio", "mean"),
                        num_segmentos=("Segmento", "nunique"),
                        segmentos=("Segmento", lambda x: ", ".join(map(str, sorted(set(x)))))
                    )
                    .sort_values("Porcentaje_promedio", ascending=False)
                    .reset_index()
                )
        
                st.dataframe(tabla)
    
        with col2:
            st.markdown("### Crudos en segmentos Sobreestimados")
        
            if not crudos_debajo.empty:
        
                tabla = (
                    crudos_debajo
                    .groupby("Crudo")
                    .agg(
                        Porcentaje_promedio=("Porcentaje_promedio", "mean"),
                        num_segmentos=("Segmento", "nunique"),
                        segmentos=("Segmento", lambda x: ", ".join(map(str, sorted(set(x)))))
                    )
                    .sort_values("Porcentaje_promedio", ascending=False)
                    .reset_index()
                )
        
                st.dataframe(tabla)
    
        with col3:
            st.markdown("### Crudos dentro del umbral")
        
            if not crudos_dentro.empty:
        
                tabla = (
                    crudos_dentro
                    .groupby("Crudo")
                    .agg(
                        Porcentaje_promedio=("Porcentaje_promedio", "mean"),
                        num_segmentos=("Segmento", "nunique"),
                        segmentos=("Segmento", lambda x: ", ".join(map(str, sorted(set(x)))))
                    )
                    .sort_values("Porcentaje_promedio", ascending=False)
                    .reset_index()
                )
        
                st.dataframe(tabla)
    variables_proceso = [
        c for c in df_validos.columns
        if c not in [
            "Velocidad experimental",
            "Velocidad teórica",
            "estado_diag",
            "delta_diag",
            "Segmento"
        ]
    ]
    
    variable_x = st.selectbox(
        "Selecciona variable de proceso",
        variables_proceso
    )
    import plotly.graph_objects as go

    df_plot = df_validos.dropna(subset=[variable_x])
    df_abajo = df_plot[df_plot["estado_diag"] == "DEBAJO"]

    fig_abajo = go.Figure()
    
    fig_abajo.add_trace(go.Scatter(
        x=df_abajo[variable_x],
        y=df_abajo["Velocidad experimental"],
        mode="markers",
        name="Segmentos DEBAJO"
    ))
    
    x_reg, y_reg, r2 = calcular_regresion(
        df_abajo[variable_x],
        df_abajo["Velocidad experimental"]
    )
    
    if x_reg is not None:
    
        fig_abajo.add_trace(go.Scatter(
            x=x_reg,
            y=y_reg,
            mode="lines",
            name="Regresión"
        ))
        if r2 is not None:
            fig_abajo.add_annotation(
                x=0.05,
                y=0.95,
                xref="paper",
                yref="paper",
                text=f"R² = {r2:.3f}",
                showarrow=False
            )
    fig_abajo.update_layout(
        title="MPA sobreestima corrosión",
        xaxis_title=variable_x,
        yaxis_title="Velocidad experimental promedio"
    )
    
    st.plotly_chart(fig_abajo, use_container_width=True)
    
    df_arriba = df_plot[df_plot["estado_diag"] == "ENCIMA"]
    fig_arriba = go.Figure()

    fig_arriba.add_trace(go.Scatter(
        x=df_arriba[variable_x],
        y=df_arriba["Velocidad experimental"],
        mode="markers",
        name="Segmentos ENCIMA"
    ))
    
    x_reg, y_reg, r2 = calcular_regresion(
        df_arriba[variable_x],
        df_arriba["Velocidad experimental"]
    )
    
    if x_reg is not None:
    
        fig_arriba.add_trace(go.Scatter(
            x=x_reg,
            y=y_reg,
            mode="lines",
            name="Regresión"
        ))
        if r2 is not None:
            fig_arriba.add_annotation(
                x=0.05,
                y=0.95,
                xref="paper",
                yref="paper",
                text=f"R² = {r2:.3f}",
                showarrow=False
            )
    fig_arriba.update_layout(
        title="MPA subestima corrosión",
        xaxis_title=variable_x,
        yaxis_title="Velocidad experimental promedio"
    )
    
    st.plotly_chart(fig_arriba, use_container_width=True)
    st.markdown("## 🛢️ Análisis por cestas de crudo")

    if uploaded_crudos is not None:
    
        hojas = leer_archivo(uploaded_crudos)
        hoja = list(hojas.keys())[0]
        df_crudos = hojas[hoja]
    
        detalle_crudos = procesar_crudos(df_crudos)
    
        cestas = construir_cestas_crudo(detalle_crudos)
    
        df_cestas = analizar_cestas(
            cestas,
            df_validos,
            st.session_state.get("df_proc")
        )
    
        if not df_cestas.empty:
    
            st.subheader("Cestas detectadas")
            st.dataframe(df_cestas)
    
            df_rank = ranking_cestas(df_cestas)
    
            st.subheader("Ranking de cestas")
            st.dataframe(df_rank)
    
        else:
            st.info("No se pudieron generar cestas")

    if not df_cestas.empty:
        # =========================================
        # 1. Ranking por estado
        # =========================================
        st.subheader("Ranking de cestas por comportamiento")
    
        df_rank_estado = ranking_cestas_por_estado(df_cestas)
    
        st.dataframe(df_rank_estado)
    
    st.markdown("## 🔬 Análisis avanzado de cestas (robusto con pocos datos)")
    st.session_state["df_cestas_global"] = df_cestas
    
    # =========================
    # ENRIQUECER RANKING
    # =========================
    df_cestas_proc = enriquecer_cestas_con_proceso(
        df_cestas,
        st.session_state.get("vars_proceso", [])
    )
    
    st.subheader("Ranking enriquecido de cestas")
    st.dataframe(df_cestas_proc)
    
    # =========================
    # % CRUDOS POR TIPO
    # =========================
    df_pct = analisis_porcentaje_crudos_top_cestas(
        df_cestas,
        detalle_crudos
    )
    
    if not df_pct.empty:
    
        st.subheader("Composición promedio por tipo de desviación")
        
    st.markdown("## 🔎 Buscador por especie de crudo")
    
    if not df_cestas.empty and "detalle_crudos" in locals():
    
        # obtener especies disponibles
        especies = sorted(detalle_crudos["Especie"].dropna().unique())
    
        especie_sel = st.selectbox(
            "Selecciona especie de crudo",
            especies
        )
    
        if especie_sel:
    
            df_result = buscar_especie_en_cestas(
                df_cestas,
                detalle_crudos,
                especie_sel
            )
    
            if not df_result.empty:
    
                st.subheader(f"Cestas donde aparece {especie_sel}")
                st.dataframe(df_result)
    
                st.subheader("Resumen por cesta")
                df_resumen = resumen_especie(df_result)
                st.dataframe(df_resumen)
    
            else:
                st.warning("La especie no aparece en ninguna cesta válida")
        st.dataframe(df_pct)
        st.subheader("Relación % crudo vs corrosión")
        
        fig = graficar_especie_vs_corrosion(
            df_result,
            especie_sel
        )
        
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No hay datos suficientes para graficar")
            
#-----------------------------------Modelo predictivo----------------------------------------------------------------     
with tabs[4]:  
    modo_modelo = st.selectbox(
        "Modo de modelado",
        ["Segmentos", "Cestas de crudo"]
    )
    st.header("Modelo predictivo")

    processed = st.session_state.get("processed_sheets", {})

    processed = {
        k: v for k, v in processed.items()
        if v.get("saved")
    }

    if not processed:
        st.info("No hay datos procesados")
        st.stop()

    # selector sondas
    sondas = list(processed.keys())

    sel_sondas = st.multiselect(
        "Selecciona sondas",
        sondas,
        default=sondas
    )

    processed_filtrado = {
        k: v for k, v in processed.items()
        if k in sel_sondas
    }

    # 1️⃣ construir tabla base
    df_comp = construir_tabla_segmentos_comparativa(
        processed_filtrado,
        st.session_state.get("df_mpa"),
        material_sel
    )
    
    # 2️⃣ aplicar filtro de error (CLAVE)
    processed_filtrado = aplicar_umbral_error_segmentos(
        processed_filtrado,
        df_comp,
        st.session_state.get("umbral_error_segmento", 30.0)
    )
    
    # 3️⃣ reconstruir tabla ya filtrada
    df_comp = construir_tabla_segmentos_comparativa(
        processed_filtrado,
        st.session_state.get("df_mpa"),
        material_sel
    )
    
    # 4️⃣ variable objetivo
    df_comp["Velocidad experimental"] = df_comp["Media velocidades"]
    if modo_modelo == "Segmentos":
    
        # =========================
        # 🔵 MODELO MPA
        # =========================
    
        st.subheader("Modelo MPA")
    
        tol = st.slider(
            "Tolerancia MPA",
            0.0, 1.0, 0.05, 0.01
        )
    
        fig_mpa = grafica_modelo_vs_real(
            df_comp["Velocidad experimental"],
            df_comp["Velocidad esperada"],
            "MPA vs Experimental",
            tol
        )
        
        st.plotly_chart(fig_mpa, use_container_width=True)
        
        # =========================
        # 🟢 MODELOS ML
        # =========================
    
        st.subheader("Modelo de datos (Machine Learning)")
    
        modelos, y_real = entrenar_modelos_ml(
            df_comp,
            st.session_state.get("vars_proceso", [])
        )
        if not modelos:
            st.warning("No hay suficientes datos para ML")
            st.stop()
    
        mejor_modelo = None
        mejor_r2 = -999
        st.subheader("Tolerancia modelos ML")

        tol_ml_global = st.slider(
            "Tolerancia para TODOS los modelos ML",
            0.0, 1.0, 0.05, 0.01,
            key="tol_ml_global"
        )
        for nombre, data in modelos.items():
    
            st.markdown(f"### {nombre} (R² = {data['r2']:.3f})")
    
            fig = grafica_modelo_vs_real(
                y_real,
                data["pred"],
                nombre,
                tolerancia=tol_ml_global
            )
    
            st.plotly_chart(fig, use_container_width=True,key=f"ml_{nombre}")
    
            if data["r2"] > mejor_r2:
                mejor_r2 = data["r2"]
                mejor_modelo = (nombre, data)
                nombre_best, data_best = mejor_modelo
    
        st.subheader("Importancia variables — mejor modelo ML")
        
        imp_ml = pd.DataFrame({
            "Variable": list(data_best["importancias"].keys()),
            "Importancia": list(data_best["importancias"].values())
        }).sort_values("Importancia", ascending=False)
        
        st.dataframe(imp_ml)
        st.subheader("Clasificación de segmentos por modelo")
        df_estado = df_comp.copy()
        
        df_estado["Segmento"] = df_estado["Segmento"]
        
        # MPA
        estado_mpa = clasificar_por_tolerancia(
            df_comp["Velocidad experimental"],
            df_comp["Velocidad esperada"],
            tol   # tu tolerancia MPA
        )
        
        df_estado["MPA"] = estado_mpa
        for nombre, data in modelos.items():
        
            estado_modelo = clasificar_por_tolerancia(
                y_real,
                data["pred"],
                tol_ml_global   # 👈 el que añadimos antes
            )
        
            df_estado[nombre] = estado_modelo
        cols_final = ["Segmento", "MPA"] + list(modelos.keys())
        df_estado_final = df_estado[cols_final]
        st.dataframe(df_estado_final)
        # =========================================
        # 📊 RESUMEN DE SEGMENTOS POR MODELO
        # =========================================
        
        st.subheader("Resumen de comportamiento por modelo")
        
        resumen = []
        
        for col in df_estado_final.columns:
        
            if col == "Segmento":
                continue
        
            counts = df_estado_final[col].value_counts()
        
            resumen.append({
                "Modelo": col,
                "ENCIMA": counts.get("ENCIMA", 0),
                "DEBAJO": counts.get("DEBAJO", 0),
                "DENTRO": counts.get("DENTRO", 0),
                "TOTAL": counts.sum()
            })
        
        df_resumen_modelos = pd.DataFrame(resumen)
        st.dataframe(df_resumen_modelos)
        import plotly.express as px

        df_plot = df_resumen_modelos.melt(
            id_vars="Modelo",
            value_vars=["ENCIMA", "DEBAJO", "DENTRO"],
            var_name="Estado",
            value_name="Cantidad"
        )
        
        fig = px.bar(
            df_plot,
            x="Modelo",
            y="Cantidad",
            color="Estado",
            barmode="group",
            title="Distribución de segmentos por modelo"
        )
        
        st.plotly_chart(fig, use_container_width=True,key="resumen_modelos")
        # =========================
        # 🟣 MEJOR MODELO
        # =========================
    
        st.subheader(f"Mejor modelo: {mejor_modelo[0]} (R²={mejor_r2:.3f})")
    
        tol_ml = st.slider(
            "Tolerancia modelo ML",
            0.0, 1.0, 0.05, 0.01,
            key="tol_ml"
        )
    
        fig_best = grafica_modelo_vs_real(
            y_real,
            mejor_modelo[1]["pred"],
            "Mejor modelo vs real",
            tol_ml
        )
    
        st.plotly_chart(fig_best, use_container_width=True, key="mejor_modelo")
        # IMPORTANCIA MPA
        imp_mpa = importancia_mpa(df_comp)
        
        if not imp_mpa.empty:
        
            df_plot = imp_ml.merge(
                imp_mpa,
                on="Variable",
                how="outer",
                suffixes=("_ML", "_MPA")
            ).fillna(0)
        
            import plotly.graph_objects as go
        
            fig = go.Figure()
        
            fig.add_trace(go.Bar(
                x=df_plot["Variable"],
                y=df_plot["Importancia_ML"],
                name="Modelo ML"
            ))
        
            fig.add_trace(go.Bar(
                x=df_plot["Variable"],
                y=df_plot["Importancia_MPA"],
                name="MPA"
            ))
        
            fig.update_layout(
                title="Comparación importancia variables (ML vs MPA)",
                barmode="group",
                xaxis_title="Variable",
                yaxis_title="Importancia"
            )
        # TODO lo que ya tienes
        df_comp = construir_tabla_segmentos_comparativa(
            processed_filtrado,
            st.session_state.get("df_mpa"),
            material_sel
        )
        
        df_comp["Velocidad experimental"] = df_comp["Media velocidades"]
    
        modelos, y_real = entrenar_modelos_ml(
            df_comp,
            st.session_state.get("vars_proceso", [])
        )
        st.plotly_chart(fig, use_container_width=True)
        # =========================================
        # 🔎 ANÁLISIS SUB / SOBRE ESTIMACIÓN (ML vs MPA)
        # =========================================
        
        st.subheader("Análisis de errores por tipo (ML vs MPA)")
        
        # =========================
        # CLASIFICACIÓN ML (SIN SPLIT)
        # =========================
        
        df_ml = pd.DataFrame({
            "real": y_real,
            "pred": data_best["pred"]
        }).reset_index(drop=True)
        
        df_ml["delta"] = df_ml["real"] - df_ml["pred"]
        
        df_ml["estado"] = df_ml["delta"].apply(
            lambda x: "DEBAJO" if x > tol_ml_global
            else "ENCIMA" if x < -tol_ml_global
            else "DENTRO"
        )
        
        # 🔗 aquí sí coincide 1:1
        df_ml_full = df_comp.copy().reset_index(drop=True)
        df_ml_full["estado"] = df_ml["estado"]
        
        
        # =========================
        # CLASIFICACIÓN MPA
        # =========================
        df_mpa = df_comp.copy()
        
        df_mpa["delta"] = (
            df_mpa["Velocidad experimental"]
            - df_mpa["Velocidad esperada"]
        )
        
        df_mpa["estado"] = df_mpa["delta"].apply(
            lambda x: "DEBAJO" if x > tol
            else "ENCIMA" if x < -tol
            else "DENTRO"
        )
        
        
        # =========================
        # FUNCIÓN IMPORTANCIA
        # =========================
        def importancia_por_subset(df, vars_proceso, target):
        
            resultados = []
        
            for var in vars_proceso:
        
                if var not in df.columns:
                    continue
        
                sub = df[[var, target]].dropna()
        
                if len(sub) < 3:
                    continue
        
                x = sub[var]
                y = sub[target]
        
                if x.std() == 0:
                    continue
        
                corr = np.corrcoef(x, y)[0,1]
        
                resultados.append({
                    "Variable": var,
                    "Importancia": abs(corr)
                })
        
            if not resultados:
                return pd.DataFrame()
        
            return pd.DataFrame(resultados).sort_values(
                "Importancia",
                ascending=False
            )
        
        
        # =========================
        # CALCULAR IMPORTANCIAS
        # =========================
        vars_proc = st.session_state.get("vars_proceso", [])
        
        imp_ml_encima = importancia_por_subset(
            df_ml_full[df_ml_full["estado"]=="ENCIMA"],
            vars_proc,
            "Velocidad experimental"
        )
        
        imp_ml_debajo = importancia_por_subset(
            df_ml_full[df_ml_full["estado"]=="DEBAJO"],
            vars_proc,
            "Velocidad experimental"
        )
        
        imp_mpa_encima = importancia_por_subset(
            df_mpa[df_mpa["estado"]=="ENCIMA"],
            vars_proc,
            "Velocidad experimental"
        )
        
        imp_mpa_debajo = importancia_por_subset(
            df_mpa[df_mpa["estado"]=="DEBAJO"],
            vars_proc,
            "Velocidad experimental"
        )
        
        
        # =========================
        # DEBUG (puedes quitar luego)
        # =========================
        st.write("ML ENCIMA:", len(df_ml_full[df_ml_full["estado"]=="ENCIMA"]))
        st.write("ML DEBAJO:", len(df_ml_full[df_ml_full["estado"]=="DEBAJO"]))
        st.write("MPA ENCIMA:", len(df_mpa[df_mpa["estado"]=="ENCIMA"]))
        st.write("MPA DEBAJO:", len(df_mpa[df_mpa["estado"]=="DEBAJO"]))
        
        
        # =========================
        # 🔴 SUBESTIMADOS
        # =========================
        st.subheader("🔴 Variables en segmentos SUBESTIMADOS")
        
        if not imp_ml_encima.empty or not imp_mpa_encima.empty:
        
            df_plot = imp_ml_encima.merge(
                imp_mpa_encima,
                on="Variable",
                how="outer",
                suffixes=("_ML", "_MPA")
            ).fillna(0)
        
            fig_sub = go.Figure()
        
            fig_sub.add_trace(go.Bar(
                x=df_plot["Variable"],
                y=df_plot["Importancia_ML"],
                name="ML"
            ))
        
            fig_sub.add_trace(go.Bar(
                x=df_plot["Variable"],
                y=df_plot["Importancia_MPA"],
                name="MPA"
            ))
        
            fig_sub.update_layout(
                title="Importancia variables — SUBESTIMADOS",
                barmode="group"
            )
        
            st.plotly_chart(fig_sub, use_container_width=True)
        
        else:
            st.info("No hay datos suficientes para SUBESTIMADOS")
        
        
        # =========================
        # 🔵 SOBREESTIMADOS
        # =========================
        st.subheader("🔵 Variables en segmentos SOBREESTIMADOS")
        
        if not imp_ml_debajo.empty or not imp_mpa_debajo.empty:
        
            df_plot = imp_ml_debajo.merge(
                imp_mpa_debajo,
                on="Variable",
                how="outer",
                suffixes=("_ML", "_MPA")
            ).fillna(0)
        
            fig_sobre = go.Figure()
        
            fig_sobre.add_trace(go.Bar(
                x=df_plot["Variable"],
                y=df_plot["Importancia_ML"],
                name="ML"
            ))
        
            fig_sobre.add_trace(go.Bar(
                x=df_plot["Variable"],
                y=df_plot["Importancia_MPA"],
                name="MPA"
            ))
        
            fig_sobre.update_layout(
                title="Importancia variables — SOBREESTIMADOS",
                barmode="group"
            )
        
            st.plotly_chart(fig_sobre, use_container_width=True)
        
        else:
            st.info("No hay datos suficientes para SOBREESTIMADOS")
    elif modo_modelo == "Cestas de crudo":
    
        df_cestas = st.session_state.get("df_cestas_global")
    
        if df_cestas is None or df_cestas.empty:
            st.warning("No hay cestas calculadas. Ve a 'Tabla corregida'")
            st.stop()
    
        # =========================
        # DATASET
        # =========================
        df_model = construir_dataset_modelo_cestas(df_cestas)
    
        st.subheader("Dataset de cestas")
        st.dataframe(df_model)
    
        # =========================
        # VARIABLES MODELO
        # =========================
        vars_proceso = st.session_state.get("vars_proceso", [])
    
        vars_especies = [
            c for c in df_model.columns
            if c.startswith("ESP_")
        ]
    
        vars_modelo = vars_proceso + vars_especies
    
        # =========================
        # MODELOS ML
        # =========================
        modelos, y_real = entrenar_modelos_ml(
            df_model,
            vars_modelo
        )
    
        if not modelos:
            st.warning("No hay suficientes datos para ML")
            st.stop()
    
        mejor_modelo = None
        mejor_r2 = -999
    
        for nombre, data in modelos.items():
            # =========================================
            # 📊 IMPORTANCIA VARIABLES POR MODELO
            # =========================================
            
            st.subheader("Importancia de variables por modelo (comparativa)")
            
            importancias_all = []
            
            for nombre, data in modelos.items():
            
                for var, imp in data["importancias"].items():
            
                    # solo variables de proceso (evita columnas raras)
                    if var in st.session_state.get("vars_proceso", []):
            
                        importancias_all.append({
                            "Modelo": nombre,
                            "Variable": var,
                            "Importancia": imp
                        })
            
            df_imp_all = pd.DataFrame(importancias_all)
            
            if not df_imp_all.empty:
            
                import plotly.express as px
            
                fig = px.bar(
                    df_imp_all,
                    x="Variable",
                    y="Importancia",
                    color="Modelo",
                    barmode="group",
                    title="Importancia de variables de proceso por modelo ML"
                )
            
                fig.update_layout(
                    xaxis_title="Variable",
                    yaxis_title="Importancia",
                    height=500
                )
            
                st.plotly_chart(fig, use_container_width=True)
            
            else:
                st.info("No hay datos de importancia disponibles")
            st.markdown(f"### {nombre} (R² = {data['r2']:.3f})")
    
            fig = grafica_modelo_vs_real(
                y_real,
                data["pred"],
                nombre,
                tolerancia=0
            )
    
            st.plotly_chart(fig, use_container_width=True)
    
            if data["r2"] > mejor_r2:
                mejor_r2 = data["r2"]
                mejor_modelo = (nombre, data)
        nombre_best, data_best = mejor_modelo
        # =========================
        # CLASIFICACIÓN ML (SIN SPLIT)
        # =========================
        
        df_ml = pd.DataFrame({
            "real": y_real,
            "pred": data_best["pred"]
        }).reset_index(drop=True)
        
        df_ml["delta"] = df_ml["real"] - df_ml["pred"]
        
        df_ml["estado"] = df_ml["delta"].apply(
            lambda x: "ENCIMA" if x > tol_ml_global
            else "DEBAJO" if x < -tol_ml_global
            else "DENTRO"
        )
        
        # 🔗 aquí sí coincide 1:1
        df_ml_full = df_comp.copy().reset_index(drop=True)
        df_ml_full["estado"] = df_ml["estado"]
        # =========================================
        # CLASIFICACIÓN MPA
        # =========================================
        
        df_mpa = df_comp.copy()
        
        df_mpa["delta"] = (
            df_mpa["Velocidad experimental"]
            - df_mpa["Velocidad esperada"]
        )
        
        df_mpa["estado"] = df_mpa["delta"].apply(
            lambda x: "DEBAJO" if x > tol
            else "ENCIMA" if x < -tol
            else "DENTRO"
        )
        vars_proc = st.session_state.get("vars_proceso", [])

        # ML
        imp_ml_encima = importancia_por_subset(
            df_ml_full[df_ml_full["estado"]=="ENCIMA"],
            vars_proc,
            "Velocidad experimental"
        )
        
        imp_ml_debajo = importancia_por_subset(
            df_ml_full[df_ml_full["estado"]=="DEBAJO"],
            vars_proc,
            "Velocidad experimental"
        )
        
        # MPA
        imp_mpa_encima = importancia_por_subset(
            df_mpa[df_mpa["estado"]=="ENCIMA"],
            vars_proc,
            "Velocidad experimental"
        )
        
        imp_mpa_debajo = importancia_por_subset(
            df_mpa[df_mpa["estado"]=="DEBAJO"],
            vars_proc,
            "Velocidad experimental"
        )
        # =========================
        # IMPORTANCIA ESPECIES
        # =========================
        st.subheader("Importancia de especies")
    
        imp_total = []
    
        for nombre, data in modelos.items():
    
            for var, imp in data["importancias"].items():
    
                if var.startswith("ESP_"):
    
                    imp_total.append({
                        "Modelo": nombre,
                        "Especie": var.replace("ESP_", ""),
                        "Importancia": imp
                    })
    
        df_imp = pd.DataFrame(imp_total)
    
        if not df_imp.empty:
            st.dataframe(
                df_imp.sort_values("Importancia", ascending=False)
            )
        st.subheader("🔴 Variables en segmentos SUBESTIMADOS")

        if not imp_ml_encima.empty or not imp_mpa_encima.empty:
        
            df_plot = imp_ml_encima.merge(
                imp_mpa_encima,
                on="Variable",
                how="outer",
                suffixes=("_ML", "_MPA")
            ).fillna(0)
        
            fig = go.Figure()
        
            fig.add_trace(go.Bar(
                x=df_plot["Variable"],
                y=df_plot["Importancia_ML"],
                name="ML"
            ))
        
            fig.add_trace(go.Bar(
                x=df_plot["Variable"],
                y=df_plot["Importancia_MPA"],
                name="MPA"
            ))
        
            fig.update_layout(
                title="Importancia variables — SUBESTIMADOS",
                barmode="group"
            )
        
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("No hay datos suficientes para SUBESTIMADOS")
            
        st.subheader("🔵 Variables en segmentos SOBREESTIMADOS")
        if not imp_ml_debajo.empty or not imp_mpa_debajo.empty:
        
            df_plot = imp_ml_debajo.merge(
                imp_mpa_debajo,
                on="Variable",
                how="outer",
                suffixes=("_ML", "_MPA")
            ).fillna(0)
        
            fig = go.Figure()
        
            fig.add_trace(go.Bar(
                x=df_plot["Variable"],
                y=df_plot["Importancia_ML"],
                name="ML"
            ))
        
            fig.add_trace(go.Bar(
                x=df_plot["Variable"],
                y=df_plot["Importancia_MPA"],
                name="MPA"
            ))
        
            fig.update_layout(
                title="Importancia variables — SOBREESTIMADOS",
                barmode="group"
            )
        
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("No hay datos suficientes para SOBREESTIMADOS")
        # =========================
        # CORRELACIÓN DIRECTA
        # =========================
        def analizar_especies_directo(df_model):
    
            resultados = []
    
            for col in df_model.columns:
    
                if col.startswith("ESP_"):
    
                    sub = df_model[[col, "Velocidad experimental"]].dropna()
    
                    if len(sub) < 3 or sub[col].sum() < 3:
                        continue
    
                    corr = np.corrcoef(
                        sub[col],
                        sub["Velocidad experimental"]
                    )[0,1]
    
                    resultados.append({
                        "Especie": col.replace("ESP_", ""),
                        "Correlacion": corr
                    })
    
            if not resultados:
                return pd.DataFrame()
    
            return pd.DataFrame(resultados).sort_values(
                "Correlacion",
                key=abs,
                ascending=False
            )
    
        df_corr = analizar_especies_directo(df_model)
    
        st.subheader("Correlación directa especies vs corrosión")
    
        if not df_corr.empty:
            st.dataframe(df_corr)
        else:
            st.info("No hay suficientes datos para correlación")
# -------------------- TAB 4: CRUDOS --------------------
with tabs[5]:

    st.header("Análisis avanzado de crudos")

    if uploaded_crudos is None:
        st.info("Sube Excel de crudos")
    
    else:
    
        hojas = leer_archivo(uploaded_crudos)
        hoja_sel = st.selectbox("Hoja crudos", list(hojas.keys()))
        df_crudos = hojas[hoja_sel]
    
        detalle_crudos = procesar_crudos(df_crudos)
    
        df_master = construir_dataset_crudos_segmentos(
            detalle_crudos,
            st.session_state.get("processed_sheets", {})
        )
        df_master["Segmento"] = "Seg " + df_master["Segmento"].astype(str)
        st.session_state["df_master_global"] = df_master
        if df_master.empty:
            st.warning("No hay coincidencias con segmentos")
            st.stop()
        # ==========================================
        # AÑADIR ESTADO DE SEGMENTO DESDE TABLA CORREGIDA
        # ==========================================
        
        df_corr = construir_tabla_corregida(
            st.session_state.get("processed_sheets", {}),
            st.session_state.get("df_mpa"),
            material_sel,
            sondas_seleccionadas
        )
        # =============================
        # BLOQUE 1 — ANALISIS SEGMENTOS
        # =============================
    
        st.subheader("Crudos y variables por segmento")
        # ======================================
        # ANALISIS GLOBAL DE CRUDOS
        # ======================================
        
        st.markdown("## 📊 Ranking de crudos más asociados a corrosión")
        
        df_rank = analizar_crudos_agresividad(df_master)
        
        if not df_rank.empty:
        
            st.dataframe(df_rank)
        
            crudo_top = df_rank.iloc[0]["Crudo"]
        
            st.success(
                f"El crudo más asociado a altas velocidades de corrosión es: **{crudo_top}**"
            )
        
        else:
            st.info("No hay suficientes datos para análisis de correlación.")

        st.dataframe(df_master)
        
        # ======================================
        # VARIABLES QUE EXPLICAN LA AGRESIVIDAD
        # ======================================
        
        if not df_rank.empty:
        
            crudo_top = df_rank.iloc[0]["Crudo"]
        
            st.markdown(f"## 🔬 Variables que explican la agresividad de {crudo_top}")
        
            df_imp_crudo = analizar_variables_en_crudo(
                df_master,
                crudo_top,
                st.session_state.get("vars_proceso", [])
            )

        
            if not df_imp_crudo.empty:
        
                st.dataframe(df_imp_crudo)
        
                var_top = df_imp_crudo.iloc[0]["Variable proceso"]
        
                st.success(
                    f"La variable más relacionada con el aumento de corrosión en {crudo_top} es: **{var_top}**"
                )
        
            else:
                st.info("No hay suficientes variables numéricas para análisis.")

        # =============================
        # BLOQUE 2 — BUSCADOR POR CRUDOS
        # =============================
    
        st.subheader("Buscador por crudo")
    
        crudos_disponibles = sorted(df_master["Crudo"].dropna().unique())
    
        crudo_sel = st.selectbox(
            "Selecciona crudo",
            crudos_disponibles
        )
    
        df_crudo = df_master[df_master["Crudo"] == crudo_sel]
    
        st.write(f"Segmentos donde aparece {crudo_sel}")

        st.dataframe(df_crudo)
    
        # -------- Dias reales ----------
        dias_crudo = detalle_crudos[
            detalle_crudos["Especie"] == crudo_sel
        ]
        
        # 👉 Añadir variables proceso
        dias_crudo = añadir_proceso_a_dias_crudo(
            dias_crudo,
            st.session_state.get("df_proc")
        )
        
        st.write("Días donde aparece el crudo")
        st.dataframe(dias_crudo)

    
        # =============================
        # BLOQUE 3 — Relación % crudo vs corrosión
        # =============================
        
        st.subheader("Relación porcentaje crudo vs velocidad corrosión")
        
        if not df_crudo.empty:
        
            vel_media = df_crudo["Velocidad_corr"].mean()
        
            st.metric("Velocidad media asociada", f"{vel_media:.4f} mm/año")
        
            fig, ax = plt.subplots(figsize=(6,5))
        
            ax.scatter(
                df_crudo["Porcentaje_promedio"],
                df_crudo["Velocidad_corr"],
                alpha=0.7
            )
        
            ax.set_xlabel("% promedio del crudo en el segmento")
            ax.set_ylabel("Velocidad corrosión (mm/año)")
            ax.set_title(f"{crudo_sel} → % vs corrosión")
        
            ax.grid(True)
        
            # 🔥 Añadir recta tendencia
            if len(df_crudo) > 1:
        
                x = df_crudo["Porcentaje_promedio"]
                y = df_crudo["Velocidad_corr"]
        
                coef = np.polyfit(x, y, 1)
                poly = np.poly1d(coef)
        
                ax.plot(x, poly(x))
        
            st.pyplot(fig)
    
        # =============================
        # EXPORTAR
        # =============================
    
        buffer = io.BytesIO()
        df_master.to_excel(buffer, index=False)
        buffer.seek(0)
    
        st.download_button(
            "Descargar dataset completo",
            buffer,
            file_name="crudos_segmentos_proceso.xlsx"
        )



# -------------------- Footer --------------------
st.markdown("---")
if user_module_path is not None:
    st.caption(f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} — Módulo usuario (si aplicable): {getattr(user_module_path,'name', str(user_module_path))}")
else:
    st.caption(f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
