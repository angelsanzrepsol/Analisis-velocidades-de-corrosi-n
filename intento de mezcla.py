# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 13:57:31 2025

@author: angel
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from dateutil.relativedelta import relativedelta
from io import BytesIO
from openpyxl.drawing.image import Image as XLImage
from PIL import Image as PILImage, ImageDraw
import matplotlib.dates as mdates
import warnings
import traceback
from datetime import datetime
import json
import pickle

warnings.filterwarnings("ignore", category=UserWarning)
plt.ion()
plt.style.use('default')

ARCHIVO_CORROSION = "1 resumen V3 sep 2025 v2.xlsx"
ARCHIVO_PROCESO = "Datos proceso Petronor GOPV.xlsx"
ARCHIVO_RESUMEN = "resumen fin.xlsx"
ARCHIVO_SALIDA = "resumen_velocidades.xlsx"
CARPETA_EXPORT = "graficos_exportados"
CARPETA_RESUMEN = "resumen_total"
CARPETA_PROCESADOS = "procesados_finales"
MIN_DIAS_SEG = 10
WL_MAX = 51
WL_MIN = 5
FMT_FECHA_EJE = "%Y-%m-%d"
FIGSIZE_DEFAULT = (14, 10)

def asegurar_carpeta(path):
    os.makedirs(path, exist_ok=True)
    return path

def safe_strcol(s):
    return "".join(c if (c.isalnum() or c in " _-") else "_" for c in str(s))

def _to_datetime_safe(s):
    try:
        return pd.to_datetime(s)
    except Exception:
        return pd.NaT

def print_header(msg):
    border = "=" * max(40, len(msg) + 4)
    print(f"\n{border}\n  {msg}\n{border}\n")

def mostrar_error(e, context=""):
    print(f"[ERROR] {context} : {e}")
    traceback.print_exc()

def detect_columns(df):
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
            if col_espesor is None:
                if pd.api.types.is_numeric_dtype(df[c]):
                    col_espesor = c
                    break
    if col_fecha is None or col_espesor is None:
        raise KeyError(f"No se encontró columna de fecha y/o espesor. Encontrados: Fecha={col_fecha}, Espesor={col_espesor}")
    return col_fecha, col_espesor

def cargar_datos_proceso(ruta_excel):
    if not os.path.exists(ruta_excel):
        raise FileNotFoundError(f"No se encontró '{ruta_excel}'")
    try:
        df_raw = pd.read_excel(ruta_excel, sheet_name=0, header=None)
        nombres = df_raw.iloc[1, 2:10].tolist()
        df_proc = df_raw.iloc[8:, 1:10].copy()
        cols = ["Fecha"] + [str(x).strip() for x in nombres]
        df_proc.columns = cols
    except Exception:
        df_proc = pd.read_excel(ruta_excel, sheet_name=0).copy()
        df_proc = df_proc.reset_index(drop=True)
        if "Fecha" not in df_proc.columns:
            cols = list(df_proc.columns)
            cols[0] = "Fecha"
            df_proc.columns = cols
    df_proc.columns = [str(c).strip() for c in df_proc.columns]
    df_proc["Fecha"] = pd.to_datetime(df_proc["Fecha"], errors="coerce")
    df_proc = df_proc.dropna(subset=["Fecha"]).sort_values("Fecha").reset_index(drop=True)
    for c in df_proc.columns:
        if c != "Fecha":
            df_proc[c] = pd.to_numeric(df_proc[c], errors="coerce")
    vars_proceso = [c for c in df_proc.columns if c != "Fecha"]
    print(f"Datos proceso cargados: {len(df_proc)} filas. Variables: {vars_proceso}")
    return df_proc, vars_proceso

def detectar_segmentos(df_original, umbral_factor=1.02, umbral=0.0005, min_dias=MIN_DIAS_SEG, wl_max=WL_MAX):
    df = df_original.copy()
    try:
        col_fecha, col_espesor = detect_columns(df)
    except KeyError as e:
        print("detect_columns error:", e)
        return None, None, [], []
    df["Sent Time"] = pd.to_datetime(df[col_fecha], errors="coerce")
    df["UT measurement (mm)"] = pd.to_numeric(df[col_espesor], errors="coerce")
    df = df.sort_values("Sent Time").reset_index(drop=True)
    df = df.dropna(subset=["Sent Time", "UT measurement (mm)"]).reset_index(drop=True)
    if len(df) == 0:
        return None, None, [], []
    n_ref = min(10, len(df))
    grosor_ref = df["UT measurement (mm)"].iloc[:n_ref].mean()
    df_filtrado = df[df["UT measurement (mm)"] <= grosor_ref * umbral_factor].reset_index(drop=True)
    if len(df_filtrado) < 5:
        return df_filtrado, None, [], []
    y = df_filtrado["UT measurement (mm)"].values
    wl = min(wl_max, (len(y) - 1) if (len(y) % 2 == 0) else len(y))
    wl = max(WL_MIN, wl)
    if wl % 2 == 0:
        wl += 1
    try:
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
    return df_filtrado, y_suave, cambios, segmentos_raw

def extraer_segmentos_validos(df_filtrado, y_suave, segmentos_raw, df_proc, vars_proceso, min_dias=MIN_DIAS_SEG):
    segmentos_validos = []
    descartados = []
    for seg in segmentos_raw:
        ini, fin = seg["ini"], seg["fin"]
        fecha_ini, fecha_fin = seg["fecha_ini"], seg["fecha_fin"]
        delta_dias = seg["delta_dias"]
        velocidad = seg["velocidad"]
        if pd.isna(fecha_ini) or pd.isna(fecha_fin):
            seg2 = dict(seg); seg2.update({"motivo": "Fechas inválidas", "estado": "descartado"}); descartados.append(seg2); continue
        if delta_dias <= 0 or delta_dias < min_dias:
            seg2 = dict(seg); seg2.update({"motivo": f"Duración < {min_dias} días", "estado": "descartado"}); descartados.append(seg2); continue
        if velocidad is None or (not np.isfinite(velocidad)):
            seg2 = dict(seg); seg2.update({"motivo": "Velocidad NaN", "estado": "descartado"}); descartados.append(seg2); continue
        if velocidad >= 0:
            seg2 = dict(seg); seg2.update({"motivo": "Velocidad no negativa", "estado": "descartado"}); descartados.append(seg2); continue
        medias = pd.Series(dtype=float)
        if df_proc is not None and not df_proc.empty:
            try:
                sub = df_proc[(df_proc["Fecha"] >= fecha_ini - pd.Timedelta(days=1)) & (df_proc["Fecha"] <= fecha_fin + pd.Timedelta(days=1))]
                medias = sub.mean(numeric_only=True)
            except Exception:
                medias = pd.Series(dtype=float)
        rd = relativedelta(fecha_fin, fecha_ini)
        anios, meses = rd.years, rd.months
        if anios == 0 and meses == 0 and rd.days > 0:
            meses = 1
        if meses == 12:
            anios += 1; meses = 0
        segmentos_validos.append({"ini": ini, "fin": fin, "fecha_ini": fecha_ini, "fecha_fin": fecha_fin, "delta_dias": delta_dias, "velocidad": velocidad, "vel_abs": abs(velocidad), "medias": medias, "anios": anios, "meses": meses, "estado": "valido", "num_segmento_valido": None})
    return segmentos_validos, descartados

def dibujar_grafica_completa(df_filtrado, y_suave, segmentos_validos, descartados, segmentos_eliminados_idx, titulo="Velocidad de corrosión", figsize=FIGSIZE_DEFAULT, show=True):
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white"); ax.grid(True, alpha=0.35)
    ax.plot(df_filtrado["Sent Time"], df_filtrado["UT measurement (mm)"].values, color="gray", alpha=0.25, linewidth=1.2, label="Mediciones")
    ymax, ymin = float(np.max(y_suave)), float(np.min(y_suave)); altura = ymax - ymin if (ymax - ymin) != 0 else max(abs(ymax), 1.0)
    ax.set_ylim(ymin - 0.05 * altura, ymax + 0.2 * altura)
    gris_alpha = 0.35
    for d in descartados:
        i, f = d["ini"], d["fin"]
        if i < 0 or f <= i or f > len(y_suave): continue
        ax.plot(df_filtrado["Sent Time"].iloc[i:f], y_suave[i:f], color="gray", alpha=gris_alpha, linewidth=2)
        ax.fill_between(df_filtrado["Sent Time"].iloc[i:f], y_suave[i:f], ymin, color="gray", alpha=gris_alpha)
    for (i, f) in segmentos_eliminados_idx:
        if i < 0 or f <= i or f > len(y_suave): continue
        ax.plot(df_filtrado["Sent Time"].iloc[i:f], y_suave[i:f], color="gray", alpha=gris_alpha, linewidth=2)
        ax.fill_between(df_filtrado["Sent Time"].iloc[i:f], y_suave[i:f], ymin, color="gray", alpha=gris_alpha)
    validos = [s for s in segmentos_validos if s.get("estado","valido") == "valido"]
    colormap = plt.cm.get_cmap("turbo", max(2, len(validos)))
    contador = 0
    for s in sorted(segmentos_validos, key=lambda x: x["fecha_ini"] if x.get("fecha_ini") is not None else pd.Timestamp.max):
        if s.get("estado","valido") != "valido": continue
        contador += 1; s["num_segmento_valido"] = contador
        i, f = s["ini"], s["fin"]; color = colormap((contador - 1) % max(1, colormap.N))
        ax.plot(df_filtrado["Sent Time"].iloc[i:f], y_suave[i:f], color=color, linewidth=2.6, label=f"Segmento {contador}: {s['fecha_ini'].strftime('%Y-%m-%d')} → {s['fecha_fin'].strftime('%Y-%m-%d')}\nDur: {s['anios']}a {s['meses']}m | Vel: {s['vel_abs']:.4f} mm/año")
        ax.fill_between(df_filtrado["Sent Time"].iloc[i:f], y_suave[i:f], ymin, color=color, alpha=0.25)
        for fecha in [s["fecha_ini"], s["fecha_fin"]]:
            ax.axvline(fecha, color="black", linestyle=":", alpha=0.5, zorder=0)
            ax.text(fecha, ymax + 0.07 * altura, fecha.strftime("%Y-%m-%d"), ha="center", va="bottom", rotation=90, fontsize=8, color="black", bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85, lw=0))
        centro_idx = min((i + f) // 2, len(df_filtrado) - 1)
        x_centro = df_filtrado["Sent Time"].iloc[centro_idx]; y_centro = ymin + 0.45 * altura
        ax.text(x_centro, y_centro, f"{s['vel_abs']:.4f} mm/año", ha="center", va="center", rotation=90, fontsize=10, fontweight="bold", color=color, bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.9, lw=0))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter(FMT_FECHA_EJE))
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center", fontsize=9)
    ax.set_title(titulo, fontsize=14, fontweight="bold"); ax.set_xlabel("Fecha", fontsize=12); ax.set_ylabel("UT measurement (mm)", fontsize=12)
    leg = ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=9, title="Segmentos", borderaxespad=0.)
    for text in leg.get_texts(): text.set_multialignment('left')
    plt.tight_layout()
    if show:
        try:
            plt.show(block=False)
        except Exception:
            plt.show()
    return fig, ax

def recalcular_segmento_local(df_filtrado, y_suave, segmento, df_proc, vars_proceso, nuevo_umbral, nuevo_umbral_factor=None, min_dias=MIN_DIAS_SEG, wl_max=WL_MAX):
    ini_g, fin_g = segmento["ini"], segmento["fin"]
    df_local = df_filtrado.iloc[ini_g:fin_g].reset_index(drop=True)
    if df_local.empty or len(df_local) < 5:
        return [], [{"ini": ini_g, "fin": fin_g, "motivo": "Datos insuficientes local", "estado": "descartado"}]
    if nuevo_umbral_factor is not None:
        n_ref_local = min(10, len(df_local)); grosor_ref_local = df_local["UT measurement (mm)"].iloc[:n_ref_local].mean()
        mask = df_local["UT measurement (mm)"] <= grosor_ref_local * nuevo_umbral_factor
        df_local = df_local[mask].reset_index(drop=True)
        if df_local.empty or len(df_local) < 5:
            return [], [{"ini": ini_g, "fin": fin_g, "motivo": "Filtro local eliminó casi todo", "estado": "descartado"}]
    y_local = df_local["UT measurement (mm)"].values
    wl = min(wl_max, (len(y_local) - 1) if (len(y_local) % 2 == 0) else len(y_local))
    wl = max(WL_MIN, wl)
    if wl % 2 == 0: wl += 1
    try:
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
        if a < 0 or b <= a or b > len(df_local): continue
        fecha_a = pd.to_datetime(df_local["Sent Time"].iloc[a], errors="coerce"); fecha_b = pd.to_datetime(df_local["Sent Time"].iloc[b - 1], errors="coerce")
        delta_dias = (fecha_b - fecha_a).days if (pd.notna(fecha_a) and pd.notna(fecha_b)) else 0
        velocidad = np.nan
        if delta_dias > 0:
            try: velocidad = (y_suave_local[b - 1] - y_suave_local[a]) / (delta_dias / 365.25)
            except Exception: velocidad = np.nan
        segmentos_raw_local.append({"ini": a, "fin": b, "fecha_ini": fecha_a, "fecha_fin": fecha_b, "delta_dias": delta_dias, "velocidad": velocidad})
    nuevos_validos_global = []; nuevos_descartados_global = []
    for s in segmentos_raw_local:
        if pd.isna(s["fecha_ini"]) or pd.isna(s["fecha_fin"]):
            nuevos_descartados_global.append({"ini": ini_g + s.get("ini", 0), "fin": ini_g + s.get("fin", 0), "motivo": "Fechas invalidas local", "estado": "descartado"}); continue
        if s["delta_dias"] <= 0 or s["delta_dias"] < min_dias:
            nuevos_descartados_global.append({"ini": ini_g + s.get("ini", 0), "fin": ini_g + s.get("fin", 0), "motivo": f"Duración < {min_dias} días (local)", "estado": "descartado"}); continue
        if s["velocidad"] is None or (not np.isfinite(s["velocidad"])) or s["velocidad"] >= 0:
            nuevos_descartados_global.append({"ini": ini_g + s.get("ini", 0), "fin": ini_g + s.get("fin", 0), "motivo": "Velocidad no negativa o NaN local", "estado": "descartado"}); continue
        sub = df_proc[(df_proc["Fecha"] >= s["fecha_ini"] - pd.Timedelta(days=1)) & (df_proc["Fecha"] <= s["fecha_fin"] + pd.Timedelta(days=1))] if (df_proc is not None and not df_proc.empty) else pd.DataFrame()
        medias = sub.mean(numeric_only=True) if not sub.empty else pd.Series(dtype=float)
        rd = relativedelta(s["fecha_fin"], s["fecha_ini"]); anios, meses = rd.years, rd.months
        if anios == 0 and meses == 0 and rd.days > 0: meses = 1
        if meses == 12: anios += 1; meses = 0
        nuevos_validos_global.append({"ini": ini_g + s["ini"], "fin": ini_g + s["fin"], "fecha_ini": s["fecha_ini"], "fecha_fin": s["fecha_fin"], "delta_dias": s["delta_dias"], "velocidad": s["velocidad"], "vel_abs": abs(s["velocidad"]), "medias": medias, "anios": anios, "meses": meses, "estado": "valido", "num_segmento_valido": None})
    return nuevos_validos_global, nuevos_descartados_global

def guardar_resultados(segmentos_validos, df_filtrado, y_suave, descartados, segmentos_eliminados_idx, df_proc, vars_proceso, hoja, archivo_resumen=ARCHIVO_RESUMEN):
    asegurar_carpeta(CARPETA_EXPORT); asegurar_carpeta(CARPETA_RESUMEN); asegurar_carpeta(CARPETA_PROCESADOS)
    carpeta_png = os.path.join(CARPETA_EXPORT, safe_strcol(hoja)); asegurar_carpeta(carpeta_png)
    validos = [s for s in segmentos_validos if s.get("estado","valido") == "valido"]
    validos = sorted(validos, key=lambda x: x["fecha_ini"])
    resumen_rows = []
    for idx, s in enumerate(validos, start=1):
        row = {"Segmento": idx, "Fecha inicio": s["fecha_ini"].strftime("%Y-%m-%d"), "Fecha fin": s["fecha_fin"].strftime("%Y-%m-%d"), "Años": s.get("anios",""), "Meses": s.get("meses",""), "Velocidad (mm/año)": round(s["vel_abs"], 6)}
        medias = s.get("medias")
        if medias is not None and not medias.empty:
            for var, val in medias.items(): row[var] = val
        resumen_rows.append(row)
    df_resumen = pd.DataFrame(resumen_rows)
    fig_final, ax_final = dibujar_grafica_completa(df_filtrado, y_suave, validos, descartados, segmentos_eliminados_idx, titulo=f"Velocidad de corrosión - {hoja}", figsize=FIGSIZE_DEFAULT, show=False)
    ruta_principal = os.path.join(carpeta_png, "01_principal_corrosion.png")
    try: fig_final.savefig(ruta_principal, dpi=300, bbox_inches="tight"); plt.close(fig_final)
    except Exception as e: print("Error guardando figura principal:", e)
    ruta_general_vars = None; img_paths = []
    variable_cols = [c for c in df_resumen.columns if c not in ["Segmento","Velocidad (mm/año)","Fecha inicio","Fecha fin","Años","Meses"]]
    if variable_cols:
        try:
            fig_vars, ax_vars = plt.subplots(figsize=(10,6))
            for col in variable_cols:
                ax_vars.scatter(df_resumen["Velocidad (mm/año)"], df_resumen[col], marker='o', label=col)
            ax_vars.set_xlabel("Velocidad de corrosión (mm/año)"); ax_vars.set_ylabel("Valor medio de variable"); ax_vars.set_title(f"Relación velocidad - variables - {hoja}")
            ax_vars.grid(True, alpha=0.4); ax_vars.legend(title="Variable", fontsize=8)
            ruta_general_vars = os.path.join(carpeta_png, "02_general_variables.png"); fig_vars.savefig(ruta_general_vars, dpi=300, bbox_inches="tight"); plt.close(fig_vars)
        except Exception as e: print("Error gráfico general variables:", e)
    for col in variable_cols:
        try:
            fig_v, ax_v = plt.subplots(figsize=(6,4)); ax_v.scatter(df_resumen["Velocidad (mm/año)"], df_resumen[col], marker='o')
            ax_v.set_xlabel("Velocidad (mm/año)"); ax_v.set_ylabel(col); ax_v.set_title(f"{col} vs Velocidad"); ax_v.grid(True, alpha=0.4)
            ruta_ind = os.path.join(carpeta_png, f"03_variable_{safe_strcol(col)}.png"); fig_v.savefig(ruta_ind, dpi=200, bbox_inches="tight"); plt.close(fig_v); img_paths.append(ruta_ind)
        except Exception as e: print("Error gráfico individual", col, e)
    ruta_caudal = None
    if "Caudal de línea" in df_proc.columns:
        try:
            fig_c, ax_c = plt.subplots(figsize=(10,6)); ax_c.scatter(df_proc["Fecha"], df_proc["Caudal de línea"], marker='o')
            ax_c.set_title("Caudal de línea"); ax_c.set_xlabel("Fecha"); ax_c.set_ylabel("Caudal de línea"); ax_c.grid(True, alpha=0.4)
            ruta_caudal = os.path.join(carpeta_png, "04_caudal_linea.png"); fig_c.savefig(ruta_caudal, dpi=300, bbox_inches="tight"); plt.close(fig_c)
        except Exception as e: print("Error guardando caudal:", e)
    try:
        mode = "a" if os.path.exists(archivo_resumen) else "w"
        with pd.ExcelWriter(archivo_resumen, engine="openpyxl", mode=mode) as writer:
            nombre_hoja = hoja[:31]
            if nombre_hoja in writer.book.sheetnames:
                std = writer.book[nombre_hoja]; writer.book.remove(std)
            df_resumen.to_excel(writer, sheet_name=nombre_hoja, index=False)
            wb = writer.book; ws = wb[nombre_hoja]
            fila_inicial = 2; salto = 38; fila = fila_inicial; rutas_insertar = []
            if os.path.exists(ruta_principal): rutas_insertar.append(ruta_principal)
            if ruta_general_vars and os.path.exists(ruta_general_vars): rutas_insertar.append(ruta_general_vars)
            rutas_insertar.extend([p for p in img_paths if os.path.exists(p)])
            if ruta_caudal and os.path.exists(ruta_caudal): rutas_insertar.append(ruta_caudal)
            for ruta in rutas_insertar:
                try:
                    with open(ruta, "rb") as f: img_bytes = BytesIO(f.read())
                    img_excel = XLImage(img_bytes); ws.add_image(img_excel, f"M{fila}"); fila += salto
                except Exception as e: print("No se pudo insertar imagen en Excel:", e)
    except Exception as e:
        print("Error guardando resumen en Excel:", e)
    try:
        imagenes_hoja = []
        if os.path.exists(ruta_principal): imagenes_hoja.append(ruta_principal)
        if ruta_general_vars and os.path.exists(ruta_general_vars): imagenes_hoja.append(ruta_general_vars)
        imagenes_hoja.extend([p for p in img_paths if os.path.exists(p)])
        if ruta_caudal and os.path.exists(ruta_caudal): imagenes_hoja.append(ruta_caudal)
        if imagenes_hoja:
            thumbs_w, thumbs_h = 800, 600; ncols = 3; nrows = int(np.ceil(len(imagenes_hoja) / ncols))
            collage = PILImage.new("RGB", (thumbs_w * ncols, thumbs_h * nrows), (255,255,255))
            for idx, ruta in enumerate(imagenes_hoja):
                x = (idx % ncols) * thumbs_w; y = (idx // ncols) * thumbs_h
                try:
                    im = PILImage.open(ruta); im.thumbnail((thumbs_w, thumbs_h)); bg = PILImage.new("RGB", (thumbs_w, thumbs_h), (255,255,255))
                    paste_x = (thumbs_w - im.width)//2; paste_y = (thumbs_h - im.height)//2; bg.paste(im, (paste_x, paste_y)); im.close()
                except Exception:
                    bg = PILImage.new("RGB", (thumbs_w, thumbs_h), (240,240,240)); draw = ImageDraw.Draw(bg); draw.text((10,10), "No disponible", fill=(0,0,0))
                collage.paste(bg, (x,y))
            ruta_collage = os.path.join(CARPETA_RESUMEN, f"{safe_strcol(hoja)}_collage.png"); collage.save(ruta_collage, dpi=(150,150))
    except Exception as e: print("Error creando collage:", e)
    datos_guardar = {"df_filtrado": df_filtrado, "y_suave": y_suave, "segmentos_validos": validos, "descartados": descartados, "segmentos_eliminados_idx": segmentos_eliminados_idx}
    ruta_procesado = os.path.join(CARPETA_PROCESADOS, f"{safe_strcol(hoja)}_procesado.pkl")
    try:
        with open(ruta_procesado, "wb") as f:
            pickle.dump(datos_guardar, f)
    except Exception as e:
        print("Error guardando procesado final:", e)
    print(f"Guardado: Excel='{archivo_resumen}', Imágenes='{carpeta_png}', Collage='{CARPETA_RESUMEN}', Procesado='{ruta_procesado}'")

def cargar_procesado_final(hoja):
    asegurar_carpeta(CARPETA_PROCESADOS)
    ruta_procesado = os.path.join(CARPETA_PROCESADOS, f"{safe_strcol(hoja)}_procesado.pkl")
    if os.path.exists(ruta_procesado):
        try:
            with open(ruta_procesado, "rb") as f:
                datos = pickle.load(f)
            return datos.get("df_filtrado"), datos.get("y_suave"), datos.get("segmentos_validos"), datos.get("descartados")
        except Exception as e:
            print("Error leyendo procesado final:", e)
            return None, None, [], []
    return None, None, [], []

def modo_combinado_y_grafico(archivo_corrosion=ARCHIVO_CORROSION, archivo_resumen=ARCHIVO_RESUMEN, archivo_salida=ARCHIVO_SALIDA):
    print_header("MODO: Combinar hojas y graficar (series procesadas y segmentadas)")
    if not os.path.exists(archivo_resumen):
        print(f"No se encontró el archivo resumen: {archivo_resumen}"); return
    if not os.path.exists(archivo_corrosion):
        print(f"No se encontró el archivo de corrosión: {archivo_corrosion}"); return
    try:
        xls_res = pd.ExcelFile(archivo_resumen)
    except Exception as e:
        print("Error leyendo archivo resumen:", e); return
    print("\nHojas disponibles en el archivo resumen:")
    for i, h in enumerate(xls_res.sheet_names, 1):
        print(f"  {i}. {h}")
    sel = input("\n✏️ Escribe los NÚMEROS o NOMBRES de las hojas a combinar/graficar (separados por comas): ").strip()
    partes = [p.strip() for p in sel.split(",") if p.strip()]
    hojas_sel = []
    for p in partes:
        if p.isdigit() and 1 <= int(p) <= len(xls_res.sheet_names):
            hojas_sel.append(xls_res.sheet_names[int(p)-1])
        elif p in xls_res.sheet_names:
            hojas_sel.append(p)
        else:
            print(f"⚠️ Hoja no encontrada en resumen: {p}")
    if not hojas_sel:
        print("❌ No se seleccionó ninguna hoja válida."); return
    try:
        xls_corr = pd.ExcelFile(archivo_corrosion)
    except Exception as e:
        print("Error leyendo archivo corrosión:", e); return
    tabla_vel = pd.DataFrame(columns=["Sonda/Tubería"])
    processed_series = []
    umbral_factor_def = 1.02
    umbral_def = 0.0005
    for hoja in hojas_sel:
        try:
            df_filtrado, y_suave, segmentos_validos, descartados = cargar_procesado_final(hoja)
            if df_filtrado is None or y_suave is None:
                if hoja not in xls_corr.sheet_names:
                    print(f"⚠️ La hoja '{hoja}' no está en el archivo de corrosión original."); continue
                df_raw = pd.read_excel(xls_corr, sheet_name=hoja)
                if df_raw is None or df_raw.empty:
                    print(f"⚠️ Hoja '{hoja}' vacía en corrosión."); continue
                df_filtrado, y_suave, cambios, segmentos_raw = detectar_segmentos(df_raw, umbral_factor_def, umbral_def)
                if df_filtrado is None or y_suave is None:
                    print(f"⚠️ No se pudo procesar hoja '{hoja}' (datos insuficientes después de filtrar)."); continue
                segmentos_validos, descartados = extraer_segmentos_validos(df_filtrado, y_suave, segmentos_raw, pd.DataFrame(columns=[]), [], min_dias=1)
            else:
                df_filtrado = df_filtrado.copy()
                y_suave = np.asarray(y_suave)
            processed_series.append({"hoja": hoja, "df_filtrado": df_filtrado, "y_suave": y_suave, "segmentos_validos": segmentos_validos, "descartados": descartados})
            row = {"Sonda/Tubería": hoja}
            for idx, s in enumerate(segmentos_validos, start=1):
                row[f"Segmento {idx}"] = s.get("vel_abs", np.nan)
            tabla_vel = pd.concat([tabla_vel, pd.DataFrame([row])], ignore_index=True)
            print(f"Procesada hoja: {hoja} | segmentos válidos: {len(segmentos_validos)}")
        except Exception as e:
            print(f"⚠️ Error procesando hoja '{hoja}': {e}")
    try:
        tabla_vel.to_excel(archivo_salida, index=False)
        print(f"\n✅ Archivo resumen de velocidades creado: {archivo_salida}")
    except Exception as e:
        print("Error guardando archivo de velocidades:", e)
    if not processed_series:
        print("No hay series procesadas para graficar."); return
    factor_espacio = 1.1
    min_gap = 0.6
    for s in processed_series:
        y = np.asarray(s["y_suave"])
        s["ymin"] = float(np.nanmin(y))
        s["ymax"] = float(np.nanmax(y))
        s["rango"] = s["ymax"] - s["ymin"] if (s["ymax"] - s["ymin"]) != 0 else 0.1
        s["mean"] = float(np.nanmean(y))
    processed_series = sorted(processed_series, key=lambda x: x["mean"], reverse=True)
    offsets = {}
    current_offset = 0.0
    for s in processed_series:
        gap = max(min_gap, s["rango"] * factor_espacio)
        offsets[s["hoja"]] = current_offset
        current_offset += gap
    # === DIBUJAR CURVAS CON DIVISIÓN POR SEGMENTOS (sin etiquetas ni rellenos) ===
    plt.ioff()
    fig, ax = plt.subplots(figsize=(14, 10))
    palette_series = plt.cm.get_cmap("tab10", max(2, len(processed_series)))

    for idx_s, s in enumerate(processed_series):
        hoja = s["hoja"]
        df_filtrado = s["df_filtrado"]
        y_suave = np.asarray(s["y_suave"])
        segmentos_validos = s["segmentos_validos"]
        off = offsets[hoja]
        color_base = palette_series(idx_s % palette_series.N)

        # --- Dibujar línea base en gris claro de referencia ---
        ax.plot(df_filtrado["Sent Time"], y_suave + off,
                linewidth=1.0, color="lightgray", alpha=0.6, zorder=1)

        # --- Dibujar segmentos válidos con colores (solo líneas, sin relleno) ---
        if segmentos_validos:
            cmap_seg = plt.cm.get_cmap("turbo", max(2, len(segmentos_validos)))
            for i, seg in enumerate(sorted(segmentos_validos,
                                           key=lambda x: x["fecha_ini"] if x.get("fecha_ini") is not None else pd.Timestamp.max)):
                if seg.get("estado", "valido") != "valido":
                    continue
                ini, fin = seg["ini"], seg["fin"]
                x_seg = df_filtrado["Sent Time"].iloc[ini:fin]
                y_seg = y_suave[ini:fin] + off
                color_seg = cmap_seg(i % cmap_seg.N)
                ax.plot(x_seg, y_seg, color=color_seg, linewidth=2.4, alpha=0.95, zorder=2)

        # --- Etiqueta final al extremo derecho de la curva ---
        try:
            x_last = pd.to_datetime(df_filtrado["Sent Time"].iloc[-1])
            y_last = y_suave[-1] + off
            ax.text(x_last, y_last, hoja, va="center", ha="left",
                    fontsize=9, color=color_base, fontweight="bold")
        except Exception:
            pass

    # --- Ajustes del gráfico ---
    ax.set_title("Curvas principales de corrosión (desplazadas verticalmente, no superpuestas)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("UT measurement (mm) + offset (mm)")
    ax.grid(True, alpha=0.35)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter(FMT_FECHA_EJE))
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center", fontsize=9)

    # --- Ajuste de límites verticales ---
    ymins = [s["ymin"] + offsets[s["hoja"]] for s in processed_series]
    ymaxs = [s["ymax"] + offsets[s["hoja"]] for s in processed_series]
    total_span = max(ymaxs) - min(ymins) if (max(ymaxs) - min(ymins)) != 0 else 1.0
    ax.set_ylim(min(ymins) - 0.05 * total_span, max(ymaxs) + 0.05 * total_span)

    # --- Leyenda con los nombres de las hojas ---
    try:
        labels = [s["hoja"] for s in processed_series]
        handles = [plt.Line2D([0], [0], color=palette_series(i % palette_series.N), lw=2)
                   for i in range(len(labels))]
        ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.02, 1.0),
                  fontsize=9, title="Hojas")
    except Exception:
        pass

    plt.tight_layout()
    try:
        plt.show()
    except Exception:
        plt.show(block=True)
    plt.ion()

    # --- Guardar figura ---
    asegurar_carpeta(CARPETA_EXPORT)
    nombre_png = f"curvas_segmentadas_sin_texto_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    ruta_png = os.path.join(CARPETA_EXPORT, nombre_png)
    try:
        fig.savefig(ruta_png, dpi=300, bbox_inches="tight")
        print(f"Figura guardada en: {ruta_png}")
    except Exception as e:
        print("No se pudo guardar la figura:", e)
    plt.close(fig)
    print("Modo combinado y graficado (segmentos sin texto ni relleno) completado.")

    plt.tight_layout()
    try:
        plt.show()
    except Exception:
        plt.show(block=True)
    plt.ion()
    asegurar_carpeta(CARPETA_EXPORT)
    nombre_png = f"curvas_principales_segmentadas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    ruta_png = os.path.join(CARPETA_EXPORT, nombre_png)
    try:
        fig.savefig(ruta_png, dpi=300, bbox_inches="tight")
        print(f"Figura guardada en: {ruta_png}")
    except Exception as e:
        print("No se pudo guardar la figura:", e)
    try:
        resp_seg = input("\n¿Deseas realizar un análisis segmentado por intervalos personalizados? (s/n): ").strip().lower()
        if resp_seg == "s":
            df_proc, vars_proceso = None, []
            if os.path.exists(ARCHIVO_PROCESO):
                try:
                    df_proc, vars_proceso = cargar_datos_proceso(ARCHIVO_PROCESO)
                except Exception as e:
                    print("⚠️ No se pudieron cargar datos de proceso:", e)
            analisis_segmentado_manual(processed_series, df_proc, vars_proceso)
    except Exception as e:
        print("⚠️ Error en análisis segmentado:", e)

    plt.close(fig)
    print("Modo combinado y graficado completado.")

def solicitar_intervalos_por_consola():
    """
    Pide por consola uno o varios intervalos de fechas.
    Formato fechas: YYYY-MM-DD
    Salir con una línea vacía.
    Devuelve lista de tuplas (fecha_inicio, fecha_fin) como pd.Timestamp.
    """
    print("\n--- Selección de intervalos para análisis segmentado ---")
    print("Introduce intervalos (fecha_inicio fecha_fin) en formato YYYY-MM-DD")
    print("Ejemplo: 2023-01-01 2023-06-30")
    print("Deja la línea vacía para terminar.\n")
    intervalos = []
    while True:
        linea = input("Intervalo (inicio fin) o ENTER para terminar: ").strip()
        if not linea:
            break
        partes = linea.split()
        if len(partes) != 2:
            print("Formato inválido. Debes escribir: YYYY-MM-DD YYYY-MM-DD")
            continue
        try:
            fi = pd.to_datetime(partes[0])
            ff = pd.to_datetime(partes[1])
            if pd.isna(fi) or pd.isna(ff) or ff <= fi:
                print("Fechas inválidas o fin <= inicio. Intenta de nuevo.")
                continue
            intervalos.append((fi, ff))
            print(f"Intervalo añadido: {fi.strftime(FMT_FECHA_EJE)} → {ff.strftime(FMT_FECHA_EJE)}")
        except Exception as e:
            print("Error parseando fechas:", e)
    if not intervalos:
        print("No se han introducido intervalos.")
    return intervalos


def analizar_segmentos(processed_series, df_proc, vars_proceso, intervalos, carpeta_export=None):
    """
    Para cada intervalo (fi, ff) hace:
      - Una figura con todas las series superpuestas en ese tramo (sin offsets).
      - resume en un excel "analisis_segmentado.xlsx" la media de las variables de proceso
        (df_proc) para cada hoja dentro del intervalo.
    processed_series: lista como la que genera modo_combinado_y_grafico (cada item: hoja, df_filtrado, y_suave, ...).
    df_proc: DataFrame de procesos con columna "Fecha".
    vars_proceso: lista de nombres de variables de proceso (columnas en df_proc) - puede ser [].
    """
    if carpeta_export is None:
        carpeta_export = os.path.join(CARPETA_EXPORT, "segmentado")
    asegurar_carpeta(carpeta_export)

    excel_rows = []
    for idx_int, (fi, ff) in enumerate(intervalos, start=1):
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.set_title(f"Intervalo {idx_int}: {fi.strftime(FMT_FECHA_EJE)} → {ff.strftime(FMT_FECHA_EJE)}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Fecha")
        ax.set_ylabel("UT measurement (mm)")
        ax.grid(True, alpha=0.35)
        plotted_any = False

        # Colormap para distinguir series
        cmap = plt.cm.get_cmap("tab20", max(2, len(processed_series)))

        for i_s, s in enumerate(processed_series):
            hoja = s["hoja"]
            df_filtrado = s["df_filtrado"].copy()
            # Nos quedamos solo con el tramo fi..ff en la serie original
            try:
                df_filtrado["Sent Time"] = pd.to_datetime(df_filtrado["Sent Time"])
            except Exception:
                pass
            mask = (df_filtrado["Sent Time"] >= fi) & (df_filtrado["Sent Time"] <= ff)
            sub = df_filtrado.loc[mask]
            # si tenemos y_suave (misma longitud que df_filtrado), extraer los índices correspondientes
            y_suave = np.asarray(s.get("y_suave") if s.get("y_suave") is not None else [])
            if sub.empty:
                continue
            plotted_any = True
            # Intentar alinear índices: localizar posiciones por Sent Time en df_filtrado original
            try:
                # Usamos index positions de df_filtrado para cortar y_suave
                idxs = sub.index.values
                y_segment = y_suave[idxs] if (len(y_suave) == len(df_filtrado)) else sub["UT measurement (mm)"].values
            except Exception:
                y_segment = sub["UT measurement (mm)"].values
            color = cmap(i_s % cmap.N)
            ax.plot(sub["Sent Time"], y_segment, label=hoja, linewidth=1.6, color=color, alpha=0.9)

            # --- Calcular medias de variables de proceso para este intervalo y hoja ---
            medias = {}
            if (df_proc is not None) and (not df_proc.empty):
                try:
                    sub_proc = df_proc[(df_proc["Fecha"] >= fi) & (df_proc["Fecha"] <= ff)]
                    if not sub_proc.empty:
                        medias = sub_proc.mean(numeric_only=True).to_dict()
                    else:
                        medias = {v: np.nan for v in vars_proceso} if vars_proceso else {}
                except Exception:
                    medias = {v: np.nan for v in vars_proceso} if vars_proceso else {}
            # Montar fila para excel
            fila = {"Intervalo": idx_int, "Fecha inicio": fi.strftime(FMT_FECHA_EJE), "Fecha fin": ff.strftime(FMT_FECHA_EJE), "Hoja": hoja}
            for k, val in medias.items():
                fila[k] = val
            excel_rows.append(fila)

        if not plotted_any:
            print(f"Intervalo {idx_int} ({fi.date()}→{ff.date()}): no hay datos en las series seleccionadas.")
            plt.close(fig)
            continue

        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter(FMT_FECHA_EJE))
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        plt.tight_layout()

        # Guardar figura del intervalo
        ruta_fig = os.path.join(carpeta_export, f"segmento_{idx_int}_{fi.strftime('%Y%m%d')}_{ff.strftime('%Y%m%d')}.png")
        try:
            fig.savefig(ruta_fig, dpi=300, bbox_inches="tight")
            print(f"Guardada figura intervalo {idx_int}: {ruta_fig}")
        except Exception as e:
            print(f"No se pudo guardar figura intervalo {idx_int}:", e)
        plt.close(fig)

    # --- Guardar excel resumen ---
    if excel_rows:
        df_excel = pd.DataFrame(excel_rows)
        ruta_excel = os.path.join(carpeta_export, "analisis_segmentado.xlsx")
        try:
            # Reordenar columnas: Intervalo, Fecha inicio, Fecha fin, Hoja, luego variables
            cols = ["Intervalo", "Fecha inicio", "Fecha fin", "Hoja"] + [c for c in df_excel.columns if c not in ["Intervalo", "Fecha inicio", "Fecha fin", "Hoja"]]
            df_excel = df_excel[cols]
            df_excel.to_excel(ruta_excel, index=False)
            print(f"Excel de análisis segmentado guardado en: {ruta_excel}")
        except Exception as e:
            print("Error guardando Excel de análisis segmentado:", e)
    else:
        print("No se generó Excel de análisis segmentado: no hay filas resultantes.")

    print("Análisis segmentado completado.")


def analisis_segmentado_manual(processed_series, df_proc=None, vars_proceso=None):
    """
    Interfaz combinada: solicita intervalos y lanza el análisis.
    processed_series: lista de series (como en modo_combinado_y_grafico).
    df_proc: DataFrame de proceso (opcional).
    vars_proceso: lista de nombres de variables de proceso (opcional).
    """
    if vars_proceso is None:
        vars_proceso = df_proc.columns.drop("Fecha").tolist() if (df_proc is not None and "Fecha" in df_proc.columns) else []
    intervalos = solicitar_intervalos_por_consola()
    if not intervalos:
        print("No hay intervalos. Saliendo.")
        return
    analizar_segmentos(processed_series, df_proc, vars_proceso, intervalos)


def revisar_segmentos_interactivamente(df_filtrado, y_suave, segmentos_validos, descartados, df_proc, vars_proceso, figsize=FIGSIZE_DEFAULT):
    segmentos_eliminados_idx = []
    segmentos_validos = sorted(segmentos_validos, key=lambda x: x["fecha_ini"] if x.get("fecha_ini") is not None else pd.Timestamp.max)
    while True:
        fig, ax = dibujar_grafica_completa(df_filtrado, y_suave, segmentos_validos, descartados, segmentos_eliminados_idx, titulo="Revisión de segmentos", figsize=figsize, show=True)
        validos_act = [s for s in segmentos_validos if s.get("estado","valido") == "valido"]
        if not validos_act:
            resp_no = input("No quedan segmentos válidos. Guardar (s) / volver a global (g) / salir (q): ").strip().lower()
            plt.close(fig)
            return segmentos_validos, descartados, segmentos_eliminados_idx
        print("\nSegmentos válidos actuales:")
        for s in validos_act:
            num = s.get("num_segmento_valido") or "?"
            print(f" {num}. {s['fecha_ini'].strftime('%Y-%m-%d')} → {s['fecha_fin'].strftime('%Y-%m-%d')}  |  Vel: {s['vel_abs']:.5f} mm/año")
        resp_todos = input("\n¿Estás conforme con todos los segmentos? (s/n): ").strip().lower()
        if resp_todos == 's':
            plt.close(fig); return segmentos_validos, descartados, segmentos_eliminados_idx
        sel = input("Número del segmento a revisar (o 'salir'): ").strip()
        if sel.lower() in ['salir','q','exit']: plt.close(fig); return segmentos_validos, descartados, segmentos_eliminados_idx
        try: sel_int = int(sel)
        except Exception: print("Entrada inválida"); plt.close(fig); continue
        elegido = None
        for s in segmentos_validos:
            if s.get("num_segmento_valido") == sel_int and s.get("estado","valido") == "valido": elegido = s; break
        if elegido is None:
            print("No encontrado. Usa el número mostrado."); plt.close(fig); continue
        print(f"Seleccionado {sel_int}: {elegido['fecha_ini'].strftime('%Y-%m-%d')} → {elegido['fecha_fin'].strftime('%Y-%m-%d')}")
        op = input("E - Eliminar | R - Recalcular | C - Cancelar : ").strip().lower()
        if op == 'c': plt.close(fig); continue
        if op == 'e':
            elegido["estado"] = "eliminado"; segmentos_eliminados_idx.append((elegido["ini"], elegido["fin"])); print(f"Segmento {sel_int} eliminado."); plt.close(fig); continue
        if op == 'r':
            try: nuevo_umbral = float(input("Nuevo umbral local (ej 0.0005): ").strip())
            except Exception: print("Umbral inválido"); plt.close(fig); continue
            cambiar = input("Cambiar umbral_factor local? (s/n): ").strip().lower(); nuevo_umbral_factor = None
            if cambiar == 's':
                try: nuevo_umbral_factor = float(input("Nuevo umbral_factor local (ej 1.02): ").strip())
                except Exception: print("umbral_factor inválido; se ignorará")
            nuevos_validos, nuevos_descartados = recalcular_segmento_local(df_filtrado, y_suave, elegido, df_proc, vars_proceso, nuevo_umbral, nuevo_umbral_factor)
            elegido["estado"] = "reemplazado"; segmentos_eliminados_idx.append((elegido["ini"], elegido["fin"]))
            for d in nuevos_descartados:
                if d not in descartados: descartados.append(d)
            segmentos_validos = [s for s in segmentos_validos if s is not elegido]
            segmentos_validos.extend(nuevos_validos)
            segmentos_validos = sorted(segmentos_validos, key=lambda x: x["fecha_ini"] if x.get("fecha_ini") is not None else pd.Timestamp.max)
            for s in segmentos_validos:
                s["ini"] = int(s["ini"]); s["fin"] = int(s["fin"])
            print(f"Recalculado. Añadidos {len(nuevos_validos)} nuevos segmentos (si los hubo).")
            plt.close(fig); continue
        print("Opción no válida."); plt.close(fig); continue

def main():
    print_header("ANÁLISIS DE CORROSIÓN - INTEGRADO")
    print("Opciones:")
    print("  1. Análisis completo de segmentos y velocidades (flujo interactivo)")
    print("  2. Combinar hojas y graficar curvas finales procesadas (segmentadas) + generar excel de velocidades")
    print("  3. Salir")
    opcion = input("Elige una opción (1/2/3) [1]: ").strip() or "1"
    if opcion == "3":
        print("Saliendo."); return
    if opcion == "2":
        modo_combinado_y_grafico(archivo_corrosion=ARCHIVO_CORROSION, archivo_resumen=ARCHIVO_RESUMEN, archivo_salida=ARCHIVO_SALIDA)
        return
    if not os.path.exists(ARCHIVO_CORROSION):
        print("Archivo corrosión no encontrado:", ARCHIVO_CORROSION); return
    if not os.path.exists(ARCHIVO_PROCESO):
        print("Archivo proceso no encontrado:", ARCHIVO_PROCESO); return
    try:
        df_proc, vars_proceso = cargar_datos_proceso(ARCHIVO_PROCESO)
    except Exception as e:
        print("Error cargando datos proceso:", e); return
    try:
        xls = pd.ExcelFile(ARCHIVO_CORROSION)
    except Exception as e:
        print("Error leyendo libro corrosión:", e); return
    print("\nHojas disponibles:")
    for i, h in enumerate(xls.sheet_names, start=1): print(f" {i}. {h}")
    sel = input("\nEscribe NOMBRE o NUMERO de la hoja: ").strip()
    hoja = xls.sheet_names[int(sel)-1] if sel.isdigit() else sel
    print("Hoja seleccionada:", hoja)
    try:
        df = pd.read_excel(ARCHIVO_CORROSION, sheet_name=hoja)
    except Exception as e:
        print("No se pudo leer la hoja:", e); return
    try:
        col_fecha, col_espesor = detect_columns(df)
    except Exception as e:
        print("No se pudo detectar columnas en la hoja seleccionada:", e); return
    df["Sent Time"] = pd.to_datetime(df[col_fecha], errors="coerce")
    df["UT measurement (mm)"] = pd.to_numeric(df[col_espesor], errors="coerce")
    if not {"Sent Time","UT measurement (mm)"}.issubset(df.columns):
        print("Faltan columnas 'Sent Time' o 'UT measurement (mm)'"); return
    umbral_factor = 1.02; umbral = 0.0005; figsize = FIGSIZE_DEFAULT
    while True:
        df_filtrado, y_suave, cambios, segmentos_raw = detectar_segmentos(df, umbral_factor, umbral)
        if df_filtrado is None or y_suave is None:
            print("No se pudo detectar segmentos. Ajusta umbrales o revisa datos."); return
        segmentos_validos, descartados = extraer_segmentos_validos(df_filtrado, y_suave, segmentos_raw, df_proc, vars_proceso)
        fig_main, ax_main = dibujar_grafica_completa(df_filtrado, y_suave, segmentos_validos, descartados, [], titulo=f"Velocidad de corrosión - {hoja}", figsize=figsize, show=True)
        resp = input("\n¿Guardar este ajuste general? (s/n): ").strip().lower()
        if resp != 's':
            try:
                umbral_factor = float(input(f"Nuevo umbral_factor (actual {umbral_factor}): ") or umbral_factor)
                umbral = float(input(f"Nuevo umbral (actual {umbral}): ") or umbral)
                ancho = float(input(f"Nuevo ancho figura (actual {figsize[0]}): ") or figsize[0])
                alto = float(input(f"Nuevo alto figura (actual {figsize[1]}): ") or figsize[1])
                figsize = (ancho, alto)
            except Exception:
                print("Entrada inválida, manteniendo parámetros.")
            plt.close(fig_main); continue
        plt.close(fig_main)
        segmentos_validos, descartados, segmentos_eliminados_idx = revisar_segmentos_interactivamente(df_filtrado, y_suave, segmentos_validos, descartados, df_proc, vars_proceso, figsize=figsize)
        while True:
            fig_final, ax_final = dibujar_grafica_completa(df_filtrado, y_suave, segmentos_validos, descartados, segmentos_eliminados_idx, titulo=f"Resultado final - {hoja}", figsize=figsize, show=True)
            resp_final = input("\n¿Estás conforme con todos y deseas guardar? (s/n): ").strip().lower()
            if resp_final == 's':
                plt.close(fig_final)
                guardar_resultados(segmentos_validos, df_filtrado, y_suave, descartados, segmentos_eliminados_idx, df_proc, vars_proceso, hoja, archivo_resumen=ARCHIVO_RESUMEN)
                print("Finalizado."); return
            plt.close(fig_final)
            sub = input("Revisar más segmentos (r) / Reajustar global (g) / Salir sin guardar (q): ").strip().lower()
            if sub == 'r':
                segmentos_validos, descartados, segmentos_eliminados_idx = revisar_segmentos_interactivamente(df_filtrado, y_suave, segmentos_validos, descartados, df_proc, vars_proceso, figsize=figsize)
                continue
            elif sub == 'g':
                break
            elif sub in ['q','salir','exit']:
                print("Saliendo sin guardar."); return
            else:
                print("Opción no reconocida.")
        continue

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrumpido por usuario.")
    except Exception as e:
        mostrar_error(e, "Error inesperado en main")

