
# streamlit_app_final.py (renamed fixed for usuario: interfazprograma_fixed.py)
"""
Streamlit — Analizador de corrosión (versión final corregida)

Instrucciones:
    streamlit run interfazprograma_fixed.py

Este archivo es una versión limpiada y funcional del script que subiste:
- Elimina bloques duplicados que causaban NameError en corr_path
- Añade manejo robusto de archivos subidos y procesos temporales
- Usa fallbacks si no hay funciones del usuario
- Mantiene la UI y funcionalidades principales
"""

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

# Configuración básica y estilo
st.set_page_config(page_title="Analizador de corrosión", layout="wide")
st.markdown("<h1 class='darkblue-title'>Analísis de corrosión</h1>", unsafe_allow_html=True)

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
uploaded_corr = st.sidebar.file_uploader("Archivo de corrosión (.xlsx)", type=["xlsx"], key="file_uploader_corr")
uploaded_proc = st.sidebar.file_uploader("Archivo de proceso (.xlsx) — opcional", type=["xlsx"], key="file_uploader_proc")

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
        xls = pd.ExcelFile(uploaded_file)
        return xls.sheet_names
    except Exception:
        return []

@st.cache_data(show_spinner=False)
def cached_read_excel_sheet_df(uploaded_file, sheet_name):
    if uploaded_file is None:
        return pd.DataFrame()
    try:
        df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
        df.columns = [str(c) for c in df.columns]
        return df
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
    wl = min(wl_max, (len(y) - 1) if (len(y) % 2 == 0) else len(y))
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
                medias = sub.mean(numeric_only=True)
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

def extraer_segmentos_validos_wrapper(df_filtrado, y_suave, segmentos_raw, df_proc, vars_proceso, min_dias_val):
    fn = safe_get("extraer_segmentos_validos")
    if fn is not None:
        try:
            return fn(df_filtrado, y_suave, segmentos_raw, df_proc, vars_proceso, min_dias=min_dias_val)
        except Exception:
            pass
    return extraer_segmentos_validos_fallback(df_filtrado, y_suave, segmentos_raw, df_proc, vars_proceso, min_dias=min_dias_val)

def dibujar_grafica_completa_wrapper(df_filtrado, y_suave, segmentos_validos, descartados, segmentos_eliminados_idx, titulo, figsize, show=False):
    fn = safe_get("dibujar_grafica_completa")
    if fn is not None:
        try:
            return fn(df_filtrado, y_suave, segmentos_validos, descartados, segmentos_eliminados_idx, titulo=titulo, figsize=figsize, show=show)
        except Exception:
            pass
    return dibujar_grafica_completa_fallback(df_filtrado, y_suave, segmentos_validos, descartados, segmentos_eliminados_idx, titulo=titulo, figsize=figsize, show=show)

def recalcular_segmento_local_wrapper(df_filtrado, y_suave, segmento, df_proc, vars_proceso, nuevo_umbral, nuevo_umbral_factor=None, min_dias=10):
    fn = safe_get("recalcular_segmento_local")
    if fn is not None:
        try:
            return fn(df_filtrado, y_suave, segmento, df_proc, vars_proceso, nuevo_umbral, nuevo_umbral_factor, min_dias=min_dias)
        except Exception:
            pass
    return recalcular_segmento_local_fallback(df_filtrado, y_suave, segmento, df_proc, vars_proceso, nuevo_umbral, nuevo_umbral_factor, min_dias)

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

# -------------------- Pestañas UI --------------------
tabs = st.tabs(["Procesar hoja", "Combinar hojas", "Revisión / Guardado"])

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
            df_proc = pd.read_excel(tmp_proc_path)
            vars_proceso = [c for c in df_proc.columns if "fecha" not in c.lower()]

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
        if corr_path is not None:
            try:
                xls_corr = pd.ExcelFile(corr_path)
                hojas = xls_corr.sheet_names
            except Exception as e:
                st.error(f"No se pudieron leer las hojas del archivo: {e}")
                hojas = []
        
        if not hojas:
            st.warning("No se encontraron hojas en el archivo subido.")
        else:
            hoja_sel = st.selectbox("Selecciona hoja", options=hojas)
        
            try:
                df_original = pd.read_excel(corr_path, sheet_name=hoja_sel)
                df_original.columns = [str(c) for c in df_original.columns]
                st.success(f"Hoja cargada: {hoja_sel} — filas: {len(df_original)}")
            except Exception as e:
                st.error(f"No se pudo leer la hoja seleccionada: {e}")
                df_original = pd.DataFrame()

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
                        seg_list = []
                        try:
                            seg_list = [f"{i+1}: {s.get('fecha_ini')} → {s.get('fecha_fin')}  | Vel: {s.get('vel_abs')}" for i,s in enumerate(st.session_state["processed_sheets"][key]["segmentos_validos"])]
                        except Exception:
                            seg_list = []

                        if seg_list:
                            sel_idx = st.selectbox("Selecciona segmento", options=list(range(1, len(seg_list)+1)), format_func=lambda x: seg_list[x-1], key=f"selseg_{key}")

                            colA, colB, colC = st.columns(3)

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
                                    pkl_path = out_dir / f"{data_now['source_name']}_{data_now['hoja']}_procesado.pkl"
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
                                        img_dir = Path.cwd() / "graficos_exportados"
                                        img_dir.mkdir(exist_ok=True)
                                        figpath = img_dir / f"{data_now['source_name']}_{data_now['hoja']}_grafica.png"
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
    if "processed_sheets" in st.session_state and st.session_state["processed_sheets"]:
        opciones = list(st.session_state["processed_sheets"].keys())
        sel_key = st.selectbox("Selecciona hoja procesada", options=opciones)
        datos = st.session_state["processed_sheets"][sel_key]
        segs = datos.get("segmentos_validos", [])

        df_medias = pd.DataFrame([
            {"Segmento": i + 1, "Velocidad (mm/año)": s.get("vel_abs"), **(s.get("medias", {}))}
            for i, s in enumerate(segs)
            if s.get("estado") == "valido"
        ])

        if df_medias.empty:
            st.info("No hay datos de medias por segmento para mostrar.")
        else:
            st.subheader("Medias por segmento")
            st.dataframe(df_medias)

            columnas_vars = [c for c in df_medias.columns if c not in ["Segmento", "Velocidad (mm/año)"]]
            if columnas_vars:
                var_sel = st.selectbox("Variable de proceso a graficar:", options=columnas_vars)
                st.markdown(f"**Gráfica: {var_sel} vs Velocidad (mm/año)**")
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.scatter(df_medias["Velocidad (mm/año)"], df_medias[var_sel], alpha=0.7)
                ax.set_xlabel("Velocidad de corrosión (mm/año)")
                ax.set_ylabel(var_sel)
                ax.grid(True, alpha=0.4)
                st.pyplot(fig)
    else:
        st.info("No hay hojas procesadas aún. Procesa primero una hoja en la pestaña 'Procesar hoja'.")

    saved_list = [k for k,v in st.session_state.get("processed_sheets", {}).items() if v.get("saved")]
    if not saved_list:
        st.info("No hay procesados guardados en sesión.")
    else:
        choice = st.selectbox("Selecciona procesado guardado", options=saved_list, format_func=lambda x: f"{st.session_state['processed_sheets'][x]['source_name']} | {st.session_state['processed_sheets'][x]['hoja']}")
        data = st.session_state['processed_sheets'][choice]
        st.subheader(f"{data['source_name']} | {data['hoja']}")
        img_dir = Path.cwd() / "graficos_exportados"
        img_file = img_dir / f"{data['source_name']}_{data['hoja']}_grafica.png"
        col1, col2 = st.columns([2,1])
        with col1:
            if img_file.exists():
                st.image(str(img_file), caption="Gráfica guardada (definitiva)")
            else:
                try:
                    fig, ax = dibujar_grafica_completa_wrapper(data['df_filtrado'], data['y_suave'], data['segmentos_validos'], data['descartados'], [], titulo=f"{data['hoja']}", figsize=(fig_w, fig_h), show=False)
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"No se pudo mostrar gráfica: {e}")
        with col2:
            st.markdown("### Resumen y acciones")
            st.write(f"Segmentos válidos: {len(data['segmentos_validos'])} — Descartados: {len(data['descartados'])}")
            try:
                mean_ut = float(np.nanmean(data['df_filtrado']['UT measurement (mm)']))
                st.metric(label="Media UT (mm)", value=f"{mean_ut:.4f}")
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

        st.markdown("### Tabla definitiva — medias por segmento (si hay datos de proceso)")
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
        if rows:
            df_rows = pd.DataFrame(rows)
            st.dataframe(df_rows)
        if 'df_rows' in locals() and not df_rows.empty:
            st.write("### Medias de variables de proceso por segmento")
            columnas_medias = [c for c in df_rows.columns if c not in ['Segmento', 'Inicio', 'Fin', 'Días', 'Vel (mm/año)']]
            if columnas_medias:
                st.dataframe(df_rows[columnas_medias].round(4))
            else:
                st.info("No se encontraron variables de proceso en los segmentos.")
        else:
            st.write("No hay segmentos válidos para este procesado.")

# -------------------- Footer --------------------
st.markdown("---")
if user_module_path is not None:
    st.caption(f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} — Módulo usuario (si aplicable): {getattr(user_module_path,'name', str(user_module_path))}")
else:
    st.caption(f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
