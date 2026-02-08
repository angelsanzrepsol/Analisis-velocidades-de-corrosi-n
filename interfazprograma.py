
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


import zipfile
import pandas as pd
import io

import re

import json

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
    comp_cols = [c for c in df.columns if "comp" in c.lower() and "porc" not in c.lower()]
    porc_cols = [c for c in df.columns if "porccomp" in c.lower()]

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

            resumen = (
                sub.groupby("Especie")["Porcentaje"]
                .mean()
                .reset_index()
            )

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

            resumen = (
                sub.groupby("Especie")["Porcentaje"]
                .mean()
                .reset_index()
            )

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
    fila_inicio = None
    for i in range(len(df_raw)):
        fila = df_raw.iloc[i]
        # Si alguna celda es numérica → esta fila es inicio
        if fila.apply(lambda x: isinstance(x, (int,float)) or str(x).replace('.', '', 1).isdigit()).any():
            fila_inicio = i
            break

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

    # Convertir todo lo posible a numérico
    for c in df.columns:
        if c != "Fecha":
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

def aplicar_segmentacion_referencia(
        df_filtrado,
        segmentos_ref,
        df_proc,
        vars_proceso,
        min_dias=5
):
    nuevos_segmentos = []

    for ref in segmentos_ref:

        fi = ref["fecha_ini"]
        ff = ref["fecha_fin"]

        sub_df = df_filtrado[
            (df_filtrado["Sent Time"] >= fi) &
            (df_filtrado["Sent Time"] <= ff)
        ]

        if sub_df.empty:
            continue

        y = sub_df["UT measurement (mm)"].values
        delta = (ff - fi).days

        if delta < min_dias:
            continue

        velocidad = (y[-1] - y[0]) / (delta / 365.25)

        medias = {}
        if df_proc is not None:
            sub_proc = df_proc[
                (df_proc["Fecha"] >= fi) &
                (df_proc["Fecha"] <= ff)
            ]
            medias = sub_proc.mean(numeric_only=True)

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

    return nuevos_segmentos


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
if uploaded_mpa is not None:

    try:

        df_mpa = None

        # Intento 1 → openpyxl
        try:
            df_mpa = pd.read_excel(uploaded_mpa, engine="openpyxl")
        except:
            pass

        # Intento 2 → xlrd
        if df_mpa is None:
            try:
                uploaded_mpa.seek(0)
                df_mpa = pd.read_excel(uploaded_mpa, engine="xlrd")
            except:
                pass

        # Intento 3 → lectura automática pandas
        if df_mpa is None:
            try:
                uploaded_mpa.seek(0)
                df_mpa = pd.read_excel(uploaded_mpa)
            except:
                pass

        if df_mpa is None:
            st.sidebar.error("No se pudo leer el archivo MPA. Revisa formato Excel.")
        else:
            df_mpa.columns = [str(c).strip() for c in df_mpa.columns]
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

def buscar_velocidad_mpa(df_mpa, temp, tan, material):

    if df_mpa is None or pd.isna(temp) or pd.isna(tan):
        return None

    col_temp = "Temperature"
    col_tan = "Acid measurement"
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
tabs = st.tabs([
    "Procesar hoja",
    "Combinar hojas",
    "Revisión / Guardado",
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
    material_sel = st.radio(
        "Material",
        ["Carbon Steel", "5 Cr"],
        horizontal=True
    )

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
    df_mpa = st.session_state.get("df_mpa")
    processed = {
        k: v for k, v in st.session_state.get("processed_sheets", {}).items()
        if v.get("saved")
    }
    
    if len(processed) == 0:
    
        st.info("No hay sondas guardadas aún.")
    
    else:
    
        # -----------------------------
        # Elegir base de segmentación
        # -----------------------------
    
        if "segmentacion_referencia" in st.session_state:
            segmentos_base = st.session_state["segmentacion_referencia"]
        else:
            # usar primera sonda guardada
            primera = list(processed.values())[0]
            segmentos_base = primera["segmentos_validos"]
    
        filas = []

        for i, seg_base in enumerate(segmentos_base, start=1):
        
            fila = {"Segmento": f"Seg {i}"}
        
            # -------- VELOCIDAD ESPERADA --------
        
            vel_esperada = None
            df_mpa = st.session_state.get("df_mpa")
        
            if df_mpa is not None:
        
                medias = seg_base.get("medias", {})
        
                temp = None
                tan = None
                
                if isinstance(medias, (dict, pd.Series)):
                
                    for k,v in medias.items():
                
                        if "temp" in k.lower():
                            temp = v
                
                        if "tan" in k.lower():
                            tan = v

                vel_esperada = buscar_velocidad_mas_cercana(
                    df_mpa,
                    temp,
                    tan,
                    material_sel
                )
        
            fila["Velocidad esperada"] = vel_esperada
        
            # -------- VELOCIDADES SONDAS --------
        
            fi_ref = pd.to_datetime(seg_base["fecha_ini"])
            ff_ref = pd.to_datetime(seg_base["fecha_fin"])
        
            for key, data in processed.items():
        
                nombre_sonda = f"{data['source_name']} | {data['hoja']}"
        
                vel = None
        
                for seg in data["segmentos_validos"]:
        
                    fi = pd.to_datetime(seg["fecha_ini"])
                    ff = pd.to_datetime(seg["fecha_fin"])
        
                    if fi == fi_ref and ff == ff_ref:
                        vel = seg.get("vel_abs")
                        break
        
                fila[nombre_sonda] = vel
        
            filas.append(fila)

    
            fi_ref = pd.to_datetime(seg_base["fecha_ini"])
            ff_ref = pd.to_datetime(seg_base["fecha_fin"])
    
            # recorrer sondas
            for key, data in processed.items():
    
                nombre_sonda = f"{data['source_name']} | {data['hoja']}"
    
                vel = None
    
                for seg in data["segmentos_validos"]:
    
                    fi = pd.to_datetime(seg["fecha_ini"])
                    ff = pd.to_datetime(seg["fecha_fin"])
    
                    # ✔ coincidencia exacta
                    if fi == fi_ref and ff == ff_ref:
                        vel = seg.get("vel_abs")
                        break
    
                fila[nombre_sonda] = vel
    
            filas.append(fila)
    
        df_comp = pd.DataFrame(filas)
    
        st.dataframe(df_comp)
    
        # Exportar
        buffer = io.BytesIO()
        df_comp.to_excel(buffer, index=False)
        buffer.seek(0)
    
        st.download_button(
            "Descargar comparativa",
            buffer,
            file_name="comparativa_sondas.xlsx"
        )

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
# -------------------- TAB 4: CRUDOS --------------------
with tabs[3]:

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
    
        if df_master.empty:
            st.warning("No hay coincidencias con segmentos")
            st.stop()
    
        # =============================
        # BLOQUE 1 — ANALISIS SEGMENTOS
        # =============================
    
        st.subheader("Crudos y variables por segmento")
    
        st.dataframe(df_master)
    
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
