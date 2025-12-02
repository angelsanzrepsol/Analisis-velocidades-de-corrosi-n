import zipfile
import pandas as pd
import os
import io

def zip_csvs_to_excel(zip_path, excel_path):
    """
    Convierte cualquier ZIP (carpetas internas, csv, txt, archivos sin extensión)
    en un Excel donde cada archivo separado por comas va en una hoja.
    """

    with zipfile.ZipFile(zip_path, 'r') as z:

        # 1) obtener lista completa de archivos válidos
        files = [
            f for f in z.namelist()
            if not f.endswith("/")  # descartar carpetas
        ]

        if not files:
            raise ValueError("El ZIP no contiene archivos válidos")

        # 2) crear excel
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:

            for file_name in files:
                try:
                    # leer archivo del ZIP
                    raw = z.read(file_name)

                    # intentar leer como CSV con autodetección
                    df = pd.read_csv(
                        io.BytesIO(raw),
                        sep=None,
                        engine="python",
                        encoding="latin1",
                        error_bad_lines=False,
                        warn_bad_lines=False
                    )

                except Exception:
                    # si falla, intentar con punto y coma
                    try:
                        df = pd.read_csv(
                            io.BytesIO(raw),
                            sep=";",
                            engine="python",
                            encoding="latin1",
                            error_bad_lines=False,
                            warn_bad_lines=False
                        )
                    except Exception:
                        # si aun así falla → hoja de error
                        df = pd.DataFrame(
                            {"ERROR": [f"No se pudo leer el archivo {file_name} como CSV"]}
                        )

                # nombre seguro para la hoja
                sheet_name = os.path.basename(file_name).replace(".csv", "").replace(".txt", "")
                if sheet_name == "":
                    sheet_name = "archivo"
                sheet_name = sheet_name[:31]  # límite Excel

                df.to_excel(writer, index=False, sheet_name=sheet_name)
