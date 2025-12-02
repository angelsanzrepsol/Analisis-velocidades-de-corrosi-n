# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 06:35:12 2025

@author: angel
"""

import zipfile
import pandas as pd

def zip_csvs_to_excel(zip_path, excel_output_path):
    # Abre el archivo ZIP
    with zipfile.ZipFile(zip_path, 'r') as z:
        # Lista los archivos CSV dentro del ZIP
        csv_files = [f for f in z.namelist() if f.lower().endswith(".csv")]
        
        # Crea un escritor de Excel
        with pd.ExcelWriter(excel_output_path, engine="openpyxl") as writer:
            for csv_name in csv_files:
                # Lee el CSV desde el ZIP
                with z.open(csv_name) as f:
                    df = pd.read_csv(f, sep=",", engine="python")
                
                # Normaliza el nombre de la hoja (máx 31 caracteres en Excel)
                sheet_name = csv_name.replace(".csv", "")[:31]
                
                # Escribe el DataFrame en una hoja
                df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"Archivo Excel creado: {excel_output_path}")


# Ejemplo de uso:
zip_csvs_to_excel("OneDrive_1_27-11-2025.zip", "resultado.xlsx")
