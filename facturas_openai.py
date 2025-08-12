# --- Lector de Facturas Ultra Simplificado ---
import streamlit as st
import pandas as pd
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import tempfile
import pathlib
import re

st.set_page_config(page_title="Lector de Facturas", layout="wide")

st.title("📄 Lector de Facturas Ultra Simplificado")
st.markdown("Sube PDF o imágenes para extraer Nº de Factura y Proveedor")

# Función para extraer texto de PDF
def extraer_texto_pdf(archivo):
    doc = fitz.open(stream=archivo.read(), filetype="pdf")
    texto = ""
    for page in doc:
        texto += page.get_text()
    doc.close()
    return texto

# Función para extraer texto de imagen
def extraer_texto_imagen(archivo):
    try:
        imagen = Image.open(archivo)
        texto = pytesseract.image_to_string(imagen, lang='spa')
        return texto
    except:
        return "Error al procesar imagen"

# Función para extraer número de factura
def extraer_factura(texto):
    # Buscar cualquier secuencia de números que pueda ser una factura
    # Priorizamos números de 8 dígitos (como 01116253)
    numeros = re.findall(r'\b\d{8}\b', texto)
    if numeros:
        return numeros[0]
    
    # Si no hay 8 dígitos, buscamos otros patrones
    patrones = [
        r'\b\d{6,10}\b',  # Cualquier número entre 6 y 10 dígitos
        r'[A-Z]\d{6,}',    # Letra seguida de números (C2025000851658)
        r'FV\d+',          # FV seguido de números
    ]
    
    for patron in patrones:
        coincidencias = re.findall(patron, texto)
        if coincidencias:
            return coincidencias[0]
    
    return "No encontrado"

# Función para extraer proveedor
def extraer_proveedor(texto):
    texto = texto.upper()
    
    # Lista de proveedores a buscar
    proveedores = [
        "EHOSA",
        "MAREGALEVILLA", 
        "SUPRACAFE",
        "HORMART",
        "HOTMART",
        "ATOCHA VALLECAS",
        "MERCADONA",
        "OUIGO"
    ]
    
    for proveedor in proveedores:
        if proveedor in texto:
            return proveedor
    
    return "No encontrado"

# Subida de archivos
archivos = st.file_uploader(
    "Selecciona archivos", 
    type=["pdf", "png", "jpg", "jpeg"], 
    accept_multiple_files=True
)

if archivos:
    resultados = []
    
    for archivo in archivos:
        # Extraer texto según tipo
        if archivo.type == "application/pdf":
            texto = extraer_texto_pdf(archivo)
        else:
            texto = extraer_texto_imagen(archivo)
        
        # Extraer datos
        numero_factura = extraer_factura(texto)
        proveedor = extraer_proveedor(texto)
        
        # Guardar resultado
        resultados.append({
            "archivo": archivo.name,
            "nro_factura": numero_factura,
            "proveedor": proveedor,
            "texto_extraido": texto[:500] + "..." if len(texto) > 500 else texto
        })
    
    # Mostrar resultados
    st.subheader("📋 Resultados")
    
    # Tabla de resultados
    df = pd.DataFrame(resultados)
    df_display = df.drop(columns=['texto_extraido'])  # No mostrar texto en la tabla principal
    st.dataframe(df_display)
    
    # Mostrar detalles
    for i, resultado in enumerate(resultados):
        with st.expander(f"Detalles de {resultado['archivo']}"):
            st.write(f"**Número de factura:** {resultado['nro_factura']}")
            st.write(f"**Proveedor:** {resultado['proveedor']}")
            st.write("**Texto extraído:**")
            st.text(resultado['texto_extraido'])
    
    # Botón para descargar resultados
    if st.button("Descargar Resultados"):
        # Crear Excel en memoria
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df_display.to_excel(writer, index=False, sheet_name='Resultados')
        output.seek(0)
        
        st.download_button(
            label="📥 Descargar Excel",
            data=output,
            file_name='resultados_facturas.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )









