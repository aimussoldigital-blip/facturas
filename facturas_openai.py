# --- Lector de Facturas con OCR + OpenAI ---
# Versión simplificada y funcional
import streamlit as st
import pandas as pd
import pytesseract
from pytesseract import TesseractNotFoundError
import fitz  # PyMuPDF
import io
import tempfile
import pathlib
import re
import json
import unicodedata
from PIL import Image
from fpdf import FPDF
from openai import OpenAI
import cv2
import numpy as np
from rapidfuzz import process, fuzz

st.set_page_config(page_title="OCR + OpenAI Facturas", layout="wide")

# =========================
# Configuración
# =========================
def has_tesseract() -> bool:
    try:
        _ = pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False
TES_AVAILABLE = has_tesseract()

def configurar_openai():
    for key_name in ["openai_api_key", "OPENAI_API_KEY"]:
        try:
            api_key = st.secrets[key_name]
            client = OpenAI(api_key=api_key)
            st.sidebar.success(f"✅ OpenAI configurado")
            return client
        except KeyError:
            continue
    st.error("❌ No se encontró la API key de OpenAI")
    st.stop()

client = configurar_openai()
st.sidebar.success("🧠 Tesseract: disponible" if TES_AVAILABLE else "⚠️ Tesseract: NO disponible")

# =========================
# Proveedores conocidos
# =========================
PROVEEDORES_CONOCIDOS = [
    "EHOSA", "MAREGALEVILLA", "SUPRACAFE", "HORMART", 
    "HOTMART BV", "ATOCHA VALLECAS", "MERCADONA S.A.",
    "COCA-COLA EUROPACIFIC PARTNERS", "OUIGO ESPAÑA S.A.U."
]

# =========================
# Funciones de extracción SIMPLES
# =========================
def extraer_numero_factura(texto):
    """Extrae número de factura de forma simple y directa"""
    st.write("🔍 Buscando número de factura...")
    
    # 1. Buscar números de 8 dígitos (como 01116253)
    numeros_8_digitos = re.findall(r'\b(\d{8})\b', texto)
    if numeros_8_digitos:
        st.write(f"✅ Encontrado (8 dígitos): {numeros_8_digitos[0]}")
        return numeros_8_digitos[0]
    
    # 2. Buscar patrones comunes
    patrones = [
        r'FV[-/]?\d+[-/]?\d+',  # FV-0-2515226
        r'C\d+',                # C2025000851658
        r'E\d+',                # E20252529117
        r'INV\d+',              # INV123456
        r'\d{6,15}',            # Cualquier número largo
    ]
    
    for patron in patrones:
        coincidencias = re.findall(patron, texto)
        if coincidencias:
            st.write(f"✅ Encontrado (patrón {patron}): {coincidencias[0]}")
            return coincidencias[0]
    
    st.write("❌ No se encontró número de factura")
    return "No encontrado"

def extraer_proveedor(texto):
    """Extrae proveedor de forma simple y directa"""
    st.write("🔍 Buscando proveedor...")
    
    texto_upper = texto.upper()
    
    # Buscar proveedores conocidos
    for proveedor in PROVEEDORES_CONOCIDOS:
        if proveedor.upper() in texto_upper:
            st.write(f"✅ Encontrado: {proveedor}")
            return proveedor
    
    # Buscar patrones específicos
    patrones = [
        (r'ATOCHA\s*VALLECAS', 'ATOCHA VALLECAS'),
        (r'HORMART', 'HORMART'),
        (r'HOTMART', 'HOTMART BV'),
        (r'MAREGALEVILLA', 'MAREGALEVILLA'),
        (r'SUPRACAFE', 'SUPRACAFE'),
        (r'EHOSA', 'EHOSA'),
    ]
    
    for patron, resultado in patrones:
        if re.search(patron, texto_upper):
            st.write(f"✅ Encontrado (patrón): {resultado}")
            return resultado
    
    st.write("❌ No se encontró proveedor")
    return "No encontrado"

# =========================
# Procesamiento de archivos
# =========================
def procesar_pdf(path):
    """Extrae texto de PDF"""
    doc = fitz.open(path)
    texto = ""
    for page in doc:
        texto += page.get_text()
    doc.close()
    return texto

def procesar_imagen(path):
    """Extrae texto de imagen con OCR"""
    if not TES_AVAILABLE:
        return "Error: Tesseract no disponible"
    
    img = Image.open(path)
    # Preprocesamiento simple
    img = img.convert('L')  # Escala de grises
    img = img.point(lambda x: 0 if x < 128 else 255, '1')  # Binarización
    
    texto = pytesseract.image_to_string(img, lang='spa')
    return texto

# =========================
# Función principal
# =========================
def procesar_archivo(archivo):
    """Procesa un archivo y extrae datos"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=archivo.name) as tmp:
        tmp.write(archivo.read())
        tmp_path = pathlib.Path(tmp.name)
    
    try:
        # Extraer texto según tipo de archivo
        if tmp_path.suffix.lower() == '.pdf':
            texto = procesar_pdf(tmp_path)
        else:
            texto = procesar_imagen(tmp_path)
        
        # Mostrar texto extraído para depuración
        with st.expander(f"Texto extraído de {archivo.name}"):
            st.text(texto[:1000])
        
        # Extraer datos
        numero_factura = extraer_numero_factura(texto)
        proveedor = extraer_proveedor(texto)
        
        return {
            "archivo": archivo.name,
            "nro_factura": numero_factura,
            "proveedor": proveedor
        }
    
    except Exception as e:
        return {
            "archivo": archivo.name,
            "nro_factura": f"Error: {str(e)}",
            "proveedor": f"Error: {str(e)}"
        }
    finally:
        try:
            tmp_path.unlink()
        except:
            pass

# =========================
# Interfaz
# =========================
st.title("📄 Lector de Facturas Simplificado")
st.markdown("Sube PDF o imágenes para extraer Nº de Factura y Proveedor")

archivos = st.file_uploader(
    "Selecciona archivos", 
    type=["pdf", "png", "jpg", "jpeg"], 
    accept_multiple_files=True
)

if archivos:
    resultados = []
    
    for archivo in archivos:
        with st.spinner(f"Procesando {archivo.name}..."):
            resultado = procesar_archivo(archivo)
            resultados.append(resultado)
    
    # Mostrar resultados
    st.subheader("📋 Resultados")
    
    # Calcular estadísticas
    facturas_ok = sum(1 for r in resultados if r["nro_factura"] not in ["No encontrado", "Error"])
    proveedores_ok = sum(1 for r in resultados if r["proveedor"] not in ["No encontrado", "Error"])
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Archivos", len(resultados))
    col2.metric("Facturas detectadas", facturas_ok)
    col3.metric("Proveedores detectados", proveedores_ok)
    
    # Tabla de resultados
    df = pd.DataFrame(resultados)
    st.dataframe(
        df,
        column_config={
            "archivo": "Archivo",
            "nro_factura": "Nº Factura",
            "proveedor": "Proveedor"
        }
    )
    
    # Detalles
    with st.expander("📊 Ver detalles"):
        for r in resultados:
            st.write(f"**{r['archivo']}**")
            st.write(f"- Factura: `{r['nro_factura']}`")
            st.write(f"- Proveedor: `{r['proveedor']}`")
            st.divider()
    
    # Descargar resultados
    if st.button("Generar Excel"):
        excel_buf = io.BytesIO()
        df.to_excel(excel_buf, index=False)
        excel_buf.seek(0)
        st.download_button(
            "📥 Descargar Excel",
            data=excel_buf.getvalue(),
            file_name="resultados_facturas.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )










