# --- Lector de Facturas con OCR + OpenAI (MEJORADO) ---
# Mejoras: patrones más robustos, mejor preprocesamiento OCR, 
# lógica simplificada y más tolerante para detección de proveedores

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

# Nuevas dependencias
import cv2
import numpy as np
from rapidfuzz import process, fuzz

# ✅ Siempre primero en Streamlit:
st.set_page_config(page_title="OCR + OpenAI Facturas", layout="wide")

# =========================
# Entorno / Dependencias
# =========================
def has_tesseract() -> bool:
    try:
        _ = pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False

TES_AVAILABLE = has_tesseract()

# --- Configuración de OpenAI ---
def configurar_openai():
    st.sidebar.write("🔍 Debug - Secrets:")
    try:
        available_keys = list(st.secrets.keys())
        st.sidebar.write(f"Keys encontradas: {available_keys}")
    except Exception:
        st.sidebar.write("No se pudieron leer los secrets")

    for key_name in ["openai_api_key","OPENAI_API_KEY","openai-api-key","openai_key","OPENAI_KEY","api_key"]:
        try:
            api_key = st.secrets[key_name]
            client = OpenAI(api_key=api_key)
            st.sidebar.success(f"✅ OpenAI configurado con key: '{key_name}'")
            return client
        except KeyError:
            continue
        except Exception as e:
            st.sidebar.error(f"Error con key '{key_name}': {e}")
            continue

    st.error("❌ No se encontró la API key de OpenAI")
    st.info("Manage app → Settings → Secrets → agrega:\n\nopenai_api_key = \"sk-...\"")
    st.stop()

client = configurar_openai()
st.sidebar.success("🧠 Tesseract: disponible" if TES_AVAILABLE else "⚠️ Tesseract: NO disponible")

# =========================
# Patrones mejorados
# =========================

# Patrones más específicos para números de factura
INVOICE_PATTERNS = [
    # Patrones con prefijos específicos
    r'\b(?:FACT(?:URA)?\.?\s*N[ºo°]?\.?\s*|N[ºo°]\.?\s*FACT(?:URA)?\.?\s*|INVOICE\s*(?:N[ºo°])?\.?\s*|FACTURE\s*N[ºo°]\.?\s*)([A-Z0-9][A-Z0-9\-/\.]{3,20})\b',
    
    # Formatos específicos comunes
    r'\b([A-Z]{1,4}[-/]\d{4,12})\b',  # FV-123456, A-123456
    r'\b([A-Z]\d{8,15})\b',           # C123456789, E123456789
    r'\b(FV[-/]?\d{1,3}[-/]?\d{4,12})\b',  # FV-0-123456
    r'\b([A-Z]{2,5}\d{3,12})\b',      # INV123456, FACT123456
    r'\b(\d{7,12})\b',                # Números largos sin prefijo
    
    # Patrones con separadores
    r'\b([A-Z0-9]{1,4}[-/][A-Z0-9]{1,4}[-/][A-Z0-9]{3,12})\b',  # A-V-123456
    r'\b(\d{4}[-/]\d{6,10})\b',       # 2024-123456
]

# Palabras que indican contexto de factura
INVOICE_CONTEXT_WORDS = [
    'FACTURA', 'FACT', 'INVOICE', 'FACTURE', 'BILL', 'RECEIPT',
    'NÚMERO', 'NUMERO', 'Nº', 'N°', 'NO', 'NUMBER', '#'
]

# Palabras que NO deben aparecer en números de factura
INVALID_INVOICE_WORDS = [
    'CLIENTE', 'CLIENT', 'CUSTOMER', 'DESTINATARIO', 'FECHA', 'DATE',
    'TELEFONO', 'TELÉFONO', 'PHONE', 'EMAIL', 'DIRECCION', 'ADDRESS',
    'CIF', 'NIF', 'VAT', 'IVA'
]

# Proveedores conocidos (más flexibles)
KNOWN_SUPPLIERS_PATTERNS = [
    r'\b(MERCADONA\s*S\.?A\.?)\b',
    r'\b(CARREFOUR)\b',
    r'\b(ALCAMPO)\b',
    r'\b(MAKRO)\b',
    r'\b(DIA(?:\s*S\.?A\.?)?)\b',
    r'\b(LIDL)\b',
    r'\b(EROSKI)\b',
    r'\b(SUPRACAFE|SUPRACAFÉ)\b',
    r'\b(MAREGALEVILLA|MARE\s*GALE\s*VILLA)\b',
    r'\b(EHOSA)\b',
    r'\b(COCA\s*COLA)\b',
    r'\b([A-ZÁÉÍÓÚÑ][A-Za-záéíóúñ\s]{2,}\s+S\.?[LA]\.?)\b',  # Empresas con S.L. o S.A.
]

# =========================
# Utilidades mejoradas
# =========================
def normalize_text(text: str) -> str:
    """Normaliza texto eliminando acentos y caracteres especiales"""
    if not text:
        return ""
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(c for c in text if not unicodedata.combining(c))
    return text.upper().strip()

def is_valid_invoice_number(candidate: str) -> bool:
    """Valida si un candidato es un número de factura válido"""
    if not candidate or len(candidate) < 3:
        return False
    
    # No debe contener palabras inválidas
    candidate_norm = normalize_text(candidate)
    if any(word in candidate_norm for word in INVALID_INVOICE_WORDS):
        return False
    
    # No debe ser solo números cortos
    if candidate.isdigit() and len(candidate) < 6:
        return False
    
    # No debe ser un NIF/CIF
    if re.match(r'^\d{8}[A-Z]$', candidate) or re.match(r'^[A-Z]\d{7}[0-9A-J]$', candidate):
        return False
    
    # No debe ser un teléfono
    if re.match(r'^\+?[\d\s\-]{9,15}$', candidate):
        return False
    
    # No debe ser una fecha
    if re.match(r'^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$', candidate):
        return False
    
    return True

def score_invoice_number(candidate: str, context: str = "") -> int:
    """Puntúa qué tan probable es que sea un número de factura"""
    if not candidate:
        return 0
    
    score = 0
    candidate_norm = normalize_text(candidate)
    context_norm = normalize_text(context)
    
    # Bonus por contexto
    if any(word in context_norm for word in ['FACTURA', 'INVOICE', 'FACT']):
        score += 10
    
    # Bonus por formato
    if re.match(r'^[A-Z]+[-/]?\d+$', candidate):
        score += 5
    if re.match(r'^\d{7,}$', candidate):
        score += 3
    if '-' in candidate or '/' in candidate:
        score += 2
    if any(char.isalpha() for char in candidate):
        score += 2
    
    # Penalización por longitud inadecuada
    if len(candidate) < 4 or len(candidate) > 25:
        score -= 3
    
    return score

# =========================
# OCR mejorado
# =========================
def preprocess_image_advanced(pil_img):
    """Preprocesamiento avanzado de imagen para OCR"""
    img_array = np.array(pil_img.convert('RGB'))
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Convertir a escala de grises
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Reducir ruido
    denoised = cv2.medianBlur(gray, 3)
    
    # Mejorar contraste
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)
    
    # Binarización adaptativa
    binary = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    # Morfología para limpiar
    kernel = np.ones((1,1), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    return cleaned

def extract_text_with_ocr(pil_img) -> str:
    """Extrae texto usando OCR con múltiples configuraciones"""
    if not TES_AVAILABLE:
        raise TesseractNotFoundError("Tesseract no está disponible")
    
    # Preprocesar imagen
    processed_img = preprocess_image_advanced(pil_img)
    
    best_text = ""
    best_score = 0
    
    # Probar diferentes configuraciones de PSM
    psm_configs = [6, 4, 11, 12, 3]  # Diferentes modos de segmentación
    
    for psm in psm_configs:
        try:
            config = f'--oem 3 --psm {psm} -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÁÉÍÓÚáéíóúñÑ/-.'
            text = pytesseract.image_to_string(processed_img, lang='spa+eng', config=config)
            
            # Puntuar la calidad del texto extraído
            score = len(text) + text.count('FACTURA') * 10 + text.count('INVOICE') * 10
            
            if score > best_score:
                best_text = text
                best_score = score
                
        except Exception as e:
            st.warning(f"Error con PSM {psm}: {e}")
            continue
    
    return best_text

# =========================
# Extracción mejorada
# =========================
def extract_invoice_number(text: str) -> str:
    """Extrae el número de factura del texto"""
    if not text:
        return "No encontrado"
    
    candidates = []
    
    # Buscar con patrones específicos
    for pattern in INVOICE_PATTERNS:
        matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
        for match in matches:
            candidate = match.group(1) if match.groups() else match.group(0)
            if is_valid_invoice_number(candidate):
                # Obtener contexto alrededor del match
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                context = text[start:end]
                
                score = score_invoice_number(candidate, context)
                candidates.append((candidate, score))
    
    # También buscar números después de palabras clave
    lines = text.split('\n')
    for i, line in enumerate(lines):
        line_norm = normalize_text(line)
        
        # Buscar líneas que contengan palabras clave de factura
        if any(word in line_norm for word in ['FACTURA', 'INVOICE', 'FACT']):
            # Buscar números en esta línea y la siguiente
            search_lines = [line]
            if i + 1 < len(lines):
                search_lines.append(lines[i + 1])
            
            for search_line in search_lines:
                numbers = re.findall(r'[A-Z0-9][A-Z0-9\-/\.]{3,20}', search_line, re.IGNORECASE)
                for num in numbers:
                    if is_valid_invoice_number(num):
                        score = score_invoice_number(num, line)
                        candidates.append((num, score))
    
    # Seleccionar el mejor candidato
    if candidates:
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_candidate = candidates[0][0]
        st.write(f"✅ Número de factura encontrado: {best_candidate} (score: {candidates[0][1]})")
        return best_candidate
    
    return "No encontrado"

def extract_supplier(text: str) -> str:
    """Extrae el proveedor del texto"""
    if not text:
        return "No encontrado"
    
    # Buscar proveedores conocidos primero
    for pattern in KNOWN_SUPPLIERS_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            supplier = match.group(1).strip()
            st.write(f"✅ Proveedor conocido encontrado: {supplier}")
            return supplier
    
    # Buscar en las primeras líneas del documento
    lines = text.split('\n')[:20]  # Solo las primeras 20 líneas
    
    for line in lines:
        line = line.strip()
        if len(line) < 3 or len(line) > 60:
            continue
            
        line_norm = normalize_text(line)
        
        # Filtrar líneas que obviamente no son proveedores
        skip_words = ['FACTURA', 'INVOICE', 'FECHA', 'DATE', 'CLIENTE', 'CUSTOMER', 
                     'TELEFONO', 'TELÉFONO', 'EMAIL', 'DIRECCION', 'ADDRESS']
        if any(word in line_norm for word in skip_words):
            continue
        
        # Buscar patrones de empresa
        if (re.search(r'S\.?[LA]\.?$', line_norm) or  # Termina en S.L. o S.A.
            re.search(r'\b[A-Z]{2,}(?:\s+[A-Z]{2,}){1,3}\b', line_norm) or  # Palabras en mayúsculas
            re.search(r'^\w+(?:\s+\w+){1,4}$', line)):  # Formato de nombre comercial
            
            st.write(f"✅ Posible proveedor encontrado: {line}")
            return line
    
    return "No encontrado"

# =========================
# Procesamiento con OpenAI mejorado
# =========================
def extract_data_with_openai_improved(text: str) -> dict:
    """Extrae datos usando OpenAI con prompt mejorado"""
    
    # Primero intentar extracción con regex
    regex_invoice = extract_invoice_number(text)
    regex_supplier = extract_supplier(text)
    
    # Limpiar texto para OpenAI (tomar solo las primeras líneas más relevantes)
    lines = text.split('\n')
    relevant_text = '\n'.join([line.strip() for line in lines[:50] if line.strip()])
    
    prompt = f"""
Eres un experto analizando facturas españolas. Extrae EXACTAMENTE estos datos:

1. NÚMERO DE FACTURA: El código único que identifica esta factura (puede tener letras, números, guiones)
2. PROVEEDOR: El nombre de la empresa que EMITE la factura (NO el cliente que la recibe)

REGLAS IMPORTANTES:
- NO extraigas NIFs, CIFs, teléfonos o fechas como números de factura
- El proveedor es quien VENDE, no quien COMPRA
- Si no encuentras algo, responde "No encontrado"
- Responde SOLO en formato JSON

TEXTO DE LA FACTURA:
{relevant_text}

Respuesta JSON:
"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Eres un experto extrayendo datos de facturas. Responde solo con JSON válido."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=150
        )
        
        content = response.choices[0].message.content.strip()
        
        # Limpiar respuesta JSON
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "").strip()
        
        result = json.loads(content)
        
        # Extraer y validar datos
        ai_invoice = result.get("invoiceNumber", result.get("numero_factura", "No encontrado"))
        ai_supplier = result.get("supplier", result.get("proveedor", "No encontrado"))
        
        # Combinar resultados de AI y regex, priorizando el mejor
        final_invoice = ai_invoice
        if (ai_invoice == "No encontrado" and regex_invoice != "No encontrado"):
            final_invoice = regex_invoice
        elif (regex_invoice != "No encontrado" and 
              score_invoice_number(regex_invoice) > score_invoice_number(ai_invoice)):
            final_invoice = regex_invoice
        
        final_supplier = ai_supplier if ai_supplier != "No encontrado" else regex_supplier
        
        st.write(f"🤖 AI - Factura: {ai_invoice}, Proveedor: {ai_supplier}")
        st.write(f"🔧 Regex - Factura: {regex_invoice}, Proveedor: {regex_supplier}")
        st.write(f"✅ Final - Factura: {final_invoice}, Proveedor: {final_supplier}")
        
        return {
            "nro_factura": final_invoice,
            "proveedor": final_supplier
        }
        
    except Exception as e:
        st.warning(f"Error con OpenAI: {e}")
        return {
            "nro_factura": regex_invoice,
            "proveedor": regex_supplier
        }

# =========================
# Procesamiento de archivos
# =========================
def extract_text_from_file(file_path: pathlib.Path) -> str:
    """Extrae texto de PDF o imagen"""
    try:
        if file_path.suffix.lower() == '.pdf':
            # Procesar PDF
            doc = fitz.open(file_path)
            text_parts = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Intentar extraer texto nativo primero
                text = page.get_text("text").strip()
                
                if len(text) > 100:  # Suficiente texto nativo
                    text_parts.append(text)
                else:
                    # Fallback a OCR
                    if TES_AVAILABLE:
                        pix = page.get_pixmap(dpi=300)
                        img_data = pix.tobytes()
                        img = Image.open(io.BytesIO(img_data))
                        ocr_text = extract_text_with_ocr(img)
                        text_parts.append(ocr_text)
                    else:
                        text_parts.append(text)  # Usar texto nativo aunque sea poco
            
            doc.close()
            return '\n'.join(text_parts)
            
        else:
            # Procesar imagen
            img = Image.open(file_path)
            
            # Rotar según EXIF si es necesario
            if hasattr(img, '_getexif'):
                exif = img._getexif()
                if exif is not None:
                    orientation = exif.get(274)
                    if orientation == 3:
                        img = img.rotate(180, expand=True)
                    elif orientation == 6:
                        img = img.rotate(270, expand=True)
                    elif orientation == 8:
                        img = img.rotate(90, expand=True)
            
            return extract_text_with_ocr(img)
            
    except Exception as e:
        raise Exception(f"Error procesando archivo: {e}")

def process_single_file(file, index: int) -> dict:
    """Procesa un solo archivo"""
    
    # Crear archivo temporal
    with tempfile.NamedTemporaryFile(delete=False, suffix=pathlib.Path(file.name).suffix) as tmp_file:
        tmp_file.write(file.read())
        tmp_path = pathlib.Path(tmp_file.name)
    
    try:
        # Extraer texto
        text = extract_text_from_file(tmp_path)
        
        if not text or not text.strip():
            return {
                "archivo": file.name,
                "nro_factura": "Error: No se pudo extraer texto",
                "proveedor": "Error: No se pudo extraer texto"
            }
        
        # Debug: mostrar texto extraído
        with st.expander(f"🔍 Texto extraído - {file.name}"):
            st.text_area(
                "Primeras líneas:",
                text[:1000],
                height=200,
                key=f"debug_text_{index}_{file.name}"
            )
        
        # Extraer datos
        result = extract_data_with_openai_improved(text)
        
        return {
            "archivo": file.name,
            **result
        }
        
    except Exception as e:
        return {
            "archivo": file.name,
            "nro_factura": f"Error: {str(e)}",
            "proveedor": f"Error: {str(e)}"
        }
    finally:
        # Limpiar archivo temporal
        try:
            tmp_path.unlink()
        except:
            pass

# =========================
# Interfaz de usuario
# =========================
st.title("📄 Lector de Facturas Mejorado - OCR + OpenAI")
st.markdown("**Versión mejorada** con mejor detección de números de factura y proveedores")

# Test de conexión OpenAI
if st.sidebar.button("🧪 Test OpenAI"):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Responde solo: OK"}],
            max_tokens=10
        )
        st.sidebar.success("✅ Conexión OpenAI funcionando")
    except Exception as e:
        st.sidebar.error(f"❌ Error OpenAI: {e}")

# Información
with st.expander("ℹ️ Información"):
    st.markdown("""
    **Mejoras implementadas:**
    - ✅ Patrones más específicos para números de factura
    - ✅ Mejor preprocesamiento de imágenes para OCR
    - ✅ Validación mejorada de números de factura
    - ✅ Detección más robusta de proveedores
    - ✅ Combinación inteligente de resultados AI + Regex
    - ✅ Mejor manejo de errores y debugging
    """)

# Cargar archivos
uploaded_files = st.file_uploader(
    "Selecciona archivos de facturas",
    type=['pdf', 'png', 'jpg', 'jpeg'],
    accept_multiple_files=True,
    help="Sube PDFs o imágenes de facturas"
)

if uploaded_files:
    st.info(f"📁 {len(uploaded_files)} archivo(s) cargado(s)")
    
    if st.button("🚀 Procesar Facturas", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        results = []
        
        for i, file in enumerate(uploaded_files):
            status_text.text(f"Procesando {file.name}... ({i+1}/{len(uploaded_files)})")
            
            result = process_single_file(file, i)
            results.append(result)
            
            progress_bar.progress((i + 1) / len(uploaded_files))
        
        status_text.text("✅ ¡Procesamiento completado!")
        
        # Mostrar resultados
        st.subheader("📊 Resultados")
        
        df = pd.DataFrame(results)
        
        # Métricas
        col1, col2, col3 = st.columns(3)
        
        with col1:
            facturas_ok = sum(1 for r in results 
                            if r["nro_factura"] not in ["No encontrado", "Error: No se pudo extraer texto"] 
                            and not r["nro_factura"].startswith("Error:"))
            st.metric("✅ Facturas detectadas", facturas_ok)
        
        with col2:
            proveedores_ok = sum(1 for r in results 
                               if r["proveedor"] not in ["No encontrado", "Error: No se pudo extraer texto"] 
                               and not r["proveedor"].startswith("Error:"))
            st.metric("✅ Proveedores detectados", proveedores_ok)
        
        with col3:
            st.metric("📄 Total archivos", len(results))
        
        # Tabla con colores
        def highlight_status(val):
            if "Error:" in str(val):
                return 'background-color: #ffebee'  # Rojo claro
            elif str(val) == "No encontrado":
                return 'background-color: #fff3e0'  # Naranja claro
            else:
                return 'background-color: #e8f5e8'  # Verde claro
        
        styled_df = df.style.applymap(highlight_status, subset=['nro_factura', 'proveedor'])
        st.dataframe(styled_df, use_container_width=True)
        
        # Descargas
        st.subheader("📥 Descargas")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Excel
            excel_buffer = io.BytesIO()
            df.to_excel(excel_buffer, index=False, engine='openpyxl')
            excel_buffer.seek(0)
            
            st.download_button(
                "📊 Descargar Excel",
                data=excel_buffer.getvalue(),
                file_name="facturas_procesadas.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        with col2:
            # CSV
            csv = df.to_csv(index=False)
            st.download_button(
                "📄 Descargar CSV",
                data=csv,
                file_name="facturas_procesadas.csv",
                mime="text/csv"
            )

# Limpiar sesión
if st.sidebar.button("🗑️ Limpiar datos"):
    st.session_state.clear()
    st.rerun()








