# --- Lector de Facturas con OCR + OpenAI MEJORADO ---
# Soluciona problemas de extracción de factura/proveedor en Streamlit Cloud

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
import logging

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ✅ Siempre primero en Streamlit:
st.set_page_config(page_title="OCR + OpenAI Facturas", layout="wide")

# =========================
# MEJORA 1: Mejor detección de Tesseract
# =========================
def has_tesseract() -> bool:
    try:
        # Intentar múltiples verificaciones
        version = pytesseract.get_tesseract_version()
        logger.info(f"Tesseract version: {version}")
        
        # Test básico con imagen pequeña
        test_img = Image.new('RGB', (100, 50), color='white')
        test_result = pytesseract.image_to_string(test_img)
        logger.info("Tesseract test passed")
        return True
    except Exception as e:
        logger.error(f"Tesseract not available: {e}")
        return False

TES_AVAILABLE = has_tesseract()

# Mostrar estado de Tesseract prominentemente
if TES_AVAILABLE:
    st.sidebar.success("🧠 Tesseract: ✅ DISPONIBLE")
else:
    st.sidebar.error("⚠️ Tesseract: ❌ NO DISPONIBLE")
    st.sidebar.info("Para imágenes, se usará solo extracción de PDFs con texto nativo")

# =========================
# MEJORA 2: Configuración OpenAI más robusta
# =========================
def configurar_openai():
    st.sidebar.write("🔍 Debug - Configuración OpenAI:")
    
    # Lista de posibles nombres de secrets
    possible_keys = [
        "openai_api_key", "OPENAI_API_KEY", "openai-api-key", 
        "openai_key", "OPENAI_KEY", "api_key", "API_KEY"
    ]
    
    # Mostrar secrets disponibles para debug
    try:
        available_keys = list(st.secrets.keys())
        st.sidebar.write(f"Secrets encontrados: {available_keys}")
    except Exception as e:
        st.sidebar.write(f"Error leyendo secrets: {e}")
        available_keys = []

    # Intentar cada posible key
    for key_name in possible_keys:
        try:
            if key_name in available_keys:
                api_key = st.secrets[key_name]
                if api_key and len(api_key) > 10:  # Validación básica
                    client = OpenAI(api_key=api_key)
                    st.sidebar.success(f"✅ OpenAI configurado con: '{key_name}'")
                    return client
        except Exception as e:
            st.sidebar.warning(f"Error con key '{key_name}': {e}")
            continue

    st.sidebar.error("❌ No se encontró API key válida de OpenAI")
    st.sidebar.info("""
    **Configurar en Streamlit Cloud:**
    1. Manage app → Settings → Secrets
    2. Agregar: `openai_api_key = "sk-..."`
    """)
    st.stop()

client = configurar_openai()

# =========================
# MEJORA 3: Patrones de factura más específicos y ordenados por prioridad
# =========================
INVOICE_PATTERNS_PRIORITY = [
    # Patrones muy específicos primero (alta confianza)
    (r'([A-Z]{1,3}-[A-Z0-9]{1,4}-\d{5,15})', 10),        # A-V2025-00002609357
    (r'(FV[-/]?\d{1,2}[-/]?\d{5,10})', 9),                # FV-0-2515226
    (r'([A-Z]\d{9,18})', 8),                              # C2025000851658
    (r'(\d{4}[-/][A-Z0-9]{4,8})', 7),                     # 2025-ABC123
    (r'([A-Z]{2,5}[-/]?\d{6,12})', 6),                    # ABC-123456789
    
    # Patrones contextuales (requieren contexto)
    (r'(?:factura|invoice|fact|fac|nº|n°|no|num|number)[\s:.-]*([A-Z0-9\-/\.]{4,25})', 5),
    (r'F[-/]?(\d{6,12})', 4),
    (r'INV[-/]?(\d{6,12})', 4),
    (r'(e\d{8,15})', 3),                                  # e20252529117
    
    # Patrones más generales (baja prioridad)
    (r'([A-Z]\d{8,15})', 2),
    (r'(\d{6,15})', 1),                                   # Solo números largos
]

# =========================
# MEJORA 4: Validadores mejorados
# =========================
def is_valid_invoice_number(s: str) -> bool:
    """Validación más estricta para números de factura"""
    if not s or s == "No encontrado":
        return False
    
    s = s.strip()
    
    # Rechazar patrones obvios
    invalid_patterns = [
        r'^\d{8}[A-Z]$',        # NIF
        r'^[A-Z]\d{7}[0-9A-J]$', # CIF
        r'^\d{2}[/-]\d{2}[/-]\d{4}$', # Fechas
        r'^(\+34)?\s?\d{3}[\s-]?\d{2,3}[\s-]?\d{2,3}$', # Teléfonos
        r'^(FACTURA|INVOICE|FACT|TICKET|ALBARAN|ALBARÁN)$', # Palabras genéricas
        r'^\d{1,5}$',           # Números muy cortos
        r'^[A-Z]$',             # Una sola letra
    ]
    
    for pattern in invalid_patterns:
        if re.match(pattern, s, re.IGNORECASE):
            return False
    
    # Debe tener al menos 4 caracteres
    if len(s) < 4:
        return False
        
    # Debe contener al menos un dígito o letra
    if not re.search(r'[A-Za-z0-9]', s):
        return False
    
    return True

def score_invoice_improved(s: str) -> int:
    """Scoring mejorado para números de factura"""
    if not is_valid_invoice_number(s):
        return 0
    
    score = 0
    s_upper = s.upper()
    
    # Prefijos conocidos de facturas
    if re.search(r'\b(FV|INV|AV|A-V|DGFC|BVRES|C\d{2,})[-/A-Z]?', s_upper):
        score += 8
    
    # Formato con guiones o barras
    if '-' in s or '/' in s:
        score += 4
    
    # Mezcla de letras y números
    has_letters = bool(re.search(r'[A-Z]', s_upper))
    has_numbers = bool(re.search(r'\d', s))
    if has_letters and has_numbers:
        score += 3
    
    # Longitud apropiada
    if 6 <= len(s) <= 20:
        score += 2
    elif 4 <= len(s) <= 25:
        score += 1
    
    # Patrones específicos conocidos
    if re.match(r'^[A-Z]{1,3}-[A-Z0-9]+-\d+$', s_upper):
        score += 5
    
    return score

# =========================
# MEJORA 5: Mejor extracción de proveedores
# =========================
KNOWN_SUPPLIERS_IMPROVED = {
    # Exactos (alta prioridad)
    'MAREGALEVILLA': ['MAREGALEVILLA', 'MARE GALE VILLA', 'WWW.MAREGALEVILLA'],
    'SUPRACAFE': ['SUPRACAFE', 'SUPRACAFÉ', 'SUPRA CAFE', 'WWW.SUPRACAFE'],
    'EHOSA': ['EHOSA', 'ELESPEJO HOSTELEROS S.A.', 'EL ESPEJO HOSTELEROS'],
    'MERCADONA S.A.': ['MERCADONA', 'MERCADONA S.A.', 'MERCADONA SA'],
    'COCA-COLA EUROPACIFIC PARTNERS': ['COCA COLA EUROPACIFIC', 'COCA-COLA EUROPACIFIC'],
    
    # Otros proveedores comunes
    'MAKRO': ['MAKRO'],
    'CARREFOUR': ['CARREFOUR'],
    'ALCAMPO': ['ALCAMPO'],
    'DIA S.A.': ['DIA S.A.', 'DIA SA'],
    'LIDL': ['LIDL'],
    'EROSKI': ['EROSKI'],
}

def find_supplier_in_text(text: str) -> str:
    """Búsqueda mejorada de proveedores en el texto"""
    text_upper = text.upper()
    
    # Buscar proveedores conocidos por orden de prioridad
    for canonical_name, variations in KNOWN_SUPPLIERS_IMPROVED.items():
        for variation in variations:
            if variation in text_upper:
                logger.info(f"Proveedor encontrado: {canonical_name} (variación: {variation})")
                return canonical_name
    
    # Búsqueda por patrones de empresa
    lines = text.split('\n')[:30]  # Solo primeras 30 líneas
    
    for line in lines:
        line = line.strip()
        if len(line) < 4 or len(line) > 60:
            continue
            
        # Evitar líneas con datos de contacto
        if re.search(r'(TEL|TLF|FAX|EMAIL|@|WWW|HTTP)', line, re.IGNORECASE):
            continue
            
        # Evitar direcciones y fechas
        if re.search(r'(CALLE|C/|AVDA|PLAZA|\d{2}[/-]\d{2}[/-]\d{4})', line, re.IGNORECASE):
            continue
        
        # Buscar patrones de empresa
        empresa_patterns = [
            r'^([A-ZÁÉÍÓÚÑ][A-Za-zÁÉÍÓÚáéíóúñ\s.&\-\']{3,40}(?:S\.?[AL]\.?|SOCIEDAD|LIMITADA|ANONIMA))$',
            r'^([A-ZÁÉÍÓÚÑ]{3,}(?:\s+[A-ZÁÉÍÓÚÑ]{2,}){1,4})$',
        ]
        
        for pattern in empresa_patterns:
            match = re.match(pattern, line)
            if match:
                candidate = match.group(1).strip()
                
                # Validar que no sea palabra común
                if candidate.upper() not in {'FACTURA', 'CLIENTE', 'FECHA', 'TOTAL'}:
                    logger.info(f"Candidato a proveedor encontrado: {candidate}")
                    return candidate
    
    return "No encontrado"

# =========================
# MEJORA 6: OCR más robusto con múltiples intentos
# =========================
def extract_text_robust(pil_image) -> str:
    """Extracción de texto con múltiples estrategias"""
    if not TES_AVAILABLE:
        return ""
    
    strategies = [
        # Estrategia 1: Imagen original con PSM 6
        {
            'preprocess': lambda img: np.array(img.convert('L')),
            'config': '--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789/-.,: '
        },
        # Estrategia 2: Binarización adaptativa
        {
            'preprocess': lambda img: cv2.adaptiveThreshold(
                np.array(img.convert('L')), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            ),
            'config': '--oem 3 --psm 4'
        },
        # Estrategia 3: Imagen escalada
        {
            'preprocess': lambda img: np.array(img.resize((img.width*2, img.height*2), Image.LANCZOS).convert('L')),
            'config': '--oem 3 --psm 6'
        }
    ]
    
    best_text = ""
    best_score = 0
    
    for i, strategy in enumerate(strategies):
        try:
            processed_img = strategy['preprocess'](pil_image)
            text = pytesseract.image_to_string(
                processed_img, 
                lang='spa+eng', 
                config=strategy['config']
            )
            
            # Scoring simple: más caracteres alfanuméricos = mejor
            score = len(re.findall(r'[A-Za-z0-9]', text))
            
            if score > best_score:
                best_text = text
                best_score = score
                logger.info(f"Mejor OCR con estrategia {i+1}: {score} caracteres")
                
        except Exception as e:
            logger.error(f"Error en estrategia OCR {i+1}: {e}")
            continue
    
    return best_text

# =========================
# MEJORA 7: Extracción mejorada con OpenAI
# =========================
def extract_with_openai_improved(text: str, filename: str = "") -> dict:
    """Extracción mejorada usando OpenAI con mejor prompt"""
    
    # Limpiar texto
    clean_text = re.sub(r'[^\w\s\-.,:/()áéíóúÁÉÍÓÚñÑ&]', ' ', text)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    
    # Prompt mejorado
    prompt = f"""
Analiza esta factura española y extrae EXACTAMENTE:

1. NÚMERO DE FACTURA: Busca códigos como:
   - Formatos: A-V2025-123456, FV-0-123456, C2025000851658
   - Cerca de palabras: "FACTURA", "INVOICE", "Nº", "NÚMERO"
   - NO extraigas: DNI/NIF (8 dígitos + letra), fechas (DD/MM/AAAA), teléfonos

2. PROVEEDOR: La empresa QUE EMITE la factura (NO el cliente):
   - Busca nombres de empresa, S.L., S.A.
   - Proveedores conocidos: SUPRACAFE, MAREGALEVILLA, EHOSA, MERCADONA
   - NO extraigas: direcciones, ciudades, clientes

RESPONDE SOLO EN FORMATO JSON:
{{
  "invoiceNumber": "código encontrado o No encontrado",
  "supplier": "empresa emisora o No encontrado"
}}

TEXTO DE LA FACTURA:
{clean_text[:2500]}

NOMBRE DEL ARCHIVO (pista): {filename}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Eres un experto extractor de datos de facturas españolas. Solo devuelves JSON válido."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=200
        )
        
        content = response.choices[0].message.content.strip()
        
        # Limpiar respuesta
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "").strip()
        
        result = json.loads(content)
        
        # Validar y limpiar resultados
        invoice_num = result.get("invoiceNumber", "No encontrado").strip()
        supplier = result.get("supplier", "No encontrado").strip()
        
        # Validar número de factura
        if not is_valid_invoice_number(invoice_num):
            invoice_num = "No encontrado"
        
        # Validar proveedor
        if supplier.upper() in {'FACTURA', 'CLIENTE', 'CUSTOMER', 'FECHA', 'TOTAL', 'IVA'}:
            supplier = "No encontrado"
        
        return {
            "nro_factura": invoice_num,
            "proveedor": supplier
        }
        
    except json.JSONDecodeError as e:
        logger.error(f"Error JSON OpenAI: {e}")
        logger.error(f"Contenido recibido: {content}")
        return extract_with_regex_improved(text)
        
    except Exception as e:
        logger.error(f"Error OpenAI: {e}")
        return extract_with_regex_improved(text)

def extract_with_regex_improved(text: str) -> dict:
    """Extracción mejorada usando regex con prioridades"""
    
    result = {"nro_factura": "No encontrado", "proveedor": "No encontrado"}
    
    # 1. Buscar número de factura por prioridad
    for pattern, priority in INVOICE_PATTERNS_PRIORITY:
        matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
        for match in matches:
            if is_valid_invoice_number(match):
                if result["nro_factura"] == "No encontrado" or priority > score_invoice_improved(result["nro_factura"]):
                    result["nro_factura"] = match
                    logger.info(f"Factura encontrada (prioridad {priority}): {match}")
    
    # 2. Buscar proveedor
    supplier = find_supplier_in_text(text)
    if supplier != "No encontrado":
        result["proveedor"] = supplier
    
    return result

# =========================
# MEJORA 8: Procesamiento de archivos mejorado
# =========================
def process_file_improved(file, widget_key: str = "") -> dict:
    """Procesamiento mejorado de archivos"""
    
    st.session_state["current_filename"] = file.name
    
    # Crear archivo temporal
    with tempfile.NamedTemporaryFile(delete=False, suffix=file.name) as tmp:
        tmp.write(file.read())
        tmp_path = pathlib.Path(tmp.name)
    
    try:
        # Extraer texto según tipo de archivo
        if tmp_path.suffix.lower() == ".pdf":
            # PDF: intentar texto nativo primero
            doc = fitz.open(tmp_path)
            extracted_text = ""
            
            for page_num, page in enumerate(doc):
                # Intentar texto nativo
                native_text = page.get_text("text").strip()
                
                if len(native_text) > 100:  # Texto suficiente
                    extracted_text += native_text + "\n"
                    logger.info(f"PDF página {page_num}: texto nativo extraído")
                else:
                    # Fallback a OCR si Tesseract disponible
                    if TES_AVAILABLE:
                        try:
                            pix = page.get_pixmap(dpi=300)
                            img_data = pix.tobytes()
                            pil_img = Image.open(io.BytesIO(img_data))
                            ocr_text = extract_text_robust(pil_img)
                            extracted_text += ocr_text + "\n"
                            logger.info(f"PDF página {page_num}: OCR aplicado")
                        except Exception as e:
                            logger.error(f"Error OCR en página {page_num}: {e}")
                            extracted_text += native_text + "\n"
                    else:
                        extracted_text += native_text + "\n"
            
            doc.close()
            
        else:
            # Archivo de imagen
            if not TES_AVAILABLE:
                return {
                    "archivo": file.name,
                    "nro_factura": "Error: Tesseract no disponible para imágenes",
                    "proveedor": "Error: Tesseract no disponible para imágenes"
                }
            
            try:
                pil_image = Image.open(tmp_path)
                
                # Corregir rotación EXIF si existe
                if hasattr(pil_image, '_getexif') and pil_image._getexif():
                    exif = pil_image._getexif()
                    orientation = exif.get(274)
                    if orientation == 3:
                        pil_image = pil_image.rotate(180, expand=True)
                    elif orientation == 6:
                        pil_image = pil_image.rotate(270, expand=True)
                    elif orientation == 8:
                        pil_image = pil_image.rotate(90, expand=True)
                
                extracted_text = extract_text_robust(pil_image)
                
            except Exception as e:
                return {
                    "archivo": file.name,
                    "nro_factura": f"Error procesando imagen: {e}",
                    "proveedor": f"Error procesando imagen: {e}"
                }
    
    except Exception as e:
        return {
            "archivo": file.name,
            "nro_factura": f"Error general: {e}",
            "proveedor": f"Error general: {e}"
        }
    
    finally:
        # Limpiar archivo temporal
        try:
            tmp_path.unlink()
        except:
            pass
    
    # Verificar que se extrajo texto
    if not extracted_text or len(extracted_text.strip()) < 10:
        return {
            "archivo": file.name,
            "nro_factura": "Error: No se pudo extraer texto del archivo",
            "proveedor": "Error: No se pudo extraer texto del archivo"
        }
    
    # Debug: mostrar texto extraído
    with st.expander(f"🔍 Texto extraído de {file.name}"):
        st.text_area(
            "Contenido:",
            extracted_text[:2000] + ("..." if len(extracted_text) > 2000 else ""),
            height=300,
            key=f"debug_text_{widget_key or file.name}"
        )
    
    # Extraer datos
    result = extract_with_openai_improved(extracted_text, file.name)
    
    # Agregar información del archivo
    result["archivo"] = file.name
    
    # Log resultado
    logger.info(f"Resultado final para {file.name}: {result}")
    
    return result

# =========================
# UI MEJORADA
# =========================
st.title("📄 Lector de Facturas Mejorado - OCR + OpenAI")
st.markdown("**Versión optimizada para Streamlit Cloud** - Extrae números de factura y proveedores con mayor precisión")

# Información del sistema
col1, col2 = st.columns(2)
with col1:
    if TES_AVAILABLE:
        st.success("🧠 **Tesseract OCR**: ✅ Disponible")
        st.info("✨ Soporte completo para PDFs e imágenes")
    else:
        st.warning("🧠 **Tesseract OCR**: ⚠️ No disponible")
        st.info("📝 Solo PDFs con texto nativo")

with col2:
    st.success("🤖 **OpenAI**: ✅ Conectado")
    st.info("🧪 Análisis inteligente activado")

# Test de conectividad
if st.sidebar.button("🧪 Test Completo del Sistema"):
    with st.spinner("Probando sistema..."):
        # Test OpenAI
        try:
            test_response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Responde solo: OK"}],
                max_tokens=10
            )
            st.sidebar.success("✅ OpenAI: Funcionando")
        except Exception as e:
            st.sidebar.error(f"❌ OpenAI: {e}")
        
        # Test Tesseract
        if TES_AVAILABLE:
            try:
                test_img = Image.new('RGB', (200, 100), color='white')
                test_result = pytesseract.image_to_string(test_img)
                st.sidebar.success("✅ Tesseract: Funcionando")
            except Exception as e:
                st.sidebar.error(f"❌ Tesseract: {e}")
        else:
            st.sidebar.warning("⚠️ Tesseract: No disponible")

# Consejos mejorados
with st.expander("💡 Consejos para mejores resultados"):
    st.markdown("""
    **📱 Para imágenes:**
    - Usa buena iluminación y evita sombras
    - Mantén la factura recta (sin rotación)  
    - Resolución mínima recomendada: 1200x800px
    
    **📄 Para PDFs:**
    - PDFs con texto seleccionable dan mejores resultados
    - Si es un PDF escaneado, asegúrate de buena calidad
    
    **🎯 El sistema busca:**
    - **Números de factura**: FV-123, A-V2025-123, C2025000851658
    - **Proveedores**: Empresas emisoras (S.L., S.A.), no clientes
    """)

# Botón limpiar
if st.button("🗑️ Limpiar resultados anteriores"):
    for key in list(st.session_state.keys()):
        if key.startswith(('debug_text_', 'current_filename')):
            del st.session_state[key]
    st.rerun()

# Subida de archivos
uploaded_files = st.file_uploader(
    "Selecciona tus facturas",
    type=["pdf", "png", "jpg", "jpeg"],
    accept_multiple_files=True,
    help="Sube PDFs o imágenes de facturas. Máximo recomendado: 10 archivos"
)

if uploaded_files:
    st.subheader(f"📊 Procesando {len(uploaded_files)} archivo(s)")
    
    # Barra de progreso
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Contenedor para resultados
    results_container = st.container()
    
    results = []
    
    # Procesar cada archivo
    for i, uploaded_file in enumerate(uploaded_files):
        status_text.text(f"⚙️ Procesando {uploaded_file.name}... ({i+1}/{len(uploaded_files)})")
        
        try:
            result = process_file_improved(uploaded_file, f"file_{i}")
            results.append(result)
            
            # Actualizar progreso
            progress_bar.progress((i + 1) / len(uploaded_files))
            
        except Exception as e:
            logger.error(f"Error procesando {uploaded_file.name}: {e}")
            results.append({
                "archivo": uploaded_file.name,
                "nro_factura": f"Error: {str(e)[:100]}",
                "proveedor": f"Error: {str(e)[:100]}"
            })
    
    status_text.text("✅ ¡Procesamiento completado!")
    
    # Mostrar resultados
    if results:
        df = pd.DataFrame(results)
        
        # Métricas
        st.subheader("📈 Resumen de Resultados")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_files = len(results)
            st.metric("📁 Total archivos", total_files)
        
        with col2:
            facturas_ok = sum(1 for r in results 
                            if r.get("nro_factura", "No encontrado") not in ["No encontrado", ""] 
                            and not r.get("nro_factura", "").startswith("Error"))
            st.metric("📄 Facturas detectadas", facturas_ok, 
                     delta=f"{facturas_ok/total_files*100:.0f}%" if total_files > 0 else "0%")
        
        with col3:
            proveedores_ok = sum(1 for r in results 
                               if r.get("proveedor", "No encontrado") not in ["No encontrado", ""] 
                               and not r.get("proveedor", "").startswith("Error"))
            st.metric("🏢 Proveedores detectados", proveedores_ok,
                     delta=f"{ 
















