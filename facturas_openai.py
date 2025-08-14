# --- Lector de Facturas con OCR + OpenAI (UI simplificada) ---

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

# ✅ Config básica
st.set_page_config(page_title="OCR + OpenAI Facturas", layout="wide")

# =========================
# PDF (desde DataFrame con 2 columnas)
# =========================
def create_pdf_report(df: pd.DataFrame) -> bytes:
    """Crea un PDF con las columnas nro_factura y proveedor."""
    buffer = io.BytesIO()
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors
        from reportlab.lib.units import inch

        doc = SimpleDocTemplate(
            buffer, pagesize=A4,
            rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=24
        )
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle', parent=styles['Heading1'], fontSize=18, spaceAfter=20, alignment=1
        )

        elements = []
        elements.append(Paragraph("REPORTE DE FACTURAS (Nº + PROVEEDOR)", title_style))
        elements.append(Spacer(1, 12))

        data = [["NÚMERO FACTURA", "PROVEEDOR"]]
        for _, row in df.iterrows():
            factura = str(row.get("nro_factura", "")).strip()
            proveedor = str(row.get("proveedor", "")).strip()
            if len(factura) > 35: factura = factura[:32] + "..."
            if len(proveedor) > 50: proveedor = proveedor[:47] + "..."
            data.append([factura, proveedor])

        table = Table(data, colWidths=[2.5*inch, 3.5*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#4A4A4A")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor("#F6F1E1")),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        elements.append(table)
        doc.build(elements)
        buffer.seek(0)
        return buffer.getvalue()

    except Exception:
        # Fallback FPDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'REPORTE DE FACTURAS (Nº + PROVEEDOR)', ln=True, align='C')
        pdf.ln(8)
        pdf.set_font('Arial', 'B', 11)
        pdf.cell(90, 8, 'NÚMERO FACTURA', 1, 0, 'C')
        pdf.cell(90, 8, 'PROVEEDOR', 1, 1, 'C')
        pdf.set_font('Arial', '', 10)
        for _, row in df.iterrows():
            factura = str(row.get("nro_factura", ""))
            proveedor = str(row.get("proveedor", ""))
            factura = (factura[:42] + "...") if len(factura) > 45 else factura
            proveedor = (proveedor[:42] + "...") if len(proveedor) > 45 else proveedor
            pdf.cell(90, 8, factura, 1, 0, 'L')
            pdf.cell(90, 8, proveedor, 1, 1, 'L')
        out = pdf.output(dest="S")
        return out if isinstance(out, (bytes, bytearray)) else out.encode("latin-1")

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
    try:
        for key_name in ["openai_api_key","OPENAI_API_KEY","openai-api-key","openai_key","OPENAI_KEY","api_key"]:
            try:
                api_key = st.secrets[key_name]
                return OpenAI(api_key=api_key)
            except KeyError:
                continue
    except Exception:
        pass
    st.error("❌ No se encontró la API key de OpenAI")
    st.stop()

client = configurar_openai()

# =========================
# Patrones mejorados
# =========================
INVOICE_PATTERNS = [
    r'\b(?:FACT(?:URA)?\.?\s*N[ºo°]?\.?\s*|N[ºo°]\.?\s*FACT(?:URA)?\.?\s*|INVOICE\s*(?:N[ºo°])?\.?\s*|FACTURE\s*N[ºo°]\.?\s*)([A-Z0-9][A-Z0-9\-/\.]{3,20})\b',
    r'\b([A-Z]{1,4}[-/]\d{4,12})\b',
    r'\b([A-Z]\d{8,15})\b',
    r'\b(FV[-/]?\d{1,3}[-/]?\d{4,12})\b',
    r'\b([A-Z]{2,5}\d{3,12})\b',
    r'\b(\d{7,12})\b',
    r'\b([A-Z0-9]{1,4}[-/][A-Z0-9]{1,4}[-/][A-Z0-9]{3,12})\b',
    r'\b(\d{4}[-/]\d{6,10})\b',
]
INVALID_INVOICE_WORDS = ['CLIENTE','CLIENT','CUSTOMER','DESTINATARIO','FECHA','DATE','TELEFONO','TELÉFONO','PHONE','EMAIL','DIRECCION','ADDRESS','CIF','NIF','VAT','IVA']

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
    r'\b([A-ZÁÉÍÓÚÑ][A-Za-záéíóúñ\s]{2,}\s+S\.?[LA]\.?)\b',
]

# =========================
# Utilidades
# =========================
def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(c for c in text if not unicodedata.combining(c))
    return text.upper().strip()

def is_valid_invoice_number(candidate: str) -> bool:
    if not candidate or len(candidate) < 3:
        return False
    candidate_norm = normalize_text(candidate)
    if any(word in candidate_norm for word in INVALID_INVOICE_WORDS):
        return False
    if candidate.isdigit() and len(candidate) < 6:
        return False
    if re.match(r'^\d{8}[A-Z]$', candidate) or re.match(r'^[A-Z]\d{7}[0-9A-J]$', candidate):
        return False
    if re.match(r'^\+?[\d\s\-]{9,15}$', candidate):
        return False
    if re.match(r'^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$', candidate):
        return False
    return True

def score_invoice_number(candidate: str, context: str = "") -> int:
    if not candidate:
        return 0
    score = 0
    context_norm = normalize_text(context)
    if any(word in context_norm for word in ['FACTURA', 'INVOICE', 'FACT']):
        score += 10
    if re.match(r'^[A-Z]+[-/]?\d+$', candidate):
        score += 5
    if re.match(r'^\d{7,}$', candidate):
        score += 3
    if '-' in candidate or '/' in candidate:
        score += 2
    if any(char.isalpha() for char in candidate):
        score += 2
    if len(candidate) < 4 or len(candidate) > 25:
        score -= 3
    return score

# =========================
# OCR mejorado
# =========================
def preprocess_image_advanced(pil_img):
    img_array = np.array(pil_img.convert('RGB'))
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    denoised = cv2.medianBlur(gray, 3)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)
    binary = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    kernel = np.ones((1,1), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return cleaned

def extract_text_with_ocr(pil_img) -> str:
    if not TES_AVAILABLE:
        raise TesseractNotFoundError("Tesseract no está disponible")
    processed_img = preprocess_image_advanced(pil_img)
    processed_pil = Image.fromarray(processed_img)
    best_text, best_score = "", 0
    for psm in [6, 4, 11, 12, 3]:
        try:
            config = f'--oem 3 --psm {psm} -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÁÉÍÓÚáéíóúñÑ/-.'
            text = pytesseract.image_to_string(processed_pil, lang='spa+eng', config=config)
            score = len(text) + text.upper().count('FACTURA') * 10 + text.upper().count('INVOICE') * 10
            if score > best_score:
                best_text, best_score = text, score
        except Exception:
            pass
    return best_text

# =========================
# Extracción (regex + heurística)
# =========================
def extract_invoice_number(text: str) -> str:
    if not text:
        return "No encontrado"
    candidates = []
    for pattern in INVOICE_PATTERNS:
        for m in re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE):
            candidate = m.group(1) if m.groups() else m.group(0)
            if is_valid_invoice_number(candidate):
                start = max(0, m.start() - 50)
                end = min(len(text), m.end() + 50)
                context = text[start:end]
                score = score_invoice_number(candidate, context)
                candidates.append((candidate, score))
    lines = text.split('\n')
    for i, line in enumerate(lines):
        line_norm = normalize_text(line)
        if any(word in line_norm for word in ['FACTURA', 'INVOICE', 'FACT']):
            search_lines = [line]
            if i + 1 < len(lines):
                search_lines.append(lines[i + 1])
            for sline in search_lines:
                for num in re.findall(r'[A-Z0-9][A-Z0-9\-/\.]{3,20}', sline, re.IGNORECASE):
                    if is_valid_invoice_number(num):
                        score = score_invoice_number(num, line)
                        candidates.append((num, score))
    if candidates:
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]
    return "No encontrado"

def extract_supplier(text: str) -> str:
    if not text:
        return "No encontrado"

    text_upper = normalize_text(text)
    known_suppliers_map = {
        'OUIGO': 'OUIGO ESPAÑA S.A.U.',
        'SUPRACAFE': 'SUPRACAFE',
        'SUPRACAFÉ': 'SUPRACAFE',
        'MERCADONA': 'MERCADONA S.A.',
        'CARREFOUR': 'CARREFOUR',
        'MAKRO': 'MAKRO',
        'DIA': 'DIA S.A.',
        'LIDL': 'LIDL',
        'EROSKI': 'EROSKI',
        'EHOSA': 'EHOSA',
        'MAREGALEVILLA': 'MAREGALEVILLA',
        'COCA COLA': 'COCA-COLA'
    }
    for k, v in known_suppliers_map.items():
        if k in text_upper:
            return v

    for pat in KNOWN_SUPPLIERS_PATTERNS:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            cand = m.group(1)
            cand = re.sub(r'\s+', ' ', cand).strip()
            return cand

    lines = [ln.strip() for ln in text.split('\n') if ln.strip()]
    skip_terms = {'FACTURA','INVOICE','FECHA','DATE','CLIENTE','CUSTOMER','TELEFONO','TELÉFONO',
                  'EMAIL','DIRECCION','ADDRESS','CIF','NIF','PUEDES PAGAR','PAGAR ONLINE',
                  'SIMPLIFICADA','HTTP','WWW','PÁGINA','PAGINA','IVA'}
    for line in lines[:20]:
        u = normalize_text(line)
        if any(t in u for t in skip_terms):
            continue
        if (re.search(r'\bS\.?A\.?U?\.?\b', u) or
            re.search(r'\bS\.?L\.?U?\.?\b', u) or
            re.search(r'\bLIMITADA\b', u) or
            re.search(r'\bBV\b', u) or
            re.search(r'\bGMBH\b', u) or
            re.search(r'\bSAS\b', u) or
            re.search(r'\bLTD\b', u)):
            cleaned = re.sub(r'\s+', ' ', line).strip()
            return cleaned[:60] + '...' if len(cleaned) > 60 else cleaned
        if 4 <= len(u) <= 50 and re.match(r'^[A-ZÁÉÍÓÚÑ0-9&\-\.\s]+$', u):
            if not re.search(r'\bCALLE\b|\bAVDA\b|\bC/\b|\bEUROS?\b|\b€\b|\d{2}[/-]\d{2}[/-]\d{2,4}', u):
                return line
    return "No encontrado"

# =========================
# OpenAI (combinado, sin debug visual)
# =========================
def extract_data_with_openai_improved(text: str) -> dict:
    regex_invoice = extract_invoice_number(text)
    regex_supplier = extract_supplier(text)

    lines = text.split('\n')
    relevant_text = '\n'.join([line.strip() for line in lines[:50] if line.strip()])

    prompt = f"""
Eres un experto analizando facturas españolas. Extrae EXACTAMENTE estos datos:
1. NÚMERO DE FACTURA
2. PROVEEDOR (emisor)

Reglas:
- No NIF/CIF/teléfono/fechas como número de factura
- Si no hay dato, usa "No encontrado"
- Responde SOLO en JSON

TEXTO:
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
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "").strip()
        result = json.loads(content)

        ai_invoice = result.get("invoiceNumber", result.get("numero_factura", "No encontrado"))
        ai_supplier = result.get("supplier", result.get("proveedor", "No encontrado"))

        if ai_supplier and ai_supplier != "No encontrado":
            ai_supplier = re.sub(r'\s+', ' ', ai_supplier.replace('\n', ' ').replace('\r', ' ')).strip()
            if len(ai_supplier) > 60:
                ai_supplier = ai_supplier[:60].strip()

        final_invoice = ai_invoice
        if ai_invoice == "No encontrado" and regex_invoice != "No encontrado":
            final_invoice = regex_invoice
        elif (regex_invoice != "No encontrado" and
              score_invoice_number(regex_invoice) > score_invoice_number(ai_invoice)):
            final_invoice = regex_invoice

        final_supplier = ai_supplier if ai_supplier != "No encontrado" else regex_supplier

        return {"nro_factura": final_invoice, "proveedor": final_supplier}

    except Exception:
        return {"nro_factura": regex_invoice, "proveedor": regex_supplier}

# =========================
# I/O de archivos
# =========================
def extract_text_from_file(file_path: pathlib.Path) -> str:
    try:
        if file_path.suffix.lower() == '.pdf':
            doc = fitz.open(file_path)
            text_parts = []
            for page in doc:
                text = page.get_text("text").strip()
                if len(text) > 100:
                    text_parts.append(text)
                else:
                    if TES_AVAILABLE:
                        pix = page.get_pixmap(dpi=300)
                        img = Image.open(io.BytesIO(pix.tobytes()))
                        text_parts.append(extract_text_with_ocr(img))
                    else:
                        text_parts.append(text)
            doc.close()
            return '\n'.join(text_parts)
        else:
            img = Image.open(file_path)
            if hasattr(img, '_getexif'):
                exif = img._getexif()
                if exif:
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
    with tempfile.NamedTemporaryFile(delete=False, suffix=pathlib.Path(file.name).suffix) as tmp_file:
        tmp_file.write(file.read())
        tmp_path = pathlib.Path(tmp_file.name)
    try:
        text = extract_text_from_file(tmp_path)
        if not text or not text.strip():
            return {"archivo": file.name, "nro_factura": "Error: No se pudo extraer texto", "proveedor": "Error: No se pudo extraer texto"}
        result = extract_data_with_openai_improved(text)
        return {"archivo": file.name, **result}
    except Exception as e:
        return {"archivo": file.name, "nro_factura": f"Error: {str(e)}", "proveedor": f"Error: {str(e)}"}
    finally:
        try:
            tmp_path.unlink()
        except Exception:
            pass

# =========================
# UI
# =========================
st.title("📄 Lector de Facturas - OCR + OpenAI")
uploaded_files = st.file_uploader(
    "Selecciona archivos de facturas",
    type=['pdf', 'png', 'jpg', 'jpeg'],
    accept_multiple_files=True,
    help="Sube PDFs o imágenes de facturas"
)

if uploaded_files:
    if st.button("🚀 Procesar Facturas", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        results = []
        for i, file in enumerate(uploaded_files):
            status_text.text(f"Procesando {file.name}... ({i+1}/{len(uploaded_files)})")
            results.append(process_single_file(file, i))
            progress_bar.progress((i + 1) / len(uploaded_files))
        status_text.text("✅ ¡Procesamiento completado!")

        # DataFrame SOLO con columnas requeridas
        df = pd.DataFrame(results)[['nro_factura', 'proveedor']]

        st.subheader("📊 Datos extraídos")
        # Tabla limpia sin estilos que cambien color del texto
        st.dataframe(df, use_container_width=True)

        # Descargas (solo Excel y PDF)
        st.subheader("📥 Descargas")
        c1, c2 = st.columns(2)
        with c1:
            excel_buffer = io.BytesIO()
            df.to_excel(excel_buffer, index=False, engine='openpyxl')
            excel_buffer.seek(0)
            st.download_button(
                "📊 Descargar Excel",
                data=excel_buffer.getvalue(),
                file_name="facturas_procesadas.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        with c2:
            pdf_bytes = create_pdf_report(df)
            st.download_button(
                "📑 Descargar PDF",
                data=pdf_bytes,
                file_name="reporte_facturas.pdf",
                mime="application/pdf"
            )

# Limpiar sesión (opcional)
if st.sidebar.button("🗑️ Limpiar datos"):
    st.session_state.clear()
    st.rerun()




