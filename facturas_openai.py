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

# Configuración inicial
st.set_page_config(page_title="OCR + OpenAI Facturas", layout="wide")

# =========================
# PDF desde DataFrame
# =========================
def create_pdf_report(df: pd.DataFrame) -> bytes:
    """Genera un PDF con nro_factura y proveedor."""
    buffer = io.BytesIO()
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors
        from reportlab.lib.units import inch

        doc = SimpleDocTemplate(buffer, pagesize=A4,
                                rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=24)
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle('Title', parent=styles['Heading1'],
                                     fontSize=18, spaceAfter=20, alignment=1)

        elements = [Paragraph("REPORTE DE FACTURAS (Nº + PROVEEDOR)", title_style),
                    Spacer(1, 12)]

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
        # Fallback con FPDF
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
# Dependencias y Config OpenAI
# =========================
def has_tesseract() -> bool:
    try:
        _ = pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False

TES_AVAILABLE = has_tesseract()

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
# Utilidades OCR y Regex
# =========================
def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(c for c in text if not unicodedata.combining(c))
    return text.upper().strip()

INVALID_INVOICE_WORDS = ['CLIENTE','CUSTOMER','FECHA','DATE','TELEFONO','TELÉFONO','PHONE',
                         'EMAIL','DIRECCION','ADDRESS','CIF','NIF','VAT','IVA']

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
    return True

def extract_invoice_number(text: str) -> str:
    patterns = [
        r'\b(?:FACT(?:URA)?\.?\s*N[ºo°]?\.?\s*)([A-Z0-9][A-Z0-9\-/\.]{3,20})\b',
        r'\b([A-Z]{1,4}[-/]\d{4,12})\b',
        r'\b([A-Z]\d{8,15})\b',
        r'\b(\d{7,12})\b'
    ]
    candidates = []
    for pattern in patterns:
        for m in re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE):
            cand = m.group(1) if m.groups() else m.group(0)
            if is_valid_invoice_number(cand):
                candidates.append(cand)
    return candidates[0] if candidates else "No encontrado"

def extract_supplier(text: str) -> str:
    known = ['MERCADONA','CARREFOUR','MAKRO','DIA','LIDL','EROSKI','SUPRACAFE','COCA COLA']
    text_upper = normalize_text(text)
    for k in known:
        if k in text_upper:
            return k
    return "No encontrado"

# =========================
# OpenAI extracción
# =========================
def extract_data_with_openai(text: str) -> dict:
    regex_invoice = extract_invoice_number(text)
    regex_supplier = extract_supplier(text)

    lines = text.split('\n')
    relevant_text = '\n'.join(lines[:50])

    prompt = f"""
Extrae el número de factura y el proveedor (emisor) de la factura.
Responde solo JSON con 'nro_factura' y 'proveedor'. 
Si no hay dato, usa "No encontrado".

TEXTO:
{relevant_text}
"""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Eres un experto en análisis de facturas. Responde solo JSON válido."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=150
        )
        content = response.choices[0].message.content.strip()
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "").strip()
        result = json.loads(content)

        final_invoice = result.get("nro_factura", regex_invoice)
        final_supplier = result.get("proveedor", regex_supplier)
        return {"nro_factura": final_invoice or regex_invoice,
                "proveedor": final_supplier or regex_supplier}
    except:
        return {"nro_factura": regex_invoice, "proveedor": regex_supplier}

# =========================
# OCR
# =========================
def preprocess_image_advanced(pil_img):
    img_array = np.array(pil_img.convert('RGB'))
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    denoised = cv2.medianBlur(gray, 3)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)
    return enhanced

def extract_text_with_ocr(pil_img) -> str:
    if not TES_AVAILABLE:
        raise TesseractNotFoundError("Tesseract no está disponible")
    processed_img = preprocess_image_advanced(pil_img)
    processed_pil = Image.fromarray(processed_img)
    return pytesseract.image_to_string(processed_pil, lang='spa+eng')

def extract_text_from_file(file_path: pathlib.Path) -> str:
    if file_path.suffix.lower() == '.pdf':
        doc = fitz.open(file_path)
        text_parts = []
        for page in doc:
            text = page.get_text("text").strip()
            if not text:
                pix = page.get_pixmap(dpi=300)
                img = Image.open(io.BytesIO(pix.tobytes()))
                text = extract_text_with_ocr(img)
            text_parts.append(text)
        doc.close()
        return '\n'.join(text_parts)
    else:
        img = Image.open(file_path)
        return extract_text_with_ocr(img)

# =========================
# Procesar archivos
# =========================
def process_single_file(file) -> dict:
    with tempfile.NamedTemporaryFile(delete=False, suffix=pathlib.Path(file.name).suffix) as tmp_file:
        tmp_file.write(file.read())
        tmp_path = pathlib.Path(tmp_file.name)
    try:
        text = extract_text_from_file(tmp_path)
        return extract_data_with_openai(text)
    finally:
        tmp_path.unlink(missing_ok=True)

# =========================
# UI
# =========================
st.title("📄 Lector de Facturas - OCR + OpenAI")
uploaded_files = st.file_uploader("Selecciona archivos", type=['pdf','png','jpg','jpeg'], accept_multiple_files=True)

if uploaded_files:
    if st.button("🚀 Procesar Facturas", type="primary"):
        progress_bar = st.progress(0)
        results = []
        for i, file in enumerate(uploaded_files):
            results.append(process_single_file(file))
            progress_bar.progress((i+1)/len(uploaded_files))
        st.success("✅ Procesamiento completado")

        df = pd.DataFrame(results)
        st.subheader("📊 Datos extraídos")
        st.dataframe(df, use_container_width=True)

        st.subheader("📥 Descargas")
        # Excel
        excel_buffer = io.BytesIO()
        df.to_excel(excel_buffer, index=False, engine='openpyxl')
        excel_buffer.seek(0)
        st.download_button("📊 Descargar Excel", data=excel_buffer.getvalue(),
                           file_name="facturas.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        # PDF
        pdf_bytes = create_pdf_report(df)
        st.download_button("📑 Descargar PDF", data=pdf_bytes,
                           file_name="facturas.pdf", mime="application/pdf")




