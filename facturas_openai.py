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

# =========================
# Configuración inicial
# =========================
st.set_page_config(page_title="OCR + OpenAI Facturas", layout="wide")

# =========================
# Helpers
# =========================
def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(c for c in text if not unicodedata.combining(c))
    return text.upper().strip()

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
# Validaciones y patrones
# =========================
# Palabras que invalidan un candidato si aparecen cerca o dentro
INVALID_INVOICE_WORDS = [
    'CLIENTE','CUSTOMER','DESTINATARIO',
    'FECHA','DATE',
    'TELEFONO','TELÉFONO','TEL','TFNO','PHONE','MÓVIL','MOVIL','WHATSAPP','FAX',
    'EMAIL','CORREO',
    'DIRECCION','ADDRESS',
    'CIF','NIF','VAT','IVA',
    'IBAN','CUENTA','ACCOUNT',
    'ALBARAN','ALBARÁN','TICKET','PEDIDO','ORDER','REFERENCIA'
]

INVOICE_CONTEXT_WORDS = ['FACTURA','FAC','INVOICE','FACTURE','BILL','Nº','N°','NO','NUMERO','NÚMERO','#']

INVOICE_PATTERNS = [
    r'\b(?:FACT(?:URA)?\.?\s*(?:N[ºo°]\s*)?\.?\s*)([A-Z0-9][A-Z0-9\-/\.]{3,20})\b',
    r'\b([A-Z]{1,4}[-/]\d{4,12})\b',             # FV-123456, A-123456
    r'\b([A-Z]{2,5}\d{3,12})\b',                 # INV123456
    r'\b(FV[-/]?\d{1,3}[-/]?\d{4,12})\b',        # FV-0-123456
    r'\b([A-Z]\d{7,15})\b',                      # C12345678...
    r'\b([A-Z0-9]{1,4}[-/][A-Z0-9]{1,4}[-/][A-Z0-9]{3,12})\b',
    r'\b(\d{7,12})\b',                           # solo dígitos largos (con penalización si no hay contexto)
    r'\b(\d{4}[-/]\d{6,10})\b',                  # 2024-123456
]

# CIF/NIF español típicos (para excluir)
CIF_NIF_PATTERNS = [
    r'^\d{8}[A-Z]$',        # DNI/NIF: 12345678Z
    r'^[ABCDEFGHJNPQRSUVW]\d{7}[0-9A-J]$',  # CIF
]

# Teléfono ES típico (para excluir): 9–11 dígitos puros o +34 ...
def looks_like_phone(candidate: str, ctx_norm: str) -> bool:
    if re.match(r'^\+?\d[\d\s\-]{8,15}$', candidate):
        return True
    if candidate.isdigit() and 9 <= len(candidate) <= 11:
        # si además el contexto sugiere teléfono
        if any(w in ctx_norm for w in ['TEL','TFNO','PHONE','MOVIL','MÓVIL','WHATSAPP','FAX']):
            return True
    return False

def looks_like_cif_nif(candidate: str) -> bool:
    for pat in CIF_NIF_PATTERNS:
        if re.match(pat, candidate):
            return True
    return False

def score_invoice_number(candidate: str, context: str = "") -> int:
    """Mayor score = mejor candidato."""
    if not candidate:
        return 0
    ctx_norm = normalize_text(context)
    cand = candidate.strip()

    # Base
    score = 0

    # Fuerte evidencia de contexto de factura
    if any(w in ctx_norm for w in INVOICE_CONTEXT_WORDS):
        score += 12

    # Penalizaciones por falsos positivos
    if looks_like_phone(cand, ctx_norm):
        return 0  # descartar
    if looks_like_cif_nif(cand):
        return 0  # descartar
    if any(w in normalize_text(cand) for w in INVALID_INVOICE_WORDS):
        return 0

    # Heurísticas de forma
    if re.match(r'^[A-Z]+[-/]?\d+$', cand):
        score += 6
    if re.match(r'^\d{7,}$', cand):
        score += 2
        # Si son solo dígitos y NO hay contexto de factura → penaliza fuerte
        if not any(w in ctx_norm for w in INVOICE_CONTEXT_WORDS):
            score -= 8
    if '-' in cand or '/' in cand or '.' in cand:
        score += 2
    if any(ch.isalpha() for ch in cand):
        score += 2

    # Longitudes raras
    if len(cand) < 4 or len(cand) > 25:
        score -= 4

    return score

# =========================
# OCR
# =========================
def preprocess_image_advanced(pil_img):
    img_array = np.array(pil_img.convert('RGB'))
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    denoised = cv2.medianBlur(gray, 3)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)
    # Umbral adaptativo ayuda a texto tenue
    binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    return binary

def extract_text_with_ocr(pil_img) -> str:
    if not TES_AVAILABLE:
        raise TesseractNotFoundError("Tesseract no está disponible")
    processed = preprocess_image_advanced(pil_img)
    processed_pil = Image.fromarray(processed)
    # Probar varios PSM y quedarnos con el mejor
    best, best_score = "", -1
    for psm in [6, 4, 11, 12, 3]:
        try:
            cfg = f'--oem 3 --psm {psm} -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÁÉÍÓÚáéíóúñÑ/-.'
            txt = pytesseract.image_to_string(processed_pil, lang='spa+eng', config=cfg)
            sc = len(txt) + txt.upper().count('FACTURA') * 10 + txt.upper().count('INVOICE') * 10
            if sc > best_score:
                best, best_score = txt, sc
        except Exception:
            pass
    return best

# =========================
# Extracción con regex + contexto
# =========================
def extract_invoice_number(text: str) -> str:
    if not text:
        return "No encontrado"
    candidates = []
    for pat in INVOICE_PATTERNS:
        for m in re.finditer(pat, text, re.IGNORECASE | re.MULTILINE):
            cand = m.group(1) if m.groups() else m.group(0)
            # contexto local
            start = max(0, m.start() - 60)
            end = min(len(text), m.end() + 60)
            ctx = text[start:end]
            s = score_invoice_number(cand, ctx)
            if s > 0:
                candidates.append((cand.strip(), s))
    # Barrido por líneas: buscar cerca de palabras clave
    lines = text.splitlines()
    for i, line in enumerate(lines):
        ln = normalize_text(line)
        if any(w in ln for w in INVOICE_CONTEXT_WORDS):
            neigh = [line]
            if i + 1 < len(lines): neigh.append(lines[i+1])
            for sline in neigh:
                for match in re.finditer(r'[A-Z0-9][A-Z0-9\-/\.]{3,20}', sline):
                    cand = match.group(0)
                    s = score_invoice_number(cand, line)
                    if s > 0:
                        candidates.append((cand.strip(), s))
    if not candidates:
        return "No encontrado"
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]

# Detección robusta de proveedor (emisor)
KNOWN_SUPPLIERS_PATTERNS = [
    r'\b([A-ZÁÉÍÓÚÑ0-9&\.\-\s]{2,}?S\.?A\.?U?)\b',
    r'\b([A-ZÁÉÍÓÚÑ0-9&\.\-\s]{2,}?S\.?L\.?U?)\b',
    r'\b([A-ZÁÉÍÓÚÑ0-9&\.\-\s]{2,}?LIMITADA)\b',
    r'\b([A-ZÁÉÍÓÚÑ0-9&\.\-\s]{2,}?GMBH)\b',
    r'\b([A-ZÁÉÍÓÚÑ0-9&\.\-\s]{2,}?SAS)\b',
    r'\b([A-ZÁÉÍÓÚÑ0-9&\.\-\s]{2,}?LTD)\b',
]
KNOWN_SUPPLIERS_MAP = {
    'OUIGO': 'OUIGO ESPAÑA S.A.U.',
    'SUPRACAFE':'SUPRACAFE',
    'SUPRACAFÉ':'SUPRACAFE',
    'MERCADONA':'MERCADONA S.A.',
    'CARREFOUR':'CARREFOUR',
    'MAKRO':'MAKRO',
    'DIA':'DIA S.A.',
    'LIDL':'LIDL',
    'EROSKI':'EROSKI',
    'EHOSA':'EHOSA',
    'COCA COLA':'COCA-COLA',
}

def extract_supplier(text: str) -> str:
    if not text:
        return "No encontrado"
    tnorm = normalize_text(text)

    # 1) map directo rápido
    for k, v in KNOWN_SUPPLIERS_MAP.items():
        if k in tnorm:
            return v

    # 2) patrones corporativos en primeras líneas
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    skip_terms = {'FACTURA','INVOICE','FECHA','DATE','CLIENTE','CUSTOMER','DESTINATARIO',
                  'TELEFONO','TELÉFONO','PHONE','EMAIL','DIRECCION','ADDRESS',
                  'CIF','NIF','VAT','IVA','IBAN','CUENTA','TOTAL','BASE','IVA','HTTP','WWW'}
    for line in lines[:35]:
        u = normalize_text(line)
        if any(t in u for t in skip_terms):  # evita bloques de cliente/datos fiscales
            continue
        for pat in KNOWN_SUPPLIERS_PATTERNS:
            m = re.search(pat, line)
            if m:
                cand = re.sub(r'\s+', ' ', m.group(1)).strip(' -.,')
                return cand[:60] + '...' if len(cand) > 60 else cand
        # Nombre comercial en mayúsculas sin parecer dirección ni importe
        if 4 <= len(u) <= 60 and re.match(r'^[A-ZÁÉÍÓÚÑ0-9&\-\.\s]+$', u):
            if not re.search(r'\b(CALLE|AVDA|C/|€|EUROS?|IVA|FECHA|TEL)\b', u):
                return line
    return "No encontrado"

# =========================
# OpenAI (combinado, sin UI de debug)
# =========================
def extract_data_with_openai(text: str) -> dict:
    regex_invoice = extract_invoice_number(text)
    regex_supplier = extract_supplier(text)

    lines = [ln for ln in text.split('\n') if ln.strip()]
    relevant_text = '\n'.join(lines[:80])

    prompt = f"""
Eres experto en facturas españolas. Devuelve SOLO JSON con estas claves:
- "nro_factura": código único de la factura (puede tener letras/números/guiones)
- "proveedor": empresa EMISORA (no el cliente)

Reglas:
- No uses NIF/CIF, teléfono, fecha o IBAN como número de factura.
- Si no hay dato, usa "No encontrado".

TEXTO:
{relevant_text}

Respuesta JSON:
"""
    try:
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Devuelve solo JSON válido."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=180
        )
        content = resp.choices[0].message.content.strip()
        if content.startswith("```json"):
            content = content.replace("```json","").replace("```","").strip()
        data = json.loads(content)
        ai_invoice = data.get("nro_factura") or "No encontrado"
        ai_supplier = data.get("proveedor") or "No encontrado"

        # Combinar con regex/heurística
        final_invoice = ai_invoice
        if final_invoice == "No encontrado" and regex_invoice != "No encontrado":
            final_invoice = regex_invoice
        else:
            # si ambos existen, elige por score
            if regex_invoice != "No encontrado":
                s_regex = score_invoice_number(regex_invoice, relevant_text)
                s_ai = score_invoice_number(ai_invoice, relevant_text)
                if s_regex > s_ai:
                    final_invoice = regex_invoice

        final_supplier = ai_supplier if ai_supplier != "No encontrado" else regex_supplier
        return {"nro_factura": final_invoice, "proveedor": final_supplier}

    except Exception:
        return {"nro_factura": regex_invoice, "proveedor": regex_supplier}

# =========================
# Lectura de archivos (PDF/imagen)
# =========================
def extract_text_from_file(file_path: pathlib.Path) -> str:
    try:
        if file_path.suffix.lower() == '.pdf':
            doc = fitz.open(file_path)
            parts = []
            for page in doc:
                txt = page.get_text("text") or ""
                if len(txt.strip()) < 50:
                    # OCR si la capa de texto es pobre
                    if TES_AVAILABLE:
                        pix = page.get_pixmap(dpi=300)
                        img = Image.open(io.BytesIO(pix.tobytes()))
                        txt = extract_text_with_ocr(img)
                parts.append(txt.strip())
            doc.close()
            return "\n".join([p for p in parts if p])
        else:
            img = Image.open(file_path)
            # girar por EXIF si hace falta
            if hasattr(img, "_getexif") and img._getexif():
                orientation = img._getexif().get(274)
                if orientation == 3: img = img.rotate(180, expand=True)
                elif orientation == 6: img = img.rotate(270, expand=True)
                elif orientation == 8: img = img.rotate(90, expand=True)
            return extract_text_with_ocr(img)
    except Exception as e:
        return f""

def process_single_file(file) -> dict:
    with tempfile.NamedTemporaryFile(delete=False, suffix=pathlib.Path(file.name).suffix) as tmp:
        tmp.write(file.read())
        tmp_path = pathlib.Path(tmp.name)
    try:
        text = extract_text_from_file(tmp_path)
        if not text or not text.strip():
            return {"nro_factura": "No encontrado", "proveedor": "No encontrado"}
        return extract_data_with_openai(text)
    finally:
        try:
            tmp_path.unlink()
        except Exception:
            pass

# =========================
# Generación de PDF (siempre BytesIO) y descargas
# =========================
def create_pdf_report(df: pd.DataFrame):
    """Devuelve BytesIO con el reporte; si falla, BytesIO con PDF mínimo."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors
        from reportlab.lib.units import inch

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4,
                                rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=24)
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle('Title', parent=styles['Heading1'],
                                     fontSize=18, spaceAfter=20, alignment=1)

        elements = [Paragraph("REPORTE DE FACTURAS (Nº + PROVEEDOR)", title_style),
                    Spacer(1, 12)]
        data = [["NÚMERO FACTURA", "PROVEEDOR"]]
        for _, row in df.iterrows():
            data.append([str(row.get("nro_factura","")).strip(),
                         str(row.get("proveedor","")).strip()])
        table = Table(data, colWidths=[2.8*inch, 3.2*inch])
        table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT')
        ]))
        elements.append(table)
        doc.build(elements)
        buffer.seek(0)
        return buffer
    except Exception:
        try:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 10, 'REPORTE DE FACTURAS', ln=True, align='C')
            pdf.set_font('Arial', '', 10)
            for _, row in df.iterrows():
                pdf.cell(95, 8, str(row.get("nro_factura","")), 1)
                pdf.cell(95, 8, str(row.get("proveedor","")), 1, ln=True)
            out = pdf.output(dest="S")
            if not isinstance(out, (bytes, bytearray)):
                out = out.encode("latin-1")
            return io.BytesIO(out)
        except Exception:
            return io.BytesIO(b"%PDF-1.4\n%EOF")

def bytes_for_streamlit(data_obj):
    """Convierte a bytes seguros para st.download_button."""
    if hasattr(data_obj, "getvalue"):
        return data_obj.getvalue()
    if isinstance(data_obj, memoryview):
        return data_obj.tobytes()
    if isinstance(data_obj, (bytes, bytearray)):
        return data_obj
    return b"%PDF-1.4\n%EOF"

# =========================
# UI
# =========================
st.title("📄 Lector de Facturas - OCR + OpenAI")
uploaded_files = st.file_uploader(
    "Selecciona archivos", type=['pdf','png','jpg','jpeg'], accept_multiple_files=True
)

if uploaded_files:
    if st.button("🚀 Procesar Facturas", type="primary"):
        progress = st.progress(0)
        results = []
        for i, file in enumerate(uploaded_files):
            results.append(process_single_file(file))
            progress.progress((i+1)/len(uploaded_files))
        st.success("✅ Procesamiento completado")

        df = pd.DataFrame(results)[['nro_factura','proveedor']]
        st.subheader("📊 Datos extraídos")
        st.dataframe(df, use_container_width=True)

        st.subheader("📥 Descargas")
        # Excel
        xls_buf = io.BytesIO()
        df.to_excel(xls_buf, index=False, engine='openpyxl')
        xls_buf.seek(0)
        st.download_button(
            "📊 Descargar Excel",
            data=xls_buf.getvalue(),
            file_name="facturas.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        # PDF (conversión segura a bytes)
        pdf_buf = create_pdf_report(df)
        pdf_bytes = bytes_for_streamlit(pdf_buf)
        st.download_button(
            "📑 Descargar PDF",
            data=pdf_bytes,
            file_name="facturas.pdf",
            mime="application/pdf"
        )




