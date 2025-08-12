# --- Lector de Facturas con OCR + OpenAI ---
# Cloud: OCR robusto (OpenCV + Tesseract multi-PSM), patrones fuertes Nº factura,
# filtros NIF/CIF/teléfono/fecha, filtro de localidades y frases geo, fuzzy de proveedor,
# normalizaciones (MAREGALEVILLA/SUPRACAFE/EHOSA), EHOSA contextual,
# keys únicos y PDF/Excel de salida.

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
# Reglas / Diccionarios
# =========================
STOPWORDS_SUPPLIER = {
    "FACTURA","FACT","INVOICE","ALBARAN","ALBARÁN","TICKET",
    "CLIENTE","DESTINATARIO","FACTURAR A","BILL TO","CUSTOMER",
    "FECHA","DATE","IVA","CIF","NIF","NIE","PAGINA","PÁGINA",
    "TEL","TLF","TLF.","TELEFONO","TELÉFONO","MOVIL","MÓVIL","FAX","EMAIL","E-MAIL","@","WEB","WWW","HTTP","HTTPS"
}

CLIENT_HINTS = [
    r'MISU\s+\d+\s+S\.?L\.?', r'\bMISU\b',
    r'RINCON\s+DE\s+LA\s+TORTILLA', r'EL\s+RINCON\s+DE\s+LA\s+TORTILLA',
    r'RESTAURANTE\s+[A-ZÁÉÍÓÚÑ]+', r'BAR\s+[A-ZÁÉÍÓÚÑ]+', r'CAFETERIA\s+[A-ZÁÉÍÓÚÑ]+',
    r'TABERNA\s+[A-ZÁÉÍÓÚÑ]+', r'ASADOR\s+[A-ZÁÉÍÓÚÑ]+', r'PIZZERIA\s+[A-ZÁÉÍÓÚÑ]+',
    r'CLIENTE:', r'CUSTOMER:', r'DESTINATARIO:', r'FACTURAR\s+A:', r'BILL\s+TO:'
]

COMMON_GEO_WORDS = {
    "MADRID","GETAFE","VALLECAS","ATOCHA","ALCORCON","ALCORCÓN","MOSTOLES","MÓSTOLES",
    "BARCELONA","SEVILLA","VALENCIA","MALAGA","MÁLAGA","ZARAGOZA","BILBAO","TOLEDO",
    "ALICANTE","CORDOBA","CÓRDOBA","VALLADOLID","GIJON","GIJÓN","LEGANES","LEGANÉS",
    "PARLA","FUENLABRADA","RIVAS","MAJADAHONDA","ALCOBENDAS","SANSE","SAN SEBASTIAN",
    "TARRAGONA","CASTELLON","CASTELLÓN","ESPAÑA","SPAIN"
}

# Patrones conocidos de proveedores (detección inicial)
KNOWN_SUPPLIERS = [
    r'MAREGALEVILLA', r'MARE\s*GALE\s*VILLA', r'WWW\.MAREGALEVILLA\.\w{2,}',
    r'SUPRACAFE', r'SUPRACAFÉ', r'SUPRA\s*CAFE', r'WWW\.SUPRACAFE\.\w{2,}',
    r'EHOSA', r'ELESPEJO\s+HOSTELEROS?\s*S\.?A\.?',
    r'CONGELADOS?\s*EL\s*GORDO', r'CONGELADOS?ELGORDO',
    r'MERCADONA\s*S\.?A\.?', r'MAKRO', r'CARREFOUR', r'ALCAMPO', r'DIA\s*S\.?A\.?',
    r'LIDL', r'ALDI', r'EROSKI', r'HIPERCOR', r'CORTE\s*INGLES', r'METRO\s*CASH',
    r'ALIMENTACION\s+[A-ZÁÉÍÓÚÑ]+', r'DISTRIBUCIONES\s+[A-ZÁÉÍÓÚÑ]+',
    r'MAYORISTAS?\s+[A-ZÁÉÍÓÚÑ]+', r'SUMINISTROS?\s+[A-ZÁÉÍÓÚÑ]+',
]
# Lista canónica para fuzzy
KNOWN_SUPPLIERS_CANON = [
    "SUPRACAFE", "MAREGALEVILLA", "EHOSA", "MERCADONA S.A.",
    "COCA-COLA EUROPACIFIC PARTNERS", "HOTMART BV", "OUIGO ESPAÑA S.A.U."
]
SHORT_VALID_SUPPLIERS = {"DIA"}  # permitir siglas reales

# =========================
# Utilidades de normalización / validación
# =========================
def _norm(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.upper()

def is_geo_word(s: str) -> bool:
    u = _norm(s).strip()
    return (u in COMMON_GEO_WORDS) or \
           (re.fullmatch(r'[A-ZÁÉÍÓÚÑ]{3,12}', u) and u not in SHORT_VALID_SUPPLIERS and u not in STOPWORDS_SUPPLIER)

def looks_like_geo_phrase(s: str) -> bool:
    u = _norm(s or "")
    return (" ESPAÑA" in u) or (" SPAIN" in u) or (any(g in u for g in COMMON_GEO_WORDS) and len(u) <= 20)

def is_probably_supplier_line(s: str) -> bool:
    s_up = _norm(s).strip()
    if not s_up:
        return False
    if any(x in s_up for x in STOPWORDS_SUPPLIER):
        return False
    if is_geo_word(s_up) or looks_like_geo_phrase(s_up):
        return False
    if len(s_up) <= 3 and s_up not in SHORT_VALID_SUPPLIERS:
        return False
    if len(s_up) > 60:
        return False
    if re.search(r'\b(EJECUTAR|DERECHO|DEVOL|CONDICIONES|POLITICA|POLÍTICA|AVISO|LEGAL|PEDIDOS|ENVIO|ENVÍO|FORMA|PAGO)\b', s_up):
        return False
    letters = [ch for ch in s_up if ch.isalpha()]
    caps_ratio = (sum(ch.isupper() for ch in letters) / max(1, len(letters))) if letters else 0
    if (" S.L" in s_up or " S.A" in s_up or re.search(r'\bS\.?L\.?\b|\bS\.?A\.?\b', s_up)) or caps_ratio > 0.6:
        return True
    return 4 <= len(s_up) <= 24 and re.match(r'^[A-ZÁÉÍÓÚÑ0-9&\-\s\.]+$', s_up) is not None

def looks_like_corporate_name(s: str) -> bool:
    u = _norm(s or "")
    return bool(re.search(r'\b(S\.?L\.?U?|S\.?A\.?U?|S\.?L|S\.?A|BV|GMBH|SAS|LTD|LIMITADA|AN(Ó|O)NIMA)\b', u))

def normalize_supplier_from_text(full_text: str, current_supplier: str) -> str:
    """Prioriza MAREGALEVILLA si aparece; mantiene SUPRACAFE/EHOSA; evita sobreescribir otros fuertes."""
    t = _norm(full_text)
    cur = _norm(current_supplier)

    # 1) MAREGALEVILLA
    mare_in_doc = (re.search(r'\bMAREGALEVILLA\b', t) is not None) or ("WWW.MAREGALEVILLA." in t)
    if mare_in_doc:
        if (not cur) or ("NO ENCONTRADO" in cur) or (len(cur) <= 3) or (cur in STOPWORDS_SUPPLIER) or ("CONGELADOS" in cur and "GORDO" in cur):
            return "MAREGALEVILLA"

    # 2) SUPRACAFE
    if re.search(r'\bSUPRACAFE\b', t) or re.search(r'\bSUPRA\s*CAFE\b', t) or "WWW.SUPRACAFE." in t:
        if (not cur) or ("NO ENCONTRADO" in cur) or (len(cur) <= 3) or (cur in STOPWORDS_SUPPLIER):
            return "SUPRACAFE"
        if not any(name in cur for name in ["MERCADONA","CARREFOUR","ALCAMPO","EROSKI","EHOSA","MAKRO","MAREGALEVILLA"]):
            return "SUPRACAFE"

    # 3) EHOSA
    if "EHOSA" in t and ((not cur) or ("NO ENCONTRADO" in cur) or (len(cur) <= 3) or (cur in STOPWORDS_SUPPLIER)):
        return "EHOSA"

    return current_supplier

def normalize_supplier_label(label: str, full_text: str) -> str:
    """Corrige compactados y aplica preferencia contextual (MAREGALEVILLA sobre 'Congelados El Gordo')."""
    u = _norm(label)
    t = _norm(full_text)

    if re.search(r'\bMAREGALEVILLA\b', t) or "WWW.MAREGALEVILLA." in t:
        return "MAREGALEVILLA"

    mapping = [
        (r'^CONGELADOSELGORDO$', 'CONGELADOS EL GORDO'),
        (r'^COCA COLA EUROPACIFIC PARTN(ERS|ER)?S?$', 'COCA-COLA EUROPACIFIC PARTNERS'),
        (r'^SUPRACAFE$', 'SUPRACAFE'),
        (r'^EHOSA$', 'EHOSA'),
    ]
    for pat, rep in mapping:
        if re.match(pat, u):
            return rep
    return label

# =========================
# OCR robusto (OpenCV + multi-PSM + confianza)
# =========================
def preprocess_cv2(pil_img):
    img = np.array(pil_img.convert("RGB"))[:, :, ::-1]  # PIL->BGR
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Deskew (aprox.)
    coords = np.column_stack(np.where(gray < 255))
    angle = 0
    if coords.size:
        angle = cv2.minAreaRect(coords)[-1]
        angle = -(90 + angle) if angle < -45 else -angle
    (h, w) = gray.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    gray = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    # Binarización + limpieza
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 35, 15)
    bw = cv2.medianBlur(bw, 3)
    return bw

def ocr_image_conf(pil_img) -> str:
    if not TES_AVAILABLE:
        raise TesseractNotFoundError("Tesseract no está instalado en el servidor.")
    bw = preprocess_cv2(pil_img)
    best_text, best_score = "", -1
    for psm in [6, 4, 11, 12, 13]:
        cfg = f"--oem 1 --psm {psm}"
        data = pytesseract.image_to_data(bw, lang="spa+eng", config=cfg, output_type=pytesseract.Output.DICT)
        # Reconstrucción por líneas con confianza >=70
        lines = {}
        for i, conf in enumerate(data["conf"]):
            try:
                c = int(conf)
            except:
                c = -1
            if c < 70:
                continue
            ln = (data["page_num"][i], data["block_num"][i], data["par_num"][i], data["line_num"][i])
            token = (data["text"][i] or "").strip()
            if token:
                lines.setdefault(ln, []).append(token)
        text = "\n".join(" ".join(t for t in toks if t.strip()) for toks in lines.values())
        score = len(re.findall(r'(?:factura|invoice|n[º°o]|num|serie)', text, re.I)) * 5 + len(text)
        if score > best_score:
            best_text, best_score = text, score
    return best_text or ""

# =========================
# Extracción de texto (PDF / Imagen)
# =========================
def text_from_pdf(path: pathlib.Path) -> str:
    doc = fitz.open(path)
    partes = []
    for page in doc:
        emb = page.get_text("text").strip()
        if len(emb) > 50:
            partes.append(emb)
        else:
            # Sin texto nativo → OCR página
            if TES_AVAILABLE:
                pix = page.get_pixmap(dpi=400)
                with Image.open(io.BytesIO(pix.tobytes())) as im:
                    partes.append(ocr_image_conf(im))
            else:
                partes.append(emb)
    doc.close()
    return "\n".join(partes)

def limpiar_texto(texto: str) -> str:
    lineas = texto.split("\n")
    lineas_limpias = []
    for linea in lineas:
        if len(linea.strip()) < 3:
            continue
        if re.search(r'^[A-Z]-?\d{8}[A-Z]?$', linea.strip(), re.IGNORECASE):
            continue
        linea_limpia = re.sub(r'[^\w\s\-.,:/()áéíóúÁÉÍÓÚñÑ&]', ' ', linea)
        linea_limpia = re.sub(r'\s+', ' ', linea_limpia).strip()
        if linea_limpia:
            lineas_limpias.append(linea_limpia)
    return "\n".join(lineas_limpias)

# =========================
# Nº de factura: score y validadores
# =========================
def score_invoice(s: str) -> int:
    if not s or s == "No encontrado":
        return 0
    sc = 0
    s2 = s.strip()
    # Añadido E\d{6,} como patrón fuerte
    if re.search(r'\b(FV|INV|AV|A-V|DGFC|BVRES|C\d{2,}|E\d{6,})', s2, re.I):
        sc += 4
    if re.search(r'[A-Z]', s2):
        sc += 2
    if '-' in s2 or '/' in s2:
        sc += 2
    if len(s2) >= 6:
        sc += 1
    if re.fullmatch(r'\d{1,5}', s2):
        sc -= 2
    return sc

def looks_like_nif(s: str) -> bool:
    return re.fullmatch(r'\d{8}[A-Z]', s or "") is not None

def looks_like_cif(s: str) -> bool:
    return re.fullmatch(r'[ABCDEFGHJNPQRSUVW]\d{7}[0-9A-J]', s or "", re.I) is not None

def looks_like_phone(s: str) -> bool:
    return re.search(r'(\+34)?\s?\d{3}[\s-]?\d{2,3}[\s-]?\d{2,3}', s or "") is not None

def looks_like_date(s: str) -> bool:
    s = (s or "").strip()
    return bool(re.fullmatch(r'\d{2}[/-]\d{2}[/-]\d{4}', s) or
                re.fullmatch(r'\d{4}[/-]\d{2}[/-]\d{2}', s))

def invalid_invoice_like(s: str) -> bool:
    return looks_like_nif(s) or looks_like_cif(s) or looks_like_phone(s) or looks_like_date(s)

# =========================
# EHOSA: contextualización del nº de factura
# =========================
EHOSA_BAD_PAT = re.compile(r'^(?:C|CL|CT)-?\d{2,6}$', re.I)
EHOSA_GOOD_NUM = re.compile(r'\b\d{7,10}\b')

def extract_invoice_contextual(texto: str) -> str:
    """Busca nº de factura solo cuando aparece con contexto (FACTURA/Nº/INVOICE)."""
    lines = [l.strip() for l in texto.splitlines() if l.strip()]
    pat_inline = re.compile(
        r'(?:^|\s)(?:FACTURA(?:\s+N[º°o\.])?|N[º°o]\s*FACTURA|N[º°o]|'
        r'NUM(?:ERO)?\s*(?:DE\s*)?FACTURA|INVOICE(?:\s+NO|\s*#)?|FACT\s*:|FAC\s*:)'
        r'\s*[:#-]?\s*(\d{7,10}|[A-Z0-9][A-Z0-9\-/\.]{4,})',
        re.IGNORECASE
    )
    pat_next = re.compile(r'(FACTURA|INVOICE|N[º°o]\s*FACTURA|NUM(?:ERO)?\s*FACTURA)\s*[:#-]?\s*$', re.IGNORECASE)

    for i, l in enumerate(lines[:200]):
        m = pat_inline.search(l)
        if m:
            val = m.group(1).strip()
            if not invalid_invoice_like(val) and score_invoice(val) >= 3:
                return val
        if pat_next.search(l) and i + 1 < len(lines):
            vals = re.findall(r'(\d{7,10}|[A-Z0-9][A-Z0-9\-/\.]{4,})', lines[i + 1])
            if vals:
                cand = vals[0]
                if not invalid_invoice_like(cand) and score_invoice(cand) >= 3:
                    return cand
    return "No encontrado"

def fix_invoice_for_supplier(nro: str, texto: str, supplier: str) -> str:
    """Si proveedor es EHOSA, rechaza C-#### y usa nº contextual."""
    sup = _norm(supplier)
    if "EHOSA" in sup:
        if EHOSA_BAD_PAT.match(nro or "") or re.fullmatch(r'\d{1,5}', nro or "") or invalid_invoice_like(nro or "") or not EHOSA_GOOD_NUM.search(nro or ""):
            ctx = extract_invoice_contextual(texto)
            if ctx != "No encontrado":
                return ctx
    return nro

# =========================
# Regex: factura / proveedor
# =========================
def extract_with_regex(texto: str) -> dict:
    resultado = {"nro_factura": "No encontrado", "proveedor": "No encontrado"}
    st.write("🔍 Debug - Primeras líneas OCR:", texto[:500])

    # Patrones fuertes primero
    patrones_factura = [
        r'([A-Z]{1,3}-[A-Z0-9]{1,4}-\d{5,15})',   # A-V2025-00002609357
        r'(FV[-/]?\d{1,2}[-/]?\d{5,10})',          # FV-0-2515226
        r'([A-Z]\d{9,18})',                        # C2025000851658 / e20252529117
        # Patrones previos
        r'(?:factura|invoice|fact|fac|nº|n°|no|num|number)[\s:.-]*([A-Z0-9\-/\.]{3,25})',
        r'([A-Z]{1,2}[-/]?\d{4,12})',
        r'(\d{6,15})',
        r'(\d{4,12}[-/][A-Z0-9]{1,8})',
        r'([A-Z]\d{8,15})',
        r'(e\d{8,15})',
        r'F[-/]?(\d{4,12})',
        r'INV[-/]?(\d{4,12})',
        r'([A-Z]{2,5}[-/]?\d{3,12})',
        r'\b\d{7,8}\b'
    ]
    for i, patron in enumerate(patrones_factura):
        matches = re.findall(patron, texto, re.IGNORECASE | re.MULTILINE)
        if matches:
            for match in matches:
                if (len(match) >= 6 and not match.isalpha() and
                    not re.match(r'^M\d{4,6}$', match) and
                    not re.match(r'^[A-Z]-?\d{8}[A-Z]?$', match, re.IGNORECASE) and
                    not invalid_invoice_like(match)):
                    resultado["nro_factura"] = match
                    st.write(f"✅ Factura encontrada con patrón {i+1}: {match}")
                    break
            if resultado["nro_factura"] != "No encontrado":
                break

    lineas = texto.split('\n')

    def has_contact_noise(s: str) -> bool:
        return re.search(r'(TEL|TLF|TELEF|M[ÓO]VIL|FAX|EMAIL|E-?MAIL|@|WHATSAPP|WEB|WWW|HTTP)', s, re.IGNORECASE) is not None

    # 1) Proveedores conocidos
    for linea in lineas[:60]:
        l = linea.strip()
        if has_contact_noise(l):
            continue
        if any(re.search(t, l, re.IGNORECASE) for t in CLIENT_HINTS):
            st.write(f"❌ Descartado como cliente: {l}")
            continue
        for pat in KNOWN_SUPPLIERS:
            if re.search(pat, l, re.IGNORECASE) and is_probably_supplier_line(l):
                resultado["proveedor"] = l.strip()
                resultado["proveedor"] = normalize_supplier_from_text(texto, resultado["proveedor"])
                resultado["proveedor"] = normalize_supplier_label(resultado["proveedor"], texto)
                if is_geo_word(resultado["proveedor"]) or looks_like_geo_phrase(resultado["proveedor"]):
                    resultado["proveedor"] = "No encontrado"
                if resultado["proveedor"] not in ("No encontrado", "", None):
                    # fuzzy: canónicos; si no, exige nombre corporativo o marca clara
                    cand = resultado["proveedor"]
                    best = process.extractOne(_norm(cand), KNOWN_SUPPLIERS_CANON, scorer=fuzz.token_sort_ratio)
                    if not (best and best[1] >= 90) and not (looks_like_corporate_name(cand) or re.fullmatch(r'[A-Z0-9&\-\s]{4,24}', _norm(cand))):
                        resultado["proveedor"] = "No encontrado"
                resultado["nro_factura"] = fix_invoice_for_supplier(resultado["nro_factura"], texto, resultado["proveedor"])
                st.write(f"✅ Proveedor conocido: {resultado['proveedor']}")
                return resultado

    # 2) Dominio web
    for linea in lineas:
        if has_contact_noise(linea):
            continue
        m = re.search(r'www\.([a-z0-9\-]+)\.\w{2,}', linea.lower())
        if m:
            prov = m.group(1).replace('-', ' ').upper()
            if prov and prov not in STOPWORDS_SUPPLIER and is_probably_supplier_line(prov):
                resultado["proveedor"] = prov
                resultado["proveedor"] = normalize_supplier_from_text(texto, resultado["proveedor"])
                resultado["proveedor"] = normalize_supplier_label(resultado["proveedor"], texto)
                if is_geo_word(resultado["proveedor"]) or looks_like_geo_phrase(resultado["proveedor"]):
                    resultado["proveedor"] = "No encontrado"
                if resultado["proveedor"] not in ("No encontrado", "", None):
                    cand = resultado["proveedor"]
                    best = process.extractOne(_norm(cand), KNOWN_SUPPLIERS_CANON, scorer=fuzz.token_sort_ratio)
                    if not (best and best[1] >= 90) and not (looks_like_corporate_name(cand) or re.fullmatch(r'[A-Z0-9&\-\s]{4,24}', _norm(cand))):
                        resultado["proveedor"] = "No encontrado"
                resultado["nro_factura"] = fix_invoice_for_supplier(resultado["nro_factura"], texto, resultado["proveedor"])
                st.write(f"✅ Proveedor por dominio: {resultado['proveedor']}")
                return resultado

    # 3) Patrón general de empresa
    patrones_empresa = [
        r'([A-ZÁÉÍÓÚÑ][A-Za-zÁÉÍÓÚáéíóúñ\s.&\-\']+(?:S\.?[AL]\.?|SOCIEDAD|LIMITADA|AN(Ó|O)NIMA))',
        r'([A-ZÁÉÍÓÚÑ]{3,}(?:\s+[A-ZÁÉÍÓÚÑ]{3,})*(?:\s+S\.?[AL]\.?){0,1})',
        r'([A-ZÁÉÍÓÚÑ]+\s+\d{4}\s+S\.?L\.?)',
        r'([A-ZÁÉÍÓÚÑ][A-Za-zÁÉÍÓÚáéíóúñ]{2,}(?:\s+[A-ZÁÉÍÓÚÑ][A-Za-zÁÉÍÓÚáéíóúñ]{2,}){1,3})',
    ]
    for linea in lineas[:25]:
        l = linea.strip()
        if len(l) < 5 or has_contact_noise(l):
            continue
        if re.search(r'^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|^[A-Z]-?\d{8}|^\d+\s*€|^C/|^CALLE|^AVDA|FECHA|DATE|^TEL|^TLF|^FAX|^EMAIL', l, re.IGNORECASE):
            continue
        for pat in patrones_empresa:
            m = re.search(pat, l)
            if m:
                cand = m.group(1).strip()
                cup = _norm(cand)
                if (cup not in STOPWORDS_SUPPLIER and is_probably_supplier_line(cand) and
                    not any(re.search(t, cand, re.IGNORECASE) for t in CLIENT_HINTS)):
                    resultado["proveedor"] = cand
                    resultado["proveedor"] = normalize_supplier_from_text(texto, resultado["proveedor"])
                    resultado["proveedor"] = normalize_supplier_label(resultado["proveedor"], texto)
                    if is_geo_word(resultado["proveedor"]) or looks_like_geo_phrase(resultado["proveedor"]):
                        resultado["proveedor"] = "No encontrado"
                    if resultado["proveedor"] not in ("No encontrado", "", None):
                        cand2 = resultado["proveedor"]
                        best = process.extractOne(_norm(cand2), KNOWN_SUPPLIERS_CANON, scorer=fuzz.token_sort_ratio)
                        if not (best and best[1] >= 90) and not (looks_like_corporate_name(cand2) or re.fullmatch(r'[A-Z0-9&\-\s]{4,24}', _norm(cand2))):
                            resultado["proveedor"] = "No encontrado"
                    resultado["nro_factura"] = fix_invoice_for_supplier(resultado["nro_factura"], texto, resultado["proveedor"])
                    st.write(f"✅ Proveedor (general): {resultado['proveedor']}")
                    return resultado

    # Final
    resultado["proveedor"] = normalize_supplier_from_text(texto, resultado["proveedor"])
    resultado["proveedor"] = normalize_supplier_label(resultado["proveedor"], texto)
    if is_geo_word(resultado["proveedor"]) or looks_like_geo_phrase(resultado["proveedor"]):
        resultado["proveedor"] = "No encontrado"
    if resultado["proveedor"] not in ("No encontrado", "", None):
        cand3 = resultado["proveedor"]
        best = process.extractOne(_norm(cand3), KNOWN_SUPPLIERS_CANON, scorer=fuzz.token_sort_ratio)
        if not (best and best[1] >= 90) and not (looks_like_corporate_name(cand3) or re.fullmatch(r'[A-Z0-9&\-\s]{4,24}', _norm(cand3))):
            resultado["proveedor"] = "No encontrado"

    resultado["nro_factura"] = fix_invoice_for_supplier(resultado["nro_factura"], texto, resultado["proveedor"])
    return resultado

# =========================
# Fallbacks por nombre de archivo
# =========================
def invoice_from_filename(fname: str) -> str:
    base = pathlib.Path(fname).stem
    cands = []
    for pat in [
        r'([A-Z]{1,3}-[A-Z0-9]{1,4}-\d{5,15})',
        r'(FV[-/]?\d{1,2}[-/]?\d{5,10})',
        r'([A-Z]\d{9,18})',
        r'([A-Z]{2,5}[-/]?\d{3,12})'
    ]:
        m = re.search(pat, base, re.IGNORECASE)
        if m:
            cands.append(m.group(1))
    if not cands:
        return "No encontrado"
    best = max(cands, key=lambda s: score_invoice(s))
    return best if not invalid_invoice_like(best) else "No encontrado"

def supplier_from_filename(fname: str) -> str:
    if not fname:
        return "No encontrado"
    base = _norm(pathlib.Path(fname).stem)
    toks = re.split(r'[\s._-]+', base)
    for size in range(3, 0, -1):  # tríos, pares, uno
        for i in range(len(toks)-size+1):
            cand = " ".join(toks[i:i+size])
            best = process.extractOne(cand, KNOWN_SUPPLIERS_CANON, scorer=fuzz.token_sort_ratio)
            if best and best[1] >= 90:
                return best[0]
    return "No encontrado"

# =========================
# OpenAI: refuerzo y normalización
# =========================
def extract_data_with_openai(invoice_text: str) -> dict:
    texto_limpio = limpiar_texto(invoice_text)
    regex_result = extract_with_regex(texto_limpio)

    prompt = f"""
Eres un experto en facturas españolas. Extrae:
1) invoiceNumber: código de la factura
2) supplier: empresa EMISORA (no el cliente)
No aceptes valores genéricos (FACTURA/INVOICE/ALBARÁN/TICKET/CLIENTE).
Devuelve SOLO JSON:
{{
  "invoiceNumber": "valor o No encontrado",
  "supplier": "valor o No encontrado"
}}
TEXTO:
{texto_limpio[:3000]}
    """
    try:
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Extrae datos de facturas. Devuelve JSON exacto."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=300
        )
        content = resp.choices[0].message.content.strip()
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(content)

        out = {
            "nro_factura": (parsed.get("invoiceNumber") or "No encontrado").strip(),
            "proveedor": (parsed.get("supplier") or "No encontrado").strip()
        }

        # Post-procesado proveedor
        prov = _norm(out["proveedor"])
        if prov in STOPWORDS_SUPPLIER or is_geo_word(prov) or looks_like_geo_phrase(prov):
            out["proveedor"] = "No encontrado"
        if any(t in prov for t in ["MISU","RINCON","TORTILLA","RESTAURANTE","BAR","CAFETERIA","TABERNA","ASADOR","PIZZERIA","HOTEL","HOSTAL"]):
            out["proveedor"] = "No encontrado"
        if "MERCADONA" in prov:
            out["proveedor"] = "MERCADONA S.A."
        elif "SUPRACAFE" in prov:
            out["proveedor"] = "SUPRACAFE"
        elif "EHOSA" in prov:
            out["proveedor"] = "EHOSA"

        # Elegir mejor nº de factura entre LLM y regex
        r_alt = regex_result.get("nro_factura", "No encontrado")
        if out["nro_factura"] == "No encontrado" and r_alt != "No encontrado":
            out["nro_factura"] = r_alt
            st.write(f"🔄 Usando regex para factura: {out['nro_factura']}")
        else:
            best = out["nro_factura"]
            if score_invoice(r_alt) > score_invoice(best):
                out["nro_factura"] = r_alt
            if re.fullmatch(r'\d{1,5}', (best or "")) and r_alt and r_alt != "No encontrado":
                out["nro_factura"] = r_alt
        # Evitar NIF/CIF/teléfono/fecha como factura
        best_now = out["nro_factura"]
        alt = regex_result.get("nro_factura", "No encontrado")
        if invalid_invoice_like(best_now) and alt and alt != "No encontrado" and not invalid_invoice_like(alt):
            out["nro_factura"] = alt

        # Backoff proveedor desde regex si LLM no lo encontró
        if out["proveedor"] == "No encontrado" and regex_result["proveedor"] != "No encontrado":
            out["proveedor"] = regex_result["proveedor"]
            st.write(f"🔄 Usando regex para proveedor: {out['proveedor']}")

        # Normalizaciones fuertes + filtros
        out["proveedor"] = normalize_supplier_from_text(invoice_text, out["proveedor"])
        out["proveedor"] = normalize_supplier_label(out["proveedor"], invoice_text)
        if is_geo_word(out["proveedor"]) or looks_like_geo_phrase(out["proveedor"]):
            out["proveedor"] = "No encontrado"
        # Fuzzy/aceptación corporativa o marca clara
        if out["proveedor"] not in ("No encontrado", "", None):
            cand = out["proveedor"]
            best = process.extractOne(_norm(cand), KNOWN_SUPPLIERS_CANON, scorer=fuzz.token_sort_ratio)
            if not (best and best[1] >= 90) and not (looks_like_corporate_name(cand) or re.fullmatch(r'[A-Z0-9&\-\s]{4,24}', _norm(cand))):
                out["proveedor"] = "No encontrado"

        # EHOSA: fijar nº por contexto si hace falta
        out["nro_factura"] = fix_invoice_for_supplier(out["nro_factura"], invoice_text, out["proveedor"])

        # Fallback por nombre de archivo (proveedor)
        fname = st.session_state.get("current_filename", "")
        if out["proveedor"] == "No encontrado" and fname:
            altp = supplier_from_filename(fname)
            if altp != "No encontrado":
                out["proveedor"] = altp

        st.write(f"✅ Final - Factura: {out['nro_factura']} | Proveedor: {out['proveedor']}")
        return out

    except json.JSONDecodeError as e:
        st.warning(f"Error JSON OpenAI: {e}")
        st.write(f"Contenido recibido: {content}")
        regex_result["proveedor"] = normalize_supplier_from_text(invoice_text, regex_result["proveedor"])
        regex_result["proveedor"] = normalize_supplier_label(regex_result["proveedor"], invoice_text)
        if is_geo_word(regex_result["proveedor"]) or looks_like_geo_phrase(regex_result["proveedor"]):
            regex_result["proveedor"] = "No encontrado"
        if regex_result["proveedor"] not in ("No encontrado", "", None):
            cand = regex_result["proveedor"]
            best = process.extractOne(_norm(cand), KNOWN_SUPPLIERS_CANON, scorer=fuzz.token_sort_ratio)
            if not (best and best[1] >= 90) and not (looks_like_corporate_name(cand) or re.fullmatch(r'[A-Z0-9&\-\s]{4,24}', _norm(cand))):
                regex_result["proveedor"] = "No encontrado"
        # EHOSA contextual
        regex_result["nro_factura"] = fix_invoice_for_supplier(regex_result["nro_factura"], invoice_text, regex_result["proveedor"])

        # Fallback proveedor por nombre de archivo
        fname = st.session_state.get("current_filename", "")
        if regex_result["proveedor"] == "No encontrado" and fname:
            altp = supplier_from_filename(fname)
            if altp != "No encontrado":
                regex_result["proveedor"] = altp

        return regex_result

    except Exception as e:
        st.error(f"Error OpenAI: {e}")
        regex_result["proveedor"] = normalize_supplier_from_text(invoice_text, regex_result["proveedor"])
        regex_result["proveedor"] = normalize_supplier_label(regex_result["proveedor"], invoice_text)
        if is_geo_word(regex_result["proveedor"]) or looks_like_geo_phrase(regex_result["proveedor"]):
            regex_result["proveedor"] = "No encontrado"
        if regex_result["proveedor"] not in ("No encontrado", "", None):
            cand = regex_result["proveedor"]
            best = process.extractOne(_norm(cand), KNOWN_SUPPLIERS_CANON, scorer=fuzz.token_sort_ratio)
            if not (best and best[1] >= 90) and not (looks_like_corporate_name(cand) or re.fullmatch(r'[A-Z0-9&\-\s]{4,24}', _norm(cand))):
                regex_result["proveedor"] = "No encontrado"
        # EHOSA contextual
        regex_result["nro_factura"] = fix_invoice_for_supplier(regex_result["nro_factura"], invoice_text, regex_result["proveedor"])

        # Fallback proveedor por nombre de archivo
        fname = st.session_state.get("current_filename", "")
        if regex_result["proveedor"] == "No encontrado" and fname:
            altp = supplier_from_filename(fname)
            if altp != "No encontrado":
                regex_result["proveedor"] = altp

        return regex_result

# =========================
# Procesamiento de archivo (key único -> evita DuplicateWidgetID)
# =========================
def process_file(file, widget_key: str = "") -> dict:
    st.session_state["current_filename"] = file.name  # → para fallbacks por nombre
    with tempfile.NamedTemporaryFile(delete=False, suffix=file.name) as tmp:
        tmp.write(file.read())
        tmp_path = pathlib.Path(tmp.name)
        tmp_key = tmp_path.name  # fallback para key único

    try:
        if tmp_path.suffix.lower() == ".pdf":
            texto = text_from_pdf(tmp_path)
        else:
            if not TES_AVAILABLE:
                return {"archivo": file.name, "nro_factura": "Error: Tesseract no está instalado", "proveedor": "Error: Tesseract no está instalado"}
            imagen = Image.open(tmp_path)
            # (rotación EXIF si existe)
            if hasattr(imagen, '_getexif'):
                exif = imagen._getexif()
                if exif is not None:
                    orientation = exif.get(274)
                    if orientation == 3:
                        imagen = imagen.rotate(180, expand=True)
                    elif orientation == 6:
                        imagen = imagen.rotate(270, expand=True)
                    elif orientation == 8:
                        imagen = imagen.rotate(90, expand=True)
            texto = ocr_image_conf(imagen)
    except Exception as e:
        return {"archivo": file.name, "nro_factura": f"Error: {e}", "proveedor": f"Error: {e}"}
    finally:
        try:
            tmp_path.unlink()
        except Exception:
            pass

    if not texto or not texto.strip():
        return {"archivo": file.name, "nro_factura": "Error: No se pudo extraer texto", "proveedor": "Error: No se pudo extraer texto"}

    with st.expander(f"🔍 Debug - Texto extraído de {file.name}"):
        st.text_area(
            "Texto OCR:",
            texto[:1500],
            height=220,
            key=f"ta_ocr_{widget_key or tmp_key}"
        )

    result = extract_data_with_openai(texto)

    # Fallback Nº de factura desde el nombre de archivo (preferir si es mejor)
    alt_inv = invoice_from_filename(file.name)
    cur = result.get("nro_factura") or "No encontrado"
    sup = _norm(result.get("proveedor", ""))
    if alt_inv != "No encontrado":
        # Si es EHOSA y el filename trae C-####, NO lo uses
        if not ("EHOSA" in sup and EHOSA_BAD_PAT.match(alt_inv or "")):
            if invalid_invoice_like(cur) or score_invoice(alt_inv) >= max(1, score_invoice(cur)):
                result["nro_factura"] = alt_inv

    return {"archivo": file.name, **result}

# =========================
# UI
# =========================
st.title("📄 Lector de Facturas - OCR + OpenAI (Cloud)")
st.markdown("Sube PDF o imágenes. Extrae Nº de Factura y Proveedor (emisor).")

if st.sidebar.button("🧪 Test OpenAI"):
    try:
        _ = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Responde solo: OK"}],
            max_tokens=10
        )
        st.sidebar.success("✅ Conexión OpenAI OK")
    except Exception as e:
        st.sidebar.error(f"❌ Error de conexión: {e}")

with st.expander("ℹ️ Consejos"):
    st.markdown("""
- **PDF nativo** da mejor resultado que imagen rasterizada.
- Si subes **imágenes**, Tesseract debe estar instalado (ver *packages.txt*).
- El sistema identifica el **PROVEEDOR (emisor)**, no el cliente.
""")

if st.button("🗑️ Limpiar archivos cargados"):
    st.session_state.clear()
    st.rerun()

files = st.file_uploader("Selecciona archivos", type=["pdf","png","jpg","jpeg"], accept_multiple_files=True)

if files:
    progress_bar = st.progress(0)
    status_text = st.empty()
    resultados = []
    for i, file in enumerate(files):
        status_text.text(f"Procesando {file.name}... ({i+1}/{len(files)})")
        # ✅ key único por archivo para evitar DuplicateWidgetID
        resultado = process_file(file, widget_key=f"{i}_{file.name}")
        resultados.append(resultado)
        progress_bar.progress((i + 1) / len(files))

    status_text.text("¡Procesamiento completado!")
    df = pd.DataFrame(resultados)

    st.subheader("📋 Resultados")

    # Métricas SIN f-strings partidos (evita SyntaxError)
    prev_fact = st.session_state.get("prev_fact_ok", 0)
    prev_prov = st.session_state.get("prev_prov_ok", 0)
    prev_files = st.session_state.get("prev_files", 0)

    col1, col2, col3 = st.columns(3)
    with col1:
        facturas_ok = sum(1 for r in resultados if "No encontrado" not in r["nro_factura"] and "Error" not in r["nro_factura"])
        delta_fact = facturas_ok - prev_fact
        st.metric("Facturas detectadas", facturas_ok, delta=f"{delta_fact:+d}")
    with col2:
        proveedores_ok = sum(1 for r in resultados if "No encontrado" not in r["proveedor"] and "Error" not in r["proveedor"])
        delta_prov = proveedores_ok - prev_prov
        st.metric("Proveedores detectados", proveedores_ok, delta=f"{delta_prov:+d}")
    with col3:
        total_arch = len(files)
        delta_files = total_arch - prev_files
        st.metric("Total archivos", total_arch, delta=f"{delta_files:+d}")

    st.session_state["prev_fact_ok"] = facturas_ok
    st.session_state["prev_prov_ok"] = proveedores_ok
    st.session_state["prev_files"] = total_arch

    def highlight_results(row):
        colors = []
        for col in row.index:
            val = str(row[col])
            if col == "archivo":
                colors.append("")
            elif "Error" in val:
                colors.append("background-color: #f44336; color: white")
            elif "No encontrado" in val:
                colors.append("background-color: #ffeb3b")
            else:
                colors.append("background-color: #4caf50; color: white")
        return colors

    st.dataframe(
        df.style.apply(highlight_results, axis=1),
        use_container_width=True,
        column_config={
            "archivo": st.column_config.TextColumn("Archivo", width="medium"),
            "nro_factura": st.column_config.TextColumn("Número Factura", width="medium"),
            "proveedor": st.column_config.TextColumn("Proveedor", width="large")
        }
    )

    with st.expander("📊 Detalles"):
        for r in resultados:
            st.write(f"**{r['archivo']}**")
            c1, c2 = st.columns(2)
            with c1:
                st.write(f"• Factura: `{r['nro_factura']}`")
            with c2:
                st.write(f"• Proveedor: `{r['proveedor']}`")
            st.divider()

    st.subheader("📤 Descargar")

    # -------- PDF (fpdf2) --------
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("helvetica", "B", 16)
    pdf.cell(0, 10, "Resumen de Facturas Procesadas", ln=True, align="C")
    pdf.ln(10)

    pdf.set_font("helvetica", "B", 10)
    pdf.set_fill_color(200, 200, 200)
    pdf.cell(70, 10, "Archivo", 1, 0, 'C', 1)
    pdf.cell(60, 10, "Numero Factura", 1, 0, 'C', 1)
    pdf.cell(60, 10, "Proveedor", 1, 1, 'C', 1)

    pdf.set_font("helvetica", size=9)
    for r in resultados:
        archivo = r["archivo"][:32] + "..." if len(r["archivo"]) > 35 else r["archivo"]
        nro_factura = str(r["nro_factura"])
        if len(nro_factura) > 25:
            nro_factura = nro_factura[:22] + "..."
        proveedor = str(r["proveedor"])
        if len(proveedor) > 30:
            proveedor = proveedor[:27] + "..."
        pdf.cell(70, 8, archivo, 1, 0, 'L')
        pdf.cell(60, 8, nro_factura, 1, 0, 'L')
        pdf.cell(60, 8, proveedor, 1, 1, 'L')

    out = pdf.output(dest="S")
    pdf_bytes = bytes(out) if isinstance(out, (bytearray, bytes)) else out.encode("latin-1")
    st.download_button("📥 Descargar PDF", data=pdf_bytes, file_name="resumen_facturas.pdf", mime="application/pdf")

    # Excel
    excel_buf = io.BytesIO()
    df.to_excel(excel_buf, index=False, engine="openpyxl")
    excel_buf.seek(0)
    st.download_button(
        "📥 Descargar Excel",
        data=excel_buf.getvalue(),
        file_name="facturas_procesadas.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )









