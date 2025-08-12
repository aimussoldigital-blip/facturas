# --- Lector de Facturas con OCR + OpenAI (Cloud, FIX set_page_config) ---
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
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
from fpdf import FPDF
from openai import OpenAI

# ✅ Debe ser la PRIMERA llamada de Streamlit
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
    
    for key_name in ["openai_api_key", "OPENAI_API_KEY", "openai-api-key", "openai_key", "OPENAI_KEY", "api_key"]:
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
# OCR / Texto
# =========================
def preprocess(img: Image.Image) -> Image.Image:
    if min(img.size) < 800:
        scale = 800 / min(img.size)
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    img = ImageOps.grayscale(img)
    img = ImageEnhance.Contrast(img).enhance(2.0)
    img = ImageEnhance.Sharpness(img).enhance(1.5)
    img = img.filter(ImageFilter.MedianFilter(size=3))
    return img.point(lambda x: 0 if x < 140 else 255, "1")

def ocr_image(img: Image.Image) -> str:
    if not TES_AVAILABLE:
        raise TesseractNotFoundError("Tesseract no está instalado en el servidor.")
    img_proc = preprocess(img)
    config = "--psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÁÉÍÓÚáéíóúÑñ.,:-/() "
    return pytesseract.image_to_string(img_proc, lang="spa+eng", config=config)

def text_from_pdf(path: pathlib.Path) -> str:
    doc = fitz.open(path)
    partes = []
    for page in doc:
        emb = page.get_text("text").strip()
        if len(emb) > 50:
            partes.append(emb)
        else:
            if TES_AVAILABLE:
                pix = page.get_pixmap(dpi=400)
                with Image.open(io.BytesIO(pix.tobytes())) as im:
                    partes.append(ocr_image(im))
            else:
                partes.append(emb)
    doc.close()
    return "\n".join(partes)

# =========================
# Limpieza & Heurísticas
# =========================
STOPWORDS_SUPPLIER = {
    "FACTURA", "FACT", "INVOICE", "ALBARAN", "ALBARÁN", "TICKET",
    "CLIENTE", "DESTINATARIO", "FACTURAR A", "BILL TO", "CUSTOMER",
    "FECHA", "DATE", "IVA", "CIF", "NIF", "NIE", "PAGINA", "PÁGINA"
}

CLIENT_HINTS = [
    r'MISU\s+\d+\s+S\.?L\.?', r'\bMISU\b',
    r'RINCON\s+DE\s+LA\s+TORTILLA', r'EL\s+RINCON\s+DE\s+LA\s+TORTILLA',
    r'RESTAURANTE\s+[A-ZÁÉÍÓÚÑ]+', r'BAR\s+[A-ZÁÉÍÓÚÑ]+', r'CAFETERIA\s+[A-ZÁÉÍÓÚÑ]+',
    r'TABERNA\s+[A-ZÁÉÍÓÚÑ]+', r'ASADOR\s+[A-ZÁÉÍÓÚÑ]+', r'PIZZERIA\s+[A-ZÁÉÍÓÚÑ]+',
    r'CLIENTE:', r'CUSTOMER:', r'DESTINATARIO:', r'FACTURAR\s+A:', r'BILL\s+TO:'
]

KNOWN_SUPPLIERS = [
    r'SUPRACAFE', r'SUPRACAFÉ', r'EHOSA', r'ELESPEJO\s+HOSTELEROS?\s*S\.?A\.?',
    r'MERCADONA\s*S\.?A\.?', r'MAKRO', r'CARREFOUR', r'ALCAMPO', r'DIA\s*S\.?A\.?',
    r'LIDL', r'ALDI', r'EROSKI', r'HIPERCOR', r'CORTE\s*INGLES', r'METRO\s*CASH',
    r'ALIMENTACION\s+[A-ZÁÉÍÓÚÑ]+', r'DISTRIBUCIONES\s+[A-ZÁÉÍÓÚÑ]+',
    r'MAYORISTAS?\s+[A-ZÁÉÍÓÚÑ]+', r'SUMINISTROS?\s+[A-ZÁÉÍÓÚÑ]+',
]

def is_probably_supplier_line(s: str) -> bool:
    s_up = s.upper().strip()
    if not s_up or any(x in s_up for x in STOPWORDS_SUPPLIER):
        return False
    if len(s_up) > 60:
        return False
    if re.search(r'\b(EJECUTAR|DERECHO|DEVOL|CONDICIONES|POLITICA|POLÍTICA|AVISO|LEGAL|PEDIDOS|ENVIO|ENVÍO|FORMA|PAGO)\b', s_up):
        return False
    caps_ratio = sum(ch.isupper() for ch in s_up if ch.isalpha()) / max(1, sum(ch.isalpha() for ch in s_up))
    if (" S.L" in s_up or " S.A" in s_up or " SL" in s_up or " SA" in s_up) or caps_ratio > 0.6:
        return True
    if 3 <= len(s_up) <= 20 and re.match(r'^[A-ZÁÉÍÓÚÑ0-9&\-\s\.]+$', s_up):
        return True
    return False

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
# Regex: factura / proveedor
# =========================
def extract_with_regex(texto: str) -> dict:
    resultado = {"nro_factura": "No encontrado", "proveedor": "No encontrado"}
    st.write("🔍 Debug - Primeras líneas OCR:", texto[:500])

    patrones_factura = [
        r'(?:factura|invoice|fact|fac|nº|n°|no|num|number)[\s:.-]*([A-Z0-9\-/\.]{3,25})',
        r'([A-Z]{1,2}[-/]?\d{4,12})',
        r'(\d{6,15})',
        r'(\d{4,12}[-/][A-Z0-9]{1,8})',
        r'([A-Z]\d{8,15})',
        r'(e\d{8,15})',
        r'(FV[-/]?\d+[-/]?\d+)',
        r'F[-/]?(\d{4,12})',
        r'INV[-/]?(\d{4,12})',
        r'([A-Z]{2,5}[-/]?\d{3,12})',
        r'\b\d{7,8}\b'
    ]
    for i, patron in enumerate(patrones_factura):
        matches = re.findall(patron, texto, re.IGNORECASE | re.MULTILINE)
        if matches:
            for match in matches:
                if (
                    len(match) >= 6
                    and not match.isalpha()
                    and not re.match(r'^M\d{4,6}$', match)
                    and not re.match(r'^[A-Z]-?\d{8}[A-Z]?$', match, re.IGNORECASE)
                ):
                    resultado["nro_factura"] = match
                    st.write(f"✅ Factura encontrada con patrón {i+1}: {match}")
                    break
            if resultado["nro_factura"] != "No encontrado":
                break

    lineas = texto.split('\n')

    # 1) Proveedores conocidos
    for linea in lineas[:60]:
        l = linea.strip()
        if any(re.search(t, l, re.IGNORECASE) for t in CLIENT_HINTS):
            st.write(f"❌ Descartado como cliente: {l}")
            continue
        for pat in KNOWN_SUPPLIERS:
            if re.search(pat, l, re.IGNORECASE) and is_probably_supplier_line(l):
                resultado["proveedor"] = l.strip()
                st.write(f"✅ Proveedor conocido: {resultado['proveedor']}")
                return resultado

    # 2) Dominio web
    for linea in lineas:
        m = re.search(r'www\.([a-z0-9\-]+)\.\w{2,}', linea.lower())
        if m:
            prov = m.group(1).replace('-', ' ').upper()
            if prov and prov not in STOPWORDS_SUPPLIER and is_probably_supplier_line(prov):
                resultado["proveedor"] = prov
                st.write(f"✅ Proveedor por dominio: {prov}")
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
        if len(l) < 5:
            continue
        if re.search(r'^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|^[A-Z]-?\d{8}|^\d+\s*€|^C/|^CALLE|^AVDA|FECHA|DATE|^TEL|^FAX|^EMAIL', l, re.IGNORECASE):
            continue
        for pat in patrones_empresa:
            m = re.search(pat, l)
            if m:
                cand = m.group(1).strip()
                cup = cand.upper()
                if cup not in STOPWORDS_SUPPLIER and is_probably_supplier_line(cand) and not any(re.search(t, cand, re.IGNORECASE) for t in CLIENT_HINTS):
                    resultado["proveedor"] = cand
                    st.write(f"✅ Proveedor (general): {cand}")
                    return resultado

    return resultado

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

        prov = out["proveedor"].upper()
        if prov in STOPWORDS_SUPPLIER:
            out["proveedor"] = "No encontrado"
        if any(t in prov for t in ["MISU", "RINCON", "TORTILLA", "RESTAURANTE", "BAR", "CAFETERIA", "TABERNA", "ASADOR", "PIZZERIA", "HOTEL", "HOSTAL"]):
            out["proveedor"] = "No encontrado"
        if "MERCADONA" in prov:
            out["proveedor"] = "MERCADONA S.A."
        elif "SUPRACAFE" in prov or "SUPRACAFÉ" in prov:
            out["proveedor"] = "SUPRACAFE"
        elif "EHOSA" in prov:
            out["proveedor"] = "EHOSA"

        if out["nro_factura"] == "No encontrado" and regex_result["nro_factura"] != "No encontrado":
            out["nro_factura"] = regex_result["nro_factura"]
            st.write(f"🔄 Usando regex para factura: {out['nro_factura']}")
        if out["proveedor"] == "No encontrado" and regex_result["proveedor"] != "No encontrado":
            out["proveedor"] = regex_result["proveedor"]
            st.write(f"🔄 Usando regex para proveedor: {out['proveedor']}")

        st.write(f"✅ Final - Factura: {out['nro_factura']} | Proveedor: {out['proveedor']}")
        return out

    except json.JSONDecodeError as e:
        st.warning(f"Error JSON OpenAI: {e}")
        st.write(f"Contenido recibido: {content}")
        return regex_result
    except Exception as e:
        st.error(f"Error OpenAI: {e}")
        return regex_result

# =========================
# Procesamiento de archivo
# =========================
def process_file(file) -> dict:
    with tempfile.NamedTemporaryFile(delete=False, suffix=file.name) as tmp:
        tmp.write(file.read())
        tmp_path = pathlib.Path(tmp.name)

    try:
        if tmp_path.suffix.lower() == ".pdf":
            texto = text_from_pdf(tmp_path)
        else:
            if not TES_AVAILABLE:
                return {"archivo": file.name, "nro_factura": "Error: Tesseract no está instalado", "proveedor": "Error: Tesseract no está instalado"}
            imagen = Image.open(tmp_path)
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
            texto = ocr_image(imagen)
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
        st.text_area("Texto OCR:", texto[:1500], height=220)

    result = extract_data_with_openai(texto)
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
- Si subes **imágenes**, Tesseract debe estar instalado (ver packages.txt).
- El sistema identifica el **PROVEEDOR (emisor)**, no el cliente.
""")

if st.button("🗑️ Limpiar archivos cargados"):
    st.session_state.clear()
    st.rerun()

files = st.file_uploader("Selecciona archivos", type=["pdf", "png", "jpg", "jpeg"], accept_multiple_files=True)

if files:
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    resultados = []
    for i, file in enumerate(files):
        status_text.text(f"Procesando {file.name}... ({i+1}/{len(files)})")
        resultado = process_file(file)
        resultados.append(resultado)
        progress_bar.progress((i + 1) / len(files))
    
    status_text.text("¡Procesamiento completado!")
    df = pd.DataFrame(resultados)

    st.subheader("📋 Resultados")
    col1, col2, col3 = st.columns(3)
    with col1:
        facturas_ok = sum(1 for r in resultados if "No encontrado" not in r["nro_factura"] and "Error" not in r["nro_factura"])
        st.metric("Facturas detectadas", facturas_ok)
    with col2:
        proveedores_ok = sum(1 for r in resultados if "No encontrado" not in r["proveedor"] and "Error" not in r["proveedor"])
        st.metric("Proveedores detectados", proveedores_ok)
    with col3:
        st.metric("Total archivos", len(files))
    
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

    # -------- PDF (fpdf2 correcto) --------
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
















