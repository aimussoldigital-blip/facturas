# --- Lector de Facturas con OCR + OpenAI (versión corregida) ---
import streamlit as st
import pandas as pd
import pytesseract
import fitz  # PyMuPDF
import io
import tempfile
import pathlib
import re
import json
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
from fpdf import FPDF
from openai import OpenAI

# --- Configuración de OpenAI con manejo de errores ---
def configurar_openai():
    """Configura OpenAI verificando múltiples posibles nombres de keys"""
    st.sidebar.write("🔍 Debug - Secrets:")
    try:
        available_keys = list(st.secrets.keys())
        st.sidebar.write(f"Keys encontradas: {available_keys}")
    except:
        st.sidebar.write("No se pudieron leer los secrets")
    
    possible_keys = [
        "openai_api_key",
        "OPENAI_API_KEY",
        "openai-api-key", 
        "openai_key",
        "OPENAI_KEY",
        "api_key"
    ]
    for key_name in possible_keys:
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
    st.info("""
    **Para configurar OpenAI en Streamlit Cloud:**
    1) Manage app → Settings → Secrets
    2) Añade exactamente:
       openai_api_key = "sk-tu-clave"
    3) Guarda y reinicia
    """)
    st.stop()

# --- Inicializar cliente OpenAI ---
client = configurar_openai()

# --- Preprocesamiento OCR ---
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
    img_proc = preprocess(img)
    config = "--psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÁÉÍÓÚáéíóúÑñ.,:-/() "
    return pytesseract.image_to_string(img_proc, lang="spa+eng", config=config)

def text_from_pdf(path: pathlib.Path) -> str:
    doc = fitz.open(path)
    texto = []
    for page in doc:
        emb = page.get_text("text").strip()
        if len(emb) > 50:
            texto.append(emb)
        else:
            pix = page.get_pixmap(dpi=400)
            with Image.open(io.BytesIO(pix.tobytes())) as im:
                texto.append(ocr_image(im))
    doc.close()
    return "\n".join(texto)

# --- Limpieza de texto ---
def limpiar_texto(texto: str) -> str:
    lineas = texto.split("\n")
    lineas_limpias = []
    for linea in lineas:
        if len(linea.strip()) < 3:
            continue
        if re.search(r'^[A-Z]-?\d{8}[A-Z]?$', linea.strip(), re.IGNORECASE):
            continue
        linea_limpia = re.sub(r'[^\w\s\-.,:/()áéíóúÁÉÍÓÚñÑ]', ' ', linea)
        linea_limpia = re.sub(r'\s+', ' ', linea_limpia).strip()
        if linea_limpia:
            lineas_limpias.append(linea_limpia)
    return "\n".join(lineas_limpias)

# --- Extracción con regex ---
def extract_with_regex(texto: str) -> dict:
    resultado = {"nro_factura": "No encontrado", "proveedor": "No encontrado"}
    st.write("🔍 Debug - Primeras líneas del texto OCR:", texto[:500])

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

    lineas = texto.split('\n')

    proveedores_conocidos = [
        r'SUPRACAFE', r'SUPRACAFÉ', r'ELESPEJO\s+HOSTELEROS?\s*S\.?A\.?',
        r'MERCADONA\s*S\.?A\.?', r'MAKRO', r'CARREFOUR', r'ALCAMPO', r'DIA\s*S\.?A\.?',
        r'LIDL', r'ALDI', r'EROSKI', r'HIPERCOR', r'CORTE\s*INGLES', r'METRO\s*CASH',
        r'ALIMENTACION\s+[A-Z]+', r'DISTRIBUCIONES\s+[A-Z]+', r'MAYORISTAS?\s+[A-Z]+',
        r'SUMINISTROS?\s+[A-Z]+',
    ]

    terminos_cliente = [
        r'MISU\s+\d+\s+S\.?L\.?', r'\bMISU\b', r'RINCON\s+DE\s+LA\s+TORTILLA',
        r'EL\s+RINCON\s+DE\s+LA\s+TORTILLA', r'RESTAURANTE\s+[A-Z]+', r'BAR\s+[A-Z]+',
        r'CAFETERIA\s+[A-Z]+', r'TABERNA\s+[A-Z]+', r'ASADOR\s+[A-Z]+', r'PIZZERIA\s+[A-Z]+',
        r'CLIENTE:', r'CUSTOMER:', r'DESTINATARIO:', r'FACTURAR\s+A:', r'BILL\s+TO:',
    ]

    for linea in lineas[:15]:
        linea_limpia = linea.strip()
        if len(linea_limpia) < 3:
            continue

        es_cliente = False
        for termino in terminos_cliente:
            if re.search(termino, linea_limpia, re.IGNORECASE):
                es_cliente = True
                st.write(f"❌ Descartado como cliente: {linea_limpia}")
                break
        if es_cliente:
            continue

        for proveedor_patron in proveedores_conocidos:
            if re.search(proveedor_patron, linea_limpia, re.IGNORECASE):
                resultado["proveedor"] = linea_limpia.strip()
                st.write(f"✅ Proveedor conocido encontrado: {linea_limpia}")
                return resultado

    for linea in lineas:
        match = re.search(r'www\.([a-z0-9\-]+)\.\w{2,}', linea.lower())
        if match:
            proveedor_web = match.group(1).replace('-', ' ').upper()
            resultado["proveedor"] = proveedor_web
            st.write(f"✅ Proveedor detectado por dominio web: {proveedor_web}")
            return resultado

    patrones_empresa = [
        r'([A-ZÁÉÍÓÚÑ][A-Za-záéíóúñ\s]+(?:S\.?[AL]\.?|SOCIEDAD|LIMITADA|ANONIMA))',
        r'([A-ZÁÉÍÓÚÑ]{3,}(?:\s+[A-ZÁÉÍÓÚÑ]{3,})*(?:\s+S\.?[AL]\.?){0,1})',
        r'([A-ZÁÉÍÓÚÑ]+\s+\d{4}\s+S\.?L\.?)',
        r'([A-ZÁÉÍÓÚÑ][A-Za-záéíóúñ]{2,}(?:\s+[A-ZÁÉÍÓÚÑ][A-Za-záéíóúñ]{2,}){1,3})',
    ]

    for linea in lineas[:10]:
        linea_limpia = linea.strip()
        if len(linea_limpia) < 5:
            continue
        if re.search(r'^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|^[A-Z]-?\d{8}|^\d+\s*€|^C/|^CALLE|^AVDA|FECHA|DATE|^TEL|^FAX|^EMAIL', linea_limpia, re.IGNORECASE):
            continue
        for patron in patrones_empresa:
            match = re.search(patron, linea_limpia)
            if match:
                candidato = match.group(1).strip()
                if len(candidato) > 4 and not candidato.isdigit():
                    es_cliente = any(re.search(term, candidato, re.IGNORECASE) for term in terminos_cliente)
                    if not es_cliente:
                        resultado["proveedor"] = candidato
                        st.write(f"✅ Proveedor encontrado (patrón general): {candidato}")
                        return resultado
    return resultado

# --- Extracción IA con OpenAI ---
def extract_data_with_openai(invoice_text: str) -> dict:
    texto_limpio = limpiar_texto(invoice_text)
    regex_result = extract_with_regex(texto_limpio)
    
    prompt = f"""
    Eres un experto en análisis de facturas españolas. Analiza el texto y extrae:
    1. NÚMERO DE FACTURA
    2. PROVEEDOR (emisor)
    Reglas: el proveedor es quien EMITE. Cliente NO es proveedor (ej: MISU, EL RINCON DE LA TORTILLA).
    Devuelve SOLO JSON:
    {{
        "invoiceNumber": "número o No encontrado",
        "supplier": "proveedor o No encontrado"
    }}
    TEXTO:
    {texto_limpio[:3000]}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Extrae datos de facturas. Devuelve JSON exacto."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=300
        )
        content = response.choices[0].message.content.strip()
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(content)
        resultado = {
            "nro_factura": parsed.get("invoiceNumber", "No encontrado"),
            "proveedor": parsed.get("supplier", "No encontrado")
        }

        if resultado["nro_factura"] and resultado["nro_factura"] != "No encontrado":
            resultado["nro_factura"] = resultado["nro_factura"].strip()
        if resultado["proveedor"] and resultado["proveedor"] != "No encontrado":
            resultado["proveedor"] = resultado["proveedor"].strip()
            terminos_cliente_filtro = ["MISU", "RINCON", "TORTILLA", "RESTAURANTE", "BAR", "CAFETERIA", 
                                       "TABERNA", "ASADOR", "PIZZERIA", "HOTEL", "HOSTAL"]
            es_cliente = any(termino in resultado["proveedor"].upper() for termino in terminos_cliente_filtro)
            if es_cliente:
                st.write(f"❌ Descartado como cliente: {resultado['proveedor']}")
                resultado["proveedor"] = "No encontrado"
            else:
                if "MERCADONA" in resultado["proveedor"].upper():
                    resultado["proveedor"] = "MERCADONA S.A."
                elif "SUPRACAFE" in resultado["proveedor"].upper():
                    resultado["proveedor"] = "SUPRACAFE"

        if resultado["nro_factura"] == "No encontrado" and regex_result["nro_factura"] != "No encontrado":
            resultado["nro_factura"] = regex_result["nro_factura"]
            st.write(f"🔄 Usando regex para factura: {resultado['nro_factura']}")
        if resultado["proveedor"] == "No encontrado" and regex_result["proveedor"] != "No encontrado":
            resultado["proveedor"] = regex_result["proveedor"]
            st.write(f"🔄 Usando regex para proveedor: {resultado['proveedor']}")

        st.write(f"✅ Resultado final - Factura: {resultado['nro_factura']}, Proveedor: {resultado['proveedor']}")
        return resultado

    except json.JSONDecodeError as e:
        st.warning(f"Error al parsear JSON de OpenAI: {e}")
        st.write(f"Contenido recibido: {content}")
        return regex_result
    except Exception as e:
        st.error(f"Error con OpenAI: {e}")
        return regex_result

# --- Procesamiento de archivo ---
def process_file(file) -> dict:
    with tempfile.NamedTemporaryFile(delete=False, suffix=file.name) as tmp:
        tmp.write(file.read())
        tmp_path = pathlib.Path(tmp.name)

    try:
        if tmp_path.suffix.lower() == ".pdf":
            texto = text_from_pdf(tmp_path)
        else:
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
        tmp_path.unlink()

    if not texto.strip():
        return {"archivo": file.name, "nro_factura": "Error: No se pudo extraer texto", "proveedor": "Error: No se pudo extraer texto"}

    with st.expander(f"🔍 Debug - Texto extraído de {file.name}"):
        st.text_area("Texto OCR:", texto[:1000], height=200)

    result = extract_data_with_openai(texto)
    return {"archivo": file.name, **result}

# --- UI Streamlit ---
st.set_page_config(page_title="OCR + OpenAI Facturas", layout="wide")
st.title("📄 Lector de Facturas - OCR + OpenAI (Versión Corregida)")
st.markdown("Sube tus archivos PDF o imagen y extrae el Nº de Factura y Proveedor con mayor precisión.")

# Test de conexión OpenAI
if st.sidebar.button("🧪 Test OpenAI"):
    try:
        test_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Responde solo: OK"}],
            max_tokens=10
        )
        st.sidebar.success("✅ Conexión OpenAI exitosa")
    except Exception as e:
        st.sidebar.error(f"❌ Error de conexión: {e}")

with st.expander("ℹ️ Consejos para mejores resultados"):
    st.markdown("""
    - **Calidad**: imágenes nítidas y con buen contraste
    - **Orientación**: correcta rotación
    - **Formato**: PDF nativo > imagen escaneada
    - **Idioma**: mejor en español/inglés
    - **Importante**: se identifica el **PROVEEDOR (emisor)**, no el cliente
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
        facturas_encontradas = sum(1 for r in resultados if "No encontrado" not in r["nro_factura"] and "Error" not in r["nro_factura"])
        st.metric("Facturas detectadas", facturas_encontradas)
    with col2:
        proveedores_encontrados = sum(1 for r in resultados if "No encontrado" not in r["proveedor"] and "Error" not in r["proveedor"])
        st.metric("Proveedores detectados", proveedores_encontrados)
    with col3:
        st.metric("Total archivos", len(files))
    
    df_display = df.copy()
    def highlight_results(row):
        colors = []
        for col in row.index:
            if col == "archivo":
                colors.append("")
            elif "No encontrado" in str(row[col]):
                colors.append("background-color: #ffeb3b")
            elif "Error" in str(row[col]):
                colors.append("background-color: #f44336; color: white")
            else:
                colors.append("background-color: #4caf50; color: white")
        return colors
    
    st.dataframe(
        df_display.style.apply(highlight_results, axis=1),
        use_container_width=True,
        column_config={
            "archivo": st.column_config.TextColumn("Archivo", width="medium"),
            "nro_factura": st.column_config.TextColumn("Número Factura", width="medium"),
            "proveedor": st.column_config.TextColumn("Proveedor", width="large")
        }
    )
    
    with st.expander("📊 Detalles del procesamiento"):
        for resultado in resultados:
            st.write(f"**{resultado['archivo']}**")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"• Factura: `{resultado['nro_factura']}`")
            with col2:
                st.write(f"• Proveedor: `{resultado['proveedor']}`")
            st.divider()

    st.subheader("📤 Descargar Resultados")

    # -------- PDF corregido --------
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("helvetica", "B", 16)  # usar helvetica
    pdf.cell(0, 10, "Resumen de Facturas Procesadas", ln=True, align="C")
    pdf.ln(10)
    
    # Encabezados
    pdf.set_font("helvetica", "B", 10)
    pdf.set_fill_color(200, 200, 200)
    pdf.cell(70, 10, "Archivo", 1, 0, 'C', 1)
    pdf.cell(60, 10, "Numero Factura", 1, 0, 'C', 1)
    pdf.cell(60, 10, "Proveedor", 1, 1, 'C', 1)
    
    # Filas
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

    out = pdf.output(dest="S")  # bytes o bytearray en fpdf2
    pdf_bytes = bytes(out) if isinstance(out, (bytearray, bytes)) else out.encode("latin-1")

    st.download_button(
        "📥 Descargar PDF",
        data=pdf_bytes,
        file_name="resumen_facturas.pdf",
        mime="application/pdf",
    )
    # -------- fin PDF corregido --------

    # Excel
    excel_buf = io.BytesIO()
    df.to_excel(excel_buf, index=False, engine="openpyxl")
    excel_buf.seek(0)
    st.download_button(
        "📥 Descargar Excel",
        data=excel_buf.getvalue(),
        file_name="facturas_procesadas.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
















