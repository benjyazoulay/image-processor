import streamlit as st
import openai
import zipfile
import io
import requests
import base64
from PIL import Image
import os

# Configuration de la page
st.set_page_config(page_title="Flux LoRA Style Captioner", layout="wide")

# --- FONCTIONS UTILITAIRES ---

def encode_image_to_base64(image):
    """Convertit une image PIL en chaîne base64."""
    buffered = io.BytesIO()
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def resize_for_api(image, max_size=1024):
    """Redimensionne pour l'API sans toucher à l'original."""
    img_copy = image.copy()
    img_copy.thumbnail((max_size, max_size))
    return img_copy

def generate_caption(api_key, image, artist_name, custom_instructions):
    """Appelle GPT-4.1 nano avec des instructions strictes sur le format TOK."""
    client = openai.OpenAI(api_key=api_key)
    
    resized_image = resize_for_api(image)
    base64_image = encode_image_to_base64(resized_image)

    # Construction du prompt dynamique pour forcer la structure
    user_instruction = (
        f"The trigger word is 'TOK'. The artist/style to learn is '{artist_name}'.\n"
        f"Describe this image. You MUST start the caption exactly with this format:\n"
        f"'TOK {artist_name} style [medium] of ...'\n"
        f"Replace [medium] with the specific artistic technique seen in the image (e.g., oil painting, charcoal sketch, watercolor, digital illustration, etc.).\n"
        f"Then describe the subject, composition, lighting and colors."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {
                    "role": "system",
                    "content": custom_instructions
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_instruction},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=300
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Erreur OpenAI : {e}")
        return None

def process_images(images_dict, api_key, artist_name, system_prompt):
    """Boucle de traitement et création du Zip."""
    output_zip_buffer = io.BytesIO()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    total = len(images_dict)
    
    with zipfile.ZipFile(output_zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for i, (filename, image) in enumerate(images_dict.items()):
            status_text.text(f"Traitement de {filename} ({i+1}/{total})...")
            
            # 1. Générer la caption avec la structure imposée
            caption = generate_caption(api_key, image, artist_name, system_prompt)
            
            if caption:
                # 2. Sauvegarder l'image (conversion RGB + PNG)
                img_byte_arr = io.BytesIO()
                if image.mode in ("RGBA", "P"):
                    image = image.convert("RGB")
                image.save(img_byte_arr, format="PNG")
                
                base_name = os.path.splitext(filename)[0]
                
                # Écriture dans le zip
                zip_file.writestr(f"{base_name}.png", img_byte_arr.getvalue())
                zip_file.writestr(f"{base_name}.txt", caption)
            
            progress_bar.progress((i + 1) / total)
            
    status_text.text("Terminé !")
    output_zip_buffer.seek(0)
    return output_zip_buffer

# --- INTERFACE UTILISATEUR ---

st.title("🎨 Flux LoRA Style Dataset Maker")
st.markdown("Créez votre dataset pour un LoRA de style. Le format de sortie sera : `TOK [Artiste] style [medium] of...`")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Clé API OpenAI", type="password", help="Requise pour GPT-4.1 nano Vision")
    
    st.divider()
    
    st.subheader("Paramètres du Style")
    # On force le TOK en backend, mais on demande le nom de l'artiste
    artist_name = st.text_input("Nom de l'artiste / du Style", placeholder="ex: Claude Monet")
    
    st.info(f"Format des captions : \n**TOK {artist_name if artist_name else '[Artiste]'} style [medium] of...**")
    
    with st.expander("Modifier le Prompt Système (Avancé)"):
        default_sys_prompt = """You are an AI assistant specialized in creating high-quality captions for training Flux LoRA models.
Your output must be a single, fluid paragraph.
Focus on describing the visual elements strictly.
Do not write things like 'In this image' or 'This picture depicts'.
Be precise about the artistic medium (oil, pencil, digital, etc.)."""
        system_prompt = st.text_area("System Prompt", value=default_sys_prompt, height=150)

# Main Input
input_method = st.radio("Source des images :", ["Upload Fichiers", "Upload Zip", "URL Zip"], horizontal=True)

images_to_process = {}

# --- GESTION DES INPUTS (Identique précédent) ---
if input_method == "Upload Fichiers":
    uploaded_files = st.file_uploader("Choisir les images", accept_multiple_files=True, type=['png', 'jpg', 'jpeg', 'webp'])
    if uploaded_files:
        for uploaded_file in uploaded_files:
            try:
                img = Image.open(uploaded_file)
                images_to_process[uploaded_file.name] = img
            except: pass

elif input_method == "Upload Zip":
    uploaded_zip = st.file_uploader("Choisir un fichier Zip", type="zip")
    if uploaded_zip:
        try:
            with zipfile.ZipFile(uploaded_zip) as z:
                for filename in z.namelist():
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')) and not filename.startswith('__MACOSX'):
                        with z.open(filename) as f:
                            img = Image.open(io.BytesIO(f.read()))
                            clean_name = os.path.basename(filename)
                            if clean_name: images_to_process[clean_name] = img
        except Exception as e: st.error(f"Erreur Zip : {e}")

elif input_method == "URL Zip":
    zip_url = st.text_input("Entrez l'URL du fichier Zip")
    if zip_url and st.button("Charger depuis l'URL"):
        with st.spinner("Téléchargement..."):
            try:
                r = requests.get(zip_url)
                r.raise_for_status()
                z = zipfile.ZipFile(io.BytesIO(r.content))
                for filename in z.namelist():
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')) and not filename.startswith('__MACOSX'):
                        with z.open(filename) as f:
                            img = Image.open(io.BytesIO(f.read()))
                            clean_name = os.path.basename(filename)
                            if clean_name: images_to_process[clean_name] = img
                st.success(f"{len(images_to_process)} images chargées !")
            except Exception as e: st.error(f"Erreur URL : {e}")

# --- AFFICHAGE ET EXECUTION ---
if images_to_process:
    st.write(f"**{len(images_to_process)} images prêtes.**")
    
    # Preview
    cols = st.columns(6)
    for idx, (name, img) in enumerate(list(images_to_process.items())[:6]):
        cols[idx].image(img, use_container_width=True)
    
    if st.button("🚀 Lancer le Captioning", type="primary"):
        if not api_key:
            st.error("Il manque la clé API OpenAI !")
        elif not artist_name:
            st.error("Il manque le nom de l'artiste !")
        else:
            with st.spinner("Analyse des images par GPT-4.1 nano..."):
                zip_buffer = process_images(images_to_process, api_key, artist_name, system_prompt)
                
                st.balloons()
                st.success("Dataset généré avec succès !")
                
                # Nom du fichier zip optimisé
                safe_artist = "".join(x for x in artist_name if x.isalnum())
                zip_name = f"lora_TOK_{safe_artist}_dataset.zip"
                
                st.download_button(
                    label="📥 Télécharger le Dataset Final",
                    data=zip_buffer,
                    file_name=zip_name,
                    mime="application/zip"
                )