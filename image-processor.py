import streamlit as st
import openai
import zipfile
import io
import requests
import base64
from PIL import Image
import os
import shutil
import tempfile
import gc  # Garbage Collector pour forcer le nettoyage de la RAM

# Configuration de la page
st.set_page_config(page_title="Flux LoRA Style Captioner", layout="wide")

# --- FONCTIONS UTILITAIRES ---

def encode_image_to_base64(image_path):
    """Lit une image depuis le disque et l'encode en base64."""
    with Image.open(image_path) as img:
        # Conversion RGB si nécessaire
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        # Resize temporaire pour l'API uniquement (pas pour la sauvegarde)
        img.thumbnail((1024, 1024))
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

def generate_caption(api_key, image_path, artist_name, custom_instructions):
    """Appelle GPT-5-nano."""
    client = openai.OpenAI(api_key=api_key)
    base64_image = encode_image_to_base64(image_path)

    user_instruction = (
        f"The trigger word is 'TOK'. The artist/style to learn is '{artist_name}'.\n"
        f"Describe this image. You MUST start the caption exactly with this format:\n"
        f"'TOK {artist_name} style [medium] of ...'\n"
        f"Replace [medium] with the specific artistic technique seen in the image (e.g., oil painting, charcoal sketch, watercolor, digital illustration, etc.).\n"
        f"Then describe the subject, composition, lighting and colors."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-5-nano",
            messages=[
                {"role": "system", "content": custom_instructions},
                {"role": "user", "content": [
                    {"type": "text", "text": user_instruction},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "high"}},
                ]}
            ],
            max_completion_tokens=300
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Erreur OpenAI sur l'image {os.path.basename(image_path)} : {e}")
        return None

def save_processed_image(src_path, dest_folder):
    """Convertit l'image en PNG et la sauvegarde dans le dossier de destination."""
    filename = os.path.basename(src_path)
    base_name = os.path.splitext(filename)[0]
    dest_path = os.path.join(dest_folder, f"{base_name}.png")
    
    with Image.open(src_path) as img:
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        img.save(dest_path, format="PNG")

# --- INTERFACE ---

st.title("🎨 Flux LoRA Style Dataset Maker (Safe Mode)")
st.markdown("Version optimisée pour éviter les crashs mémoire. Les fichiers sont traités sur disque.")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Clé API OpenAI", type="password")
    st.divider()
    artist_name = st.text_input("Nom de l'artiste / du Style", placeholder="ex: Claude Monet")
    
    with st.expander("Modifier le Prompt Système"):
        default_sys_prompt = """You are an expert computer vision assistant specialized in captioning datasets for training FLUX.1 LoRA models.
YOUR MISSION:
Analyze the image and generate a highly detailed, objective description that captures both the subject matter and the specific artistic style.
CRITICAL INSTRUCTIONS:
1. STRICT FORMAT ADHERENCE: You will be given a mandatory starting template (e.g., "TOK [Artist] style [medium] of..."). You must complete this sentence naturally. Do not repeat the template twice, and do not ignore it.
2. MEDIUM DETECTION: Accurately identify the artistic medium (e.g., oil painting, watercolor, charcoal sketch, digital illustration, 3D render, pencil drawing) to fill in the [medium] slot if requested.
3. STYLE ANALYSIS: Focus heavily on the technique. Describe the brushwork (e.g., loose, impasto, smooth), line quality (e.g., thick outlines, delicate hatching), lighting (e.g., chiaroscuro, flat lighting), and color palette.
4. CONTENT DESCRIPTION: Describe the subject, clothing, background, and action clearly.
5. NO FILLER: Never start with "The image shows", "This is a picture of", or "In this scene". Start directly with the trigger phrase.
6. FLUIDITY: Output a single, dense, and coherent paragraph.
"""
        system_prompt = st.text_area("System Prompt", value=default_sys_prompt)

# Input
input_method = st.radio("Source des images :", ["Upload Fichiers", "Upload Zip", "URL Zip"], horizontal=True)

# Container pour stocker le résultat dans la session
if "zip_path" not in st.session_state:
    st.session_state.zip_path = None

def run_process():
    # Création de dossiers temporaires
    # On utilise un gestionnaire de contexte pour s'assurer que c'est nettoyé si ça crash
    temp_dir = tempfile.mkdtemp()
    input_dir = os.path.join(temp_dir, "input")
    output_dir = os.path.join(temp_dir, "output")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    try:
        # 1. Extraction / Sauvegarde des inputs sur le disque
        files_found = []
        
        if input_method == "Upload Fichiers" and uploaded_files:
            for up_file in uploaded_files:
                path = os.path.join(input_dir, up_file.name)
                with open(path, "wb") as f:
                    f.write(up_file.getbuffer())
                files_found.append(path)

        elif input_method == "Upload Zip" and uploaded_zip:
            with zipfile.ZipFile(uploaded_zip) as z:
                z.extractall(input_dir)
            # Scan récursif pour trouver les images extraites
            for root, dirs, files in os.walk(input_dir):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')) and not file.startswith('__MACOSX'):
                        files_found.append(os.path.join(root, file))

        elif input_method == "URL Zip" and zip_url:
            r = requests.get(zip_url)
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall(input_dir)
            for root, dirs, files in os.walk(input_dir):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')) and not file.startswith('__MACOSX'):
                        files_found.append(os.path.join(root, file))

        if not files_found:
            st.warning("Aucune image trouvée.")
            return

        st.info(f"{len(files_found)} images extraites sur le disque temporaire. Début du traitement...")
        
        # 2. Traitement Image par Image
        progress_bar = st.progress(0)
        status = st.empty()
        
        for i, img_path in enumerate(files_found):
            filename = os.path.basename(img_path)
            status.text(f"Traitement : {filename}...")
            
            # A. Générer Caption
            caption = generate_caption(api_key, img_path, artist_name, system_prompt)
            
            if caption:
                # B. Sauvegarder Image (Conversion PNG) dans Output
                save_processed_image(img_path, output_dir)
                
                # C. Sauvegarder TXT dans Output
                base_name = os.path.splitext(filename)[0]
                txt_path = os.path.join(output_dir, f"{base_name}.txt")
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(caption)
            
            # Nettoyage manuel de la mémoire pour être sûr
            gc.collect()
            progress_bar.progress((i + 1) / len(files_found))

        # 3. Création du Zip Final sur le disque
        status.text("Compression du résultat...")
        shutil.make_archive(os.path.join(temp_dir, "dataset_final"), 'zip', output_dir)
        
        # On lit le zip final pour le mettre en session state (ou on le déplace)
        # Pour éviter de tout charger en RAM, on va juste laisser le fichier là temporairement 
        # Mais Streamlit Cloud nettoie le tmp parfois. Le mieux est de le lire une fois.
        final_zip_path = os.path.join(temp_dir, "dataset_final.zip")
        
        with open(final_zip_path, "rb") as f:
            st.session_state.zip_data = f.read()
            
        st.success("Traitement terminé ! Vous pouvez télécharger.")

    except Exception as e:
        st.error(f"Une erreur est survenue : {e}")
    finally:
        # Nettoyage du dossier temporaire
        shutil.rmtree(temp_dir, ignore_errors=True)


# --- SECTION UPLOAD ---
uploaded_files = None
uploaded_zip = None
zip_url = None

if input_method == "Upload Fichiers":
    uploaded_files = st.file_uploader("Images", accept_multiple_files=True, type=['png', 'jpg', 'jpeg', 'webp'])
elif input_method == "Upload Zip":
    uploaded_zip = st.file_uploader("Zip", type="zip")
elif input_method == "URL Zip":
    zip_url = st.text_input("URL Zip")


# --- BOUTON D'ACTION ---
if st.button("🚀 Lancer le Captioning", type="primary"):
    if not api_key or not artist_name:
        st.error("Clé API et Nom de l'artiste requis.")
    else:
        with st.spinner("Traitement en cours..."):
            run_process()

# --- BOUTON DE TÉLÉCHARGEMENT ---
if "zip_data" in st.session_state:
    safe_artist = "".join(x for x in artist_name if x.isalnum())
    st.download_button(
        label="📥 Télécharger le ZIP",
        data=st.session_state.zip_data,
        file_name=f"lora_TOK_{safe_artist}_dataset.zip",
        mime="application/zip"
    )