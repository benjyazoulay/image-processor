import streamlit as st
import openai
import zipfile
import io
import os
import shutil
import tempfile
import base64
from PIL import Image

# Configuration de la page
st.set_page_config(page_title="Flux LoRA Style Captioner", layout="wide")

# --- ETATS DE SESSION ---
if "zip_data" not in st.session_state:
    st.session_state.zip_data = None
if "dataset_name" not in st.session_state:
    st.session_state.dataset_name = "dataset.zip"

# --- FONCTIONS UTILITAIRES ---

def encode_image_to_base64(image_path):
    """Lit une image, la redimensionne pour l'API et l'encode en base64."""
    try:
        with Image.open(image_path) as img:
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            # Resize pour économiser des tokens et de la bande passante
            img.thumbnail((1024, 1024))
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG", quality=90)
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception as e:
        st.error(f"Erreur lors de l'encodage de l'image : {e}")
        return None

def generate_caption(client, image_path, artist_name, custom_system_prompt):
    """
    Appelle l'API OpenAI. Utilise gpt-4.1-mini ou gpt-4.1.
    """
    base64_image = encode_image_to_base64(image_path)
    
    if not base64_image:
        return None

    user_instruction = (
        f"The trigger word is 'TOK'. The artist/style to learn is '{artist_name}'.\n"
        f"Describe this image. You MUST start the caption exactly with this format:\n"
        f"'TOK {artist_name} style [medium] of ...'\n"
        f"Replace [medium] with the specific artistic technique seen in the image.\n"
        f"Then describe the subject, composition, lighting and colors."
    )

    try:
        # CORRECTION MAJEURE : Utilisation d'un modèle valide (gpt-4.1-mini ou gpt-4.1)
        response = client.chat.completions.create(
            model="gpt-4.1-mini",  
            messages=[
                {"role": "system", "content": custom_system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": user_instruction},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                ]}
            ],
            max_completion_tokens=300
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"❌ Erreur API sur {os.path.basename(image_path)} : {e}")
        return None

def process_and_zip(files, input_type, api_key, artist, sys_prompt):
    # Initialisation du client OpenAI ici pour éviter de le recréer à chaque image
    try:
        client = openai.OpenAI(api_key=api_key)
    except Exception as e:
        st.error(f"Erreur de clé API : {e}")
        return

    # Conteneur pour l'affichage des résultats
    status_area = st.empty()
    progress_bar = st.progress(0)
    live_log = st.expander("👁️ Voir les résultats en direct", expanded=True)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        input_dir = os.path.join(temp_dir, "input")
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        # --- 1. EXTRACTION ---
        image_paths = []
        
        if input_type == "zip":
            try:
                # Si c'est un fichier uploadé Streamlit, on s'assure d'être au début du fichier
                if hasattr(files, 'seek'):
                    files.seek(0)
                with zipfile.ZipFile(files, 'r') as z:
                    z.extractall(input_dir)
                
                for root, _, filenames in os.walk(input_dir):
                    for filename in filenames:
                        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')) and not filename.startswith('.'):
                            image_paths.append(os.path.join(root, filename))
            except Exception as e:
                st.error(f"Erreur critique lors de la lecture du ZIP : {e}")
                return
        else:
            # Traitement liste de fichiers
            for uploaded_file in files:
                # Reset pointer au cas où
                uploaded_file.seek(0)
                file_path = os.path.join(input_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                image_paths.append(file_path)

        if not image_paths:
            status_area.error("⚠️ Aucune image trouvée. Vérifiez que le ZIP contient bien des images à la racine ou dans des sous-dossiers.")
            return

        status_area.info(f"📁 {len(image_paths)} images trouvées. Démarrage...")

        # --- 2. TRAITEMENT ---
        success_count = 0
        
        for i, img_path in enumerate(image_paths):
            filename = os.path.basename(img_path)
            basename = os.path.splitext(filename)[0]
            
            caption = generate_caption(client, img_path, artist, sys_prompt)

            if caption:
                success_count += 1
                
                # Sauvegarde Image
                tgt_img_path = os.path.join(output_dir, f"{basename}.png")
                try:
                    with Image.open(img_path) as img:
                        if img.mode in ("RGBA", "P"):
                            img = img.convert("RGB")
                        img.save(tgt_img_path, format="PNG")
                except Exception as e:
                    st.warning(f"Impossible de convertir l'image {filename}: {e}")
                    continue

                # Sauvegarde Texte
                tgt_txt_path = os.path.join(output_dir, f"{basename}.txt")
                with open(tgt_txt_path, "w", encoding="utf-8") as f:
                    f.write(caption)

                # Affichage (On écrit dans le conteneur créé avant la boucle)
                with live_log:
                    cols = st.columns([1, 4])
                    cols[0].image(tgt_img_path, use_container_width=True)
                    cols[1].markdown(f"**{filename}**\n```text\n{caption}\n```")
            
            progress_bar.progress((i + 1) / len(image_paths))

        # --- 3. CLÔTURE ---
        if success_count == 0:
            status_area.error("🛑 Échec total : Aucune caption générée. Vérifiez votre clé API et vos crédits OpenAI.")
            return

        # Création du ZIP
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, filenames in os.walk(output_dir):
                for filename in filenames:
                    file_path = os.path.join(root, filename)
                    zf.write(file_path, arcname=filename)
        
        st.session_state.zip_data = zip_buffer.getvalue()
        safe_artist = "".join(x for x in artist if x.isalnum() or x in (' ', '_', '-')).strip()
        st.session_state.dataset_name = f"lora_TOK_{safe_artist.replace(' ', '_')}_dataset.zip"
        
        status_area.success(f"✅ Terminé ! {success_count} images traitées.")


# --- INTERFACE ---

st.title("🎨 Flux LoRA Style Dataset Maker")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    # On utilise st.session_state pour garder la clé si l'utilisateur change d'onglet
    api_key_input = st.text_input("Clé API OpenAI", type="password", key="api_key_input")
    st.divider()
    
    default_sys_prompt = """You are an expert computer vision assistant specialized in captioning datasets for training FLUX.1 LoRA models.
YOUR MISSION:
Analyze the image and generate a highly detailed, objective description.
CRITICAL INSTRUCTIONS:
1. STRICT FORMAT: Start with "TOK [Artist] style [medium] of...".
2. MEDIUM: Identify if it is oil, watercolor, photo, 3D, etc.
3. STYLE: Describe brushwork, lighting, composition.
4. CONTENT: Describe the subject clearly.
"""
    with st.expander("Modifier le Prompt Système"):
        system_prompt_input = st.text_area("System Prompt", value=default_sys_prompt, height=200)

# CORRECTION : Utilisation de st.form pour éviter les resets intempestifs
with st.form("processing_form"):
    st.info("Configurez vos options ci-dessous et lancez le traitement.")
    
    artist_name_input = st.text_input("Nom de l'artiste / Style", placeholder="ex: Claude Monet")
    
    input_method = st.radio("Source des images :", ["Upload Fichiers", "Upload Zip"], horizontal=True)

    # Les file_uploader doivent être distincts avec des clés uniques
    uploaded_files_list = None
    uploaded_zip_file = None
    
    if input_method == "Upload Fichiers":
        uploaded_files_list = st.file_uploader("Sélectionnez vos images", accept_multiple_files=True, type=['png', 'jpg', 'jpeg', 'webp'])
    else:
        uploaded_zip_file = st.file_uploader("Sélectionnez un fichier ZIP", type="zip")

    # Le bouton Submit est ce qui déclenche tout sans recharger la page prématurément
    submitted = st.form_submit_button("🚀 Lancer le Captioning")

if submitted:
    # Validation
    if not api_key_input:
        st.error("❌ Veuillez entrer une clé API OpenAI.")
    elif not artist_name_input:
        st.error("❌ Veuillez définir un nom d'artiste.")
    elif (input_method == "Upload Fichiers" and not uploaded_files_list) and (input_method == "Upload Zip" and not uploaded_zip_file):
        st.error("❌ Veuillez uploader des fichiers.")
    else:
        # Préparation des arguments
        files_arg = uploaded_zip_file if input_method == "Upload Zip" else uploaded_files_list
        type_arg = "zip" if input_method == "Upload Zip" else "files"
        
        # Reset de l'état précédent
        st.session_state.zip_data = None
        
        # Lancement
        process_and_zip(
            files=files_arg,
            input_type=type_arg,
            api_key=api_key_input,
            artist=artist_name_input,
            sys_prompt=system_prompt_input
        )

# Zone de Téléchargement (hors du formulaire pour se mettre à jour après le traitement)
if st.session_state.zip_data is not None:
    st.divider()
    st.success("✨ Dataset prêt à télécharger !")
    st.download_button(
        label="📥 Télécharger le Dataset (.zip)",
        data=st.session_state.zip_data,
        file_name=st.session_state.dataset_name,
        mime="application/zip"
    )
