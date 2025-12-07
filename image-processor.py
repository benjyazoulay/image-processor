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
            # Resize pour l'API uniquement
            img.thumbnail((1024, 1024))
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG", quality=90)
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception as e:
        st.error(f"Erreur lors de l'encodage de l'image : {e}")
        return None

def generate_caption(api_key, image_path, artist_name, custom_system_prompt):
    """
    Appelle l'API OpenAI. 
    """
    client = openai.OpenAI(api_key=api_key)
    base64_image = encode_image_to_base64(image_path)
    
    if not base64_image:
        return None

    # Prompt Utilisateur (CONSERVÉ STRICTEMENT)
    user_instruction = (
        f"The trigger word is 'TOK'. The artist/style to learn is '{artist_name}'.\n"
        f"Describe this image. You MUST start the caption exactly with this format:\n"
        f"'TOK {artist_name} style [medium] of ...'\n"
        f"Replace [medium] with the specific artistic technique seen in the image (e.g., oil painting, charcoal sketch, watercolor, digital illustration, etc.).\n"
        f"Then describe the subject, composition, lighting and colors."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-5-nano",  # ⚠️ Si ce modèle n'existe pas pour votre clé, l'erreur sera capturée ci-dessous
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
        # C'est ici que ça échoue souvent si le modèle est incorrect
        st.error(f"❌ Erreur OpenAI sur {os.path.basename(image_path)} : {e}")
        return None

def process_and_zip(files, input_type, api_key, artist, sys_prompt):
    """
    Gère le processus complet avec affichage en temps réel.
    """
    # Conteneur pour l'affichage des résultats en direct
    result_container = st.container()

    with tempfile.TemporaryDirectory() as temp_dir:
        input_dir = os.path.join(temp_dir, "input")
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        # --- 1. EXTRACTION / SAUVEGARDE DES ENTRÉES ---
        image_paths = []
        
        if input_type == "zip":
            try:
                with zipfile.ZipFile(files, 'r') as z:
                    z.extractall(input_dir)
                for root, _, filenames in os.walk(input_dir):
                    for filename in filenames:
                        # On ignore les fichiers cachés du genre __MACOSX
                        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')) and not filename.startswith('.'):
                            image_paths.append(os.path.join(root, filename))
            except Exception as e:
                st.error(f"Erreur lors de l'extraction du ZIP : {e}")
                return
        else:
            for uploaded_file in files:
                file_path = os.path.join(input_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                image_paths.append(file_path)

        if not image_paths:
            st.error("⚠️ Aucune image valide trouvée. Vérifiez votre ZIP ou vos fichiers.")
            return

        st.info(f"📁 {len(image_paths)} images prêtes à être traitées.")

        # --- 2. TRAITEMENT BOUCLE ---
        progress_bar = st.progress(0)
        success_count = 0
        
        # On crée des colonnes pour organiser l'affichage "Live"
        live_log = st.expander("👁️ Voir les captions générées en temps réel", expanded=True)

        for i, img_path in enumerate(image_paths):
            filename = os.path.basename(img_path)
            basename = os.path.splitext(filename)[0]
            
            # Appel API
            caption = generate_caption(api_key, img_path, artist, sys_prompt)

            if caption:
                success_count += 1
                
                # 1. Sauvegarde Image (PNG) dans output
                tgt_img_path = os.path.join(output_dir, f"{basename}.png")
                with Image.open(img_path) as img:
                    if img.mode in ("RGBA", "P"):
                        img = img.convert("RGB")
                    img.save(tgt_img_path, format="PNG")

                # 2. Sauvegarde Texte dans output
                tgt_txt_path = os.path.join(output_dir, f"{basename}.txt")
                with open(tgt_txt_path, "w", encoding="utf-8") as f:
                    f.write(caption)

                # 3. Affichage LIVE
                with live_log:
                    cols = st.columns([1, 4])
                    with cols[0]:
                        st.image(tgt_img_path, use_container_width=True)
                    with cols[1]:
                        st.caption(f"**{filename}**")
                        st.code(caption, language="text")
                    st.divider()
            else:
                st.warning(f"⚠️ Pas de caption générée pour {filename} (fichier ignoré du zip).")

            progress_bar.progress((i + 1) / len(image_paths))

        # --- 3. CRÉATION DU ZIP ---
        if success_count == 0:
            st.error("🛑 STOP : Aucune image n'a été traitée avec succès. Le ZIP serait vide.")
            st.error("Vérifiez que le nom du modèle 'gpt-5-nano' est correct et que votre clé API est valide.")
            return

        st.success(f"✅ Traitement terminé : {success_count}/{len(image_paths)} images réussies.")
        
        # Création du ZIP en mémoire
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            files_to_zip = []
            for root, _, filenames in os.walk(output_dir):
                for filename in filenames:
                    files_to_zip.append(os.path.join(root, filename))
            
            if not files_to_zip:
                st.error("Erreur critique : Le dossier de sortie est vide malgré les succès rapportés.")
                return

            for file_path in files_to_zip:
                # On met le fichier à la racine du zip
                zf.write(file_path, arcname=os.path.basename(file_path))
        
        # Finalisation
        st.session_state.zip_data = zip_buffer.getvalue()
        
        safe_artist = "".join(x for x in artist if x.isalnum() or x in (' ', '_', '-')).strip()
        st.session_state.dataset_name = f"lora_TOK_{safe_artist.replace(' ', '_')}_dataset.zip"


# --- INTERFACE ---

st.title("🎨 Flux LoRA Style Dataset Maker (Debug Mode)")
st.markdown("Si le zip est vide, regardez les messages d'erreur rouges qui apparaîtront ci-dessous pendant le traitement.")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    api_key_input = st.text_input("Clé API OpenAI", type="password")
    st.divider()
    artist_name_input = st.text_input("Nom de l'artiste / Style", placeholder="ex: Claude Monet")
    
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
        system_prompt_input = st.text_area("System Prompt", value=default_sys_prompt, height=300)

# Zone principale
input_method = st.radio("Source des images :", ["Upload Fichiers", "Upload Zip"], horizontal=True)

files_to_process = None
is_zip = False

if input_method == "Upload Fichiers":
    files_to_process = st.file_uploader("Sélectionnez vos images", accept_multiple_files=True, type=['png', 'jpg', 'jpeg', 'webp'])
    is_zip = False
else:
    files_to_process = st.file_uploader("Sélectionnez un fichier ZIP", type="zip")
    is_zip = True

# Bouton Action
if st.button("🚀 Lancer le Captioning", type="primary"):
    if not api_key_input:
        st.error("Veuillez entrer une clé API OpenAI.")
    elif not artist_name_input:
        st.error("Veuillez définir un nom d'artiste ou de style.")
    elif not files_to_process:
        st.error("Veuillez uploader des images.")
    else:
        # Reset previous data
        st.session_state.zip_data = None
        process_and_zip(
            files=files_to_process,
            input_type="zip" if is_zip else "files",
            api_key=api_key_input,
            artist=artist_name_input,
            sys_prompt=system_prompt_input
        )

# Zone de Téléchargement
if st.session_state.zip_data is not None:
    st.divider()
    st.success("✨ ZIP généré avec succès !")
    st.download_button(
        label="📥 Télécharger le Dataset (.zip)",
        data=st.session_state.zip_data,
        file_name=st.session_state.dataset_name,
        mime="application/zip"
    )