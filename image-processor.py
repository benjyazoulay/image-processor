import streamlit as st
import os
from PIL import Image
import io
import zipfile
from openai import OpenAI
import base64
import tempfile

def crop_and_resize_image(image, size):
    width, height = image.size
    new_size = min(width, height)
    left = (width - new_size) / 2
    top = (height - new_size) / 2
    right = (width + new_size) / 2
    bottom = (height + new_size) / 2
    image = image.crop((left, top, right, bottom))
    return image.resize((size, size), Image.LANCZOS)

def describe_image(client, img_path):
    describe_system_prompt = '''
    You are an AI specializing in generating detailed image descriptions for fine-tuning image generation models. Your task is to provide comprehensive, precise, and vivid descriptions of images in English, focusing on the following aspects:

    1. Overall composition and style:
       - Describe the nature of the image (e.g., photograph, painting, digital art, sketch)
       - Identify the artistic style or technique
       - Mention the overall mood or atmosphere of the image

    2. Color palette and lighting:
       - Detail the dominant colors and any significant color contrasts
       - Describe the lighting conditions (e.g., bright, dim, natural, artificial)
       - Note any unique color effects or gradients

    3. Foreground elements:
       - Identify and describe the main subject(s) or focal point(s) of the image
       - Provide details on textures, materials, and patterns of foreground elements
       - For characters or people, describe their appearance, posture, expressions, and attire

    4. Background and environment:
       - Describe the setting or environment (e.g., indoor, outdoor, urban, natural)
       - Note any significant background elements and their relation to the foreground
       - Mention the depth and perspective of the scene

    5. Composition and framing:
       - Describe the layout and arrangement of elements in the image
       - Mention any unique angles, viewpoints, or framing techniques

    6. Details and nuances:
       - Point out any small but significant details that add to the image's character
       - Describe any symbolic elements or themes present in the image
       - Note any unusual or striking features that stand out

    7. Technical aspects (if applicable):
       - Mention any visible effects or post-processing techniques
       - Describe the perceived quality or resolution of the image

    Aim to be as descriptive and precise as possible while keeping the total description under 200 words. Your description should enable an image generation model to recreate a similar image based solely on your text. Avoid making assumptions about the intent behind the image or its creation process unless explicitly evident.
    Short sentences. Optimized for Midjourney or Stable Diffusion. No line break. Non-verbal descriptive sentence, saving words. Limit determiners and linking words as much as possible. The fewest words for the most meaning.
    A description ALLWAYS begins with "TOK style image of".
    70 tokens max.
    '''

    with open(img_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')

    response = client.chat.completions.create(
        model="gpt-4o-mini",  # Utilisation de GPT-4 Vision car gpt-4o-mini n'est pas disponible
        messages=[
            {
                "role": "system",
                "content": describe_system_prompt
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    },
                    {
                        "type": "text",
                        "text": "Provide a detailed description of this image based on the given instructions. Max 70 tokens."
                    }
                ]
            }
        ],
        max_tokens=300,
    )

    return response.choices[0].message.content

def main():
    st.title("Traitement d'images et génération de descriptions")

    # Initialisation de l'état
    if 'processed' not in st.session_state:
        st.session_state.processed = False
        st.session_state.images_zip = None
        st.session_state.descriptions_zip = None

    # Champ pour la clé API
    api_key = st.text_input("Entrez votre clé API OpenAI", type="password")

    uploaded_files = st.file_uploader("Importer des images", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'])

    if uploaded_files and api_key:
        size = st.radio("Taille de redimensionnement", (512, 1024))
        
        if st.button("Traiter les images") or st.session_state.processed:
            if not st.session_state.processed:
                client = OpenAI(api_key=api_key)
                processed_images = []
                descriptions = []

                progress_bar = st.progress(0)
                status_text = st.empty()

                # Création d'un répertoire temporaire
                with tempfile.TemporaryDirectory() as temp_dir:
                    for i, uploaded_file in enumerate(uploaded_files):
                        # Traitement de l'image
                        image = Image.open(uploaded_file)
                        processed_image = crop_and_resize_image(image, size)
                        
                        # Sauvegarde de l'image traitée dans un fichier temporaire
                        temp_file_path = os.path.join(temp_dir, f"temp_image_{i}.jpg")
                        processed_image.save(temp_file_path, format='JPEG')
                        
                        with open(temp_file_path, "rb") as img_file:
                            processed_images.append((f"{i}.jpg", img_file.read()))

                        # Génération de la description
                        with st.spinner(f"Génération de la description pour l'image {i}..."):
                            description = describe_image(client, temp_file_path)
                        descriptions.append((f"{i}.txt", description.encode('utf-8')))

                        # Mise à jour de la barre de progression
                        progress = (i + 1) / len(uploaded_files)
                        progress_bar.progress(progress)
                        status_text.text(f"Traitement de l'image {i+1}/{len(uploaded_files)}")

                # Création des fichiers ZIP
                images_zip = io.BytesIO()
                with zipfile.ZipFile(images_zip, 'w') as zf:
                    for filename, data in processed_images:
                        zf.writestr(filename, data)

                descriptions_zip = io.BytesIO()
                with zipfile.ZipFile(descriptions_zip, 'w') as zf:
                    for filename, data in descriptions:
                        zf.writestr(filename, data)

                # Sauvegarde des ZIP dans l'état de la session
                st.session_state.images_zip = images_zip.getvalue()
                st.session_state.descriptions_zip = descriptions_zip.getvalue()
                st.session_state.processed = True

            # Affichage des boutons de téléchargement
            st.download_button(
                label="Télécharger les images traitées",
                data=st.session_state.images_zip,
                file_name="images_traitees.zip",
                mime="application/zip"
            )

            st.download_button(
                label="Télécharger les descriptions",
                data=st.session_state.descriptions_zip,
                file_name="descriptions.zip",
                mime="application/zip"
            )

    elif not api_key:
        st.warning("Veuillez entrer votre clé API OpenAI pour commencer.")

if __name__ == "__main__":
    main()