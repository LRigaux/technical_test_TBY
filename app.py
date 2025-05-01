import streamlit as st
import pandas as pd
import duckdb
import os
import html2text
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub
from langchain.chains import LLMChain # Simple chain pour ce use case
import requests
from PIL import Image
from io import BytesIO
import time # Pour les timestamps
from rembg import remove


# --- Configuration ---
DB_FILE = "data/product_data.duckdb"
CSV_FILE = "data/descriptions.csv"
PROMPT_FILE = "data/prompt.txt"
IMAGE_DIR = "data/processed_images" # Dossier pour stocker les images traitées (optionnel)

# Modèle Hugging Face (choisir un modèle d'instruction ou texte-vers-texte)
# Exemple: 'mistralai/Mistral-7B-Instruct-v0.1', 'google/flan-t5-large', 'google/flan-t5-base'
# Attention: les modèles gratuits peuvent avoir des limitations (vitesse, contexte)
DEFAULT_MODEL_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.1"

# Créer le dossier pour les images si nécessaire
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)

# --- Fonctions Utilitaires ---

@st.cache_resource # Garde la connexion ouverte pendant la session
def get_db_connection():
    """Initialise et retourne une connexion à la base DuckDB."""
    conn = duckdb.connect(DB_FILE, read_only=False)
    # Créer la table pour les descriptions améliorées si elle n'existe pas
    conn.execute("""
        CREATE TABLE IF NOT EXISTS enhanced_descriptions (
            product_id INTEGER PRIMARY KEY,
            original_body_html TEXT,
            enhanced_description TEXT,
            vendor VARCHAR,
            product_type VARCHAR,
            image_source VARCHAR,
            processed_image_path VARCHAR, -- Chemin vers l'image traitée (bonus)
            last_updated TIMESTAMP
        );
    """)
    return conn

@st.cache_data # Cache les données chargées
def load_data(csv_path):
    """Charge les données depuis le fichier CSV."""
    try:
        df = pd.read_csv(csv_path)
        # Assurer que product_id est bien un entier pour la clé primaire
        df['product_id'] = df['product_id'].astype(int)
        # Ajoute une colonne pour la sélection dans Streamlit
        if 'Select' not in df.columns:
             df.insert(0, 'Select', False)
        return df
    except FileNotFoundError:
        st.error(f"Erreur : Le fichier '{csv_path}' n'a pas été trouvé.")
        return pd.DataFrame() # Retourne un DF vide en cas d'erreur
    except Exception as e:
        st.error(f"Erreur lors du chargement du CSV : {e}")
        return pd.DataFrame()

@st.cache_data
def load_prompt_template(prompt_path):
    """Charge le template de prompt depuis un fichier."""
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        st.error(f"Erreur : Le fichier prompt '{prompt_path}' n'a pas été trouvé.")
        return None # Retourne None si le fichier n'existe pas
    except Exception as e:
        st.error(f"Erreur lors du chargement du prompt : {e}")
        return None

def clean_html(raw_html):
    """Nettoie le code HTML pour obtenir du texte brut."""
    if not raw_html or pd.isna(raw_html):
        return ""
    h = html2text.HTML2Text()
    h.ignore_links = True # Ignorer les liens comme demandé
    h.ignore_images = True
    # Ajout pour mieux gérer les <p> et <br> en sauts de ligne
    h.body_width = 0 # Pas de retour à la ligne automatique
    text = h.handle(str(raw_html))
    # Nettoyages supplémentaires si nécessaire (ex: multiples sauts de ligne)
    text = '\n'.join(line.strip() for line in text.splitlines() if line.strip())
    return text

# --- Initialisation LLM ---
def initialize_llm(api_key, repo_id):
    """Initialise et retourne le modèle LLM via HuggingFaceHub."""
    if not api_key:
        st.error("Clé API Hugging Face non trouvée. Veuillez la configurer dans .streamlit/secrets.toml")
        return None
    try:
        # Configuration des paramètres du modèle si besoin (ex: temperature, max_length)
        llm = HuggingFaceHub(
            repo_id=repo_id,
            huggingfacehub_api_token=api_key,
            model_kwargs={"temperature": 0.6, "max_length": 512} # Ajuster si nécessaire
        )
        return llm
    except Exception as e:
        st.error(f"Erreur lors de l'initialisation du modèle LLM ({repo_id}): {e}")
        return None

# --- Fonction de Traitement d'Image (Bonus) ---
def process_image(image_url, product_id, harmonise_options):
    """Télécharge, traite (optionnel) et sauvegarde une image."""
    processed_image_path = None
    try:
        response = requests.get(image_url, timeout=15)
        response.raise_for_status() # Vérifie les erreurs HTTP
        img_data = response.content
        img = Image.open(BytesIO(img_data))
        original_format = img.format # Garde le format original (JPEG, PNG, etc.)

        processed = False
        # 1. Suppression du fond (si activé et possible)
        if harmonise_options.get('remove_bg'):
            try:
                # rembg retourne des bytes, on le reconvertit en Image PIL
                # S'assurer que l'image a un canal alpha pour la transparence
                img = img.convert("RGBA")
                output_bytes = remove(img_data)
                img = Image.open(BytesIO(output_bytes))
                processed = True
            except Exception as e:
                st.warning(f"Erreur lors de la suppression du fond pour {product_id}: {e}")


        # 2. Redimensionnement (si activé)
        if harmonise_options.get('resize'):
            max_size = harmonise_options.get('max_size', (800, 800))
            img.thumbnail(max_size, Image.Resampling.LANCZOS) # Redimensionne en gardant les proportions
            processed = True

        # 3. Conversion de format (si nécessaire, ex: tout en PNG pour transparence)
        # Si on a supprimé le fond, on sauvegarde en PNG pour garder la transparence
        save_format = 'PNG' if 'A' in img.mode else original_format or 'JPEG'

        # Sauvegarde de l'image traitée (si une action a été faite)
        if processed:
            filename = f"{product_id}_processed.{save_format.lower()}"
            processed_image_path = os.path.join(IMAGE_DIR, filename)
            img.save(processed_image_path, format=save_format)
            st.caption(f"Image traitée sauvegardée : {processed_image_path}")
        else:
             st.caption(f"Image {product_id} non modifiée (aucune option d'harmonisation activée/réussie).")


        return processed_image_path # Retourne le chemin si sauvegardé

    except requests.exceptions.RequestException as e:
        st.warning(f"Impossible de télécharger l'image {product_id} depuis {image_url}: {e}")
        return None
    except Exception as e:
        st.warning(f"Erreur lors du traitement de l'image {product_id}: {e}")
        return None


# --- Interface Streamlit ---
st.set_page_config(layout="wide", page_title="Amélioration Produit IA")
st.title("🚀 Amélioration des Descriptions Produits avec IA")

# --- Chargement des Secrets et Initialisation ---
hf_api_key = st.secrets.get("HUGGINGFACEHUB_API_TOKEN")

if not hf_api_key:
    st.error("Veuillez configurer votre clé `HUGGINGFACEHUB_API_TOKEN` dans `.streamlit/secrets.toml`")
    st.stop()

# Sélection du modèle LLM
model_repo_id = st.sidebar.text_input("Modèle Hugging Face Repo ID", DEFAULT_MODEL_REPO_ID)
llm = initialize_llm(hf_api_key, model_repo_id)

# --- Chargement des Données et du Prompt ---
conn = get_db_connection()
df_products = load_data(CSV_FILE)
prompt_template_str = load_prompt_template(PROMPT_FILE)

if df_products.empty or prompt_template_str is None or llm is None:
    st.warning("L'application ne peut pas continuer en raison d'erreurs de chargement (données, prompt ou LLM). Vérifiez les logs.")
    st.stop() # Arrête l'exécution si les éléments essentiels manquent


# --- Affichage et Sélection des Produits ---
st.header("1. Sélection des Produits à Traiter")

# Utilisation de st.data_editor pour la sélection et potentiellement éditer d'autres champs
edited_df = st.data_editor(
    df_products,
    column_config={
        "Select": st.column_config.CheckboxColumn(required=True),
        "product_id": st.column_config.NumberColumn(disabled=True),
        "vendor": st.column_config.TextColumn(disabled=True),
        "product_type": st.column_config.TextColumn(disabled=True),
        "body_html": st.column_config.TextColumn(disabled=True), # Affichage simple, pas d'édition ici
        "image_source": st.column_config.ImageColumn() # Affiche l'image directement si URL valide
    },
    use_container_width=True,
    hide_index=True,
    num_rows="dynamic" # Permet d'ajouter/supprimer si besoin, mais on le désactive pour ce cas
)

selected_rows = edited_df[edited_df['Select']]

if not selected_rows.empty:
    st.info(f"{len(selected_rows)} produit(s) sélectionné(s).")
else:
    st.warning("Veuillez sélectionner au moins un produit.")


# --- Configuration du Prompt (Bonus) ---
st.header("2. Configuration du Prompt")
editable_prompt = st.text_area(
    "Modifiez le prompt si nécessaire avant la génération :",
    value=prompt_template_str,
    height=300
)

# --- Options de Traitement (Bonus Image) ---
st.header("3. Options de Traitement")
col1_opt, col2_opt = st.columns(2)
with col1_opt:
    harmonize_images = st.checkbox("✨ Harmoniser les images produits (Bonus)")
with col2_opt:
    image_options = {}
    if harmonize_images:
        image_options['remove_bg'] = st.checkbox("Enlever le fond (nécessite 'rembg')", value=True)
        image_options['resize'] = st.checkbox("Redimensionner l'image", value=True)
        if image_options['resize']:
            img_width = st.number_input("Largeur max", min_value=50, max_value=2000, value=800)
            img_height = st.number_input("Hauteur max", min_value=50, max_value=2000, value=800)
            image_options['max_size'] = (img_width, img_height)


# --- Bouton de Génération ---
st.header("4. Lancer la Génération")
if st.button("Générer les Nouvelles Descriptions", type="primary", disabled=selected_rows.empty):
    if not llm:
        st.error("LLM non initialisé. Vérifiez la clé API et le nom du modèle.")
    elif not editable_prompt:
         st.error("Le template de prompt est vide.")
    else:
        # Préparation du LLM Chain
        try:
            prompt = PromptTemplate(
                template=editable_prompt,
                input_variables=["vendor", "product_type", "description"] # "title" n'est pas dans le CSV, on l'enlève ou on adapte
                 # Attention: le prompt original mentionne {title}, mais il n'est pas dans descriptions.csv
                 # Adaptons le prompt ou ignorons {title}. Ici on l'ignore.
                 # Si {title} est nécessaire, il faudrait l'ajouter au CSV ou le déduire autrement.
            )
            # llm_chain = LLMChain(prompt=prompt, llm=llm) # Ancienne méthode
            # Nouvelle méthode avec Runnable sequence
            llm_chain = prompt | llm

        except Exception as e:
            st.error(f"Erreur lors de la création du LLM Chain : {e}")
            st.stop()


        results = []
        total_selected = len(selected_rows)
        progress_bar = st.progress(0)
        status_text = st.empty()

        with st.spinner(f"Traitement de {total_selected} produit(s)..."):
            for i, (index, row) in enumerate(selected_rows.iterrows()):
                product_id = int(row['product_id']) # Assurer que c'est un entier
                status_text.text(f"Traitement du produit {product_id} ({i+1}/{total_selected})...")

                original_html = row['body_html']
                cleaned_description = clean_html(original_html)

                # Préparation des données pour le prompt
                input_data = {
                    "vendor": row['vendor'] or "Non spécifié",
                    "product_type": row['product_type'] or "Non spécifié",
                    "description": cleaned_description
                    # "title": row['title'] or "" # Si la colonne 'title' existait
                }

                enhanced_desc = "Erreur lors de la génération"
                processed_image_final_path = None # Chemin final de l'image traitée

                try:
                    # Appel au LLM
                    response = llm_chain.invoke(input_data)
                    # La réponse peut être une string directe ou un dict selon le LLM/wrapper
                    if isinstance(response, dict) and 'text' in response:
                         enhanced_desc = response['text'].strip()
                    elif isinstance(response, str):
                         enhanced_desc = response.strip()
                    else:
                         enhanced_desc = str(response).strip() # Fallback

                    # Traitement d'image (si activé)
                    if harmonize_images and row['image_source'] and pd.notna(row['image_source']):
                        processed_image_final_path = process_image(row['image_source'], product_id, image_options)

                    # Stockage dans DuckDB (INSERT OR REPLACE pour mettre à jour si existe déjà)
                    current_timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                    conn.execute("""
                        INSERT OR REPLACE INTO enhanced_descriptions
                        (product_id, original_body_html, enhanced_description, vendor, product_type, image_source, processed_image_path, last_updated)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (product_id, original_html, enhanced_desc, row['vendor'], row['product_type'], row['image_source'], processed_image_final_path, current_timestamp))

                    results.append({
                        "ID Produit": product_id,
                        "Description Originale (Nettoyée)": cleaned_description,
                        "Description Améliorée": enhanced_desc,
                        "Image Originale": row['image_source'],
                        "Image Traitée": processed_image_final_path
                    })

                except Exception as e:
                    st.error(f"Erreur lors du traitement du produit {product_id}: {e}")
                    results.append({
                        "ID Produit": product_id,
                        "Description Originale (Nettoyée)": cleaned_description,
                        "Description Améliorée": f"ERREUR: {e}",
                         "Image Originale": row['image_source'],
                        "Image Traitée": "Non traité (erreur)"
                    })

                # Mise à jour de la barre de progression
                progress_bar.progress((i + 1) / total_selected)

            status_text.text(f"Traitement terminé pour {total_selected} produit(s).")
            st.success("Génération terminée !")

            # Affichage des résultats
            st.header("5. Résultats de la Génération")
            df_results = pd.DataFrame(results)

            # Afficher les résultats avec les images si disponibles
            for index, res_row in df_results.iterrows():
                 st.subheader(f"Produit ID: {res_row['ID Produit']}")
                 col_res1, col_res2 = st.columns(2)
                 with col_res1:
                     st.write("**Description Originale (Nettoyée) :**")
                     st.text(res_row['Description Originale (Nettoyée)'])
                     if pd.notna(res_row['Image Originale']):
                         st.image(res_row['Image Originale'], caption="Image Originale", width=200)
                 with col_res2:
                     st.write("**Description Améliorée :**")
                     st.markdown(res_row['Description Améliorée'].replace('\n', '  \n')) # Afficher les sauts de ligne Markdown
                     if pd.notna(res_row['Image Traitée']):
                         try:
                            st.image(res_row['Image Traitée'], caption="Image Traitée", width=200)
                         except Exception as img_e:
                            st.warning(f"Impossible d'afficher l'image traitée: {img_e}")

                 st.divider()


            # Option d'export CSV (Bonus)
            st.header("6. Exporter les Résultats")
            results_db_df = conn.execute("SELECT * FROM enhanced_descriptions WHERE product_id IN (?)", (selected_rows['product_id'].tolist(),)).fetchdf()

            csv_export = results_db_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Télécharger les résultats en CSV",
                data=csv_export,
                file_name=f"enhanced_descriptions_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )

# --- Fermeture de la connexion DB à la fin (bonne pratique) ---
# Note: avec @st.cache_resource, Streamlit gère la fermeture, mais on peut être explicite.
# Le `conn.close()` n'est pas strictement nécessaire ici grâce au cache de ressource.

st.sidebar.info("Application développée par un expert IA.")