# app.py
import streamlit as st
import pandas as pd
import os
import time
from dotenv import load_dotenv # Si on utilise .env en plus

# Importer depuis les modules src
from src.config import (DEFAULT_MODEL_REPO_ID, PROMPT_FILE, IMAGE_DIR,
                        REMOVE_BG_DEFAULT, RESIZE_IMAGE_DEFAULT, DEFAULT_IMAGE_SIZE)
from src.db_handler import get_db_connection, close_db_connection, fetch_all_enhanced_data
from src.llm_models import initialize_llm
from src.graph_workflow import build_graph, WorkflowState
# S'assurer que Pillow/rembg sont bien installés
try:
    from src.image_processor import check_rembg_availability
    REM_BG_AVAILABLE = check_rembg_availability()
except ImportError:
     REM_BG_AVAILABLE = False


# --- Configuration de la Page ---
st.set_page_config(layout="wide", page_title="Amélioration Produit IA V2")
st.title("🚀 Amélioration Produit V2 (Agents & LangGraph)")

# --- Chargement Configuration & Secrets ---
# load_dotenv() # Décommenter si .env est utilisé
hf_api_key = st.secrets.get("HUGGINGFACEHUB_API_TOKEN")
# Créer dossier images si besoin
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)

if not hf_api_key:
    st.error("Veuillez configurer `HUGGINGFACEHUB_API_TOKEN` dans `.streamlit/secrets.toml`")
    st.stop()

# --- Initialisation (Lazy) ---
# Connexion DB (gérée par Streamlit via cache_resource)
@st.cache_resource
def cached_get_db_connection():
    return get_db_connection()

conn = cached_get_db_connection()

# Graphe LangGraph (compilé une fois)
@st.cache_resource
def get_compiled_graph():
    return build_graph()

app_graph = get_compiled_graph()

# --- Barre Latérale ---
st.sidebar.header("Configuration")
model_repo_id = st.sidebar.text_input("Modèle Hugging Face Repo ID", DEFAULT_MODEL_REPO_ID)

# Initialisation LLM (peut être mise en cache aussi si l'API Key ne change pas)
# @st.cache_resource # Attention si model_repo_id change
def get_llm_client(repo_id, key):
     return initialize_llm(repo_id, key)

llm = get_llm_client(model_repo_id, hf_api_key)

if not llm:
    st.sidebar.error("Impossible d'initialiser le LLM. Vérifiez l'ID et la clé API.")
    st.stop()

st.sidebar.header("Options de Traitement")
# Chargement/Édition du Prompt
try:
    with open(PROMPT_FILE, 'r', encoding='utf-8') as f:
        default_prompt = f.read()
except Exception as e:
    st.sidebar.error(f"Erreur chargement prompt ({PROMPT_FILE}): {e}")
    default_prompt = "Erreur chargement prompt."

editable_prompt = st.sidebar.text_area(
    "Prompt pour l'amélioration:",
    value=default_prompt,
    height=300
)

# Options Image
harmonize_images = st.sidebar.checkbox("✨ Harmoniser les images produits", value=True)
image_options = {}
if harmonize_images:
    image_options['remove_bg'] = st.sidebar.checkbox("Enlever le fond", value=REMOVE_BG_DEFAULT, disabled=not REM_BG_AVAILABLE)
    image_options['resize'] = st.sidebar.checkbox("Redimensionner", value=RESIZE_IMAGE_DEFAULT)
    if image_options['resize']:
        img_width = st.sidebar.number_input("Largeur max", min_value=50, max_value=2000, value=DEFAULT_IMAGE_SIZE[0])
        img_height = st.sidebar.number_input("Hauteur max", min_value=50, max_value=2000, value=DEFAULT_IMAGE_SIZE[1])
        image_options['max_size'] = (img_width, img_height)
    if not REM_BG_AVAILABLE and image_options.get('remove_bg'):
         st.sidebar.warning("'rembg' non trouvé. L'option 'Enlever le fond' est désactivée.")

# --- Section Upload ---
st.header("1. Charger Fichier(s) CSV Produit")
uploaded_files = st.file_uploader(
    "Sélectionnez ou glissez-déposez vos fichiers CSV",
    type=["csv"],
    accept_multiple_files=True
)

# --- Section Lancement Workflow ---
st.header("2. Lancer le Traitement Agentique")
if st.button("🚀 Lancer l'Amélioration", type="primary", disabled=not uploaded_files):
    if not llm:
        st.error("LLM non disponible.")
    elif not editable_prompt or editable_prompt == "Erreur chargement prompt.":
         st.error("Prompt non valide.")
    else:
        # Préparer l'état initial pour le graphe
        initial_state = WorkflowState(
            uploaded_files=uploaded_files,
            raw_dataframe=None,
            mapped_data=[],
            processing_options={
                "prompt": editable_prompt,
                "harmonize_images": harmonize_images,
                "image_options": image_options
            },
            llm_client=llm,
            results=[],
            errors=[],
            # Passer la connexion DB (attention à la gestion du cycle de vie si non cachée)
            db_connection=conn
        )

        st.info("Lancement du workflow agentique... Suivez la progression dans la console/logs.")
        final_state = None
        with st.spinner("Exécution des agents..."):
            try:
                # Invoquer le graphe LangGraph
                # Le graphe exécute les nœuds séquentiellement comme défini
                final_state = app_graph.invoke(initial_state)

                st.success("Workflow terminé !")

                # Afficher les erreurs globales du workflow
                if final_state and final_state.get('errors'):
                    st.warning("Des erreurs sont survenues pendant le traitement :")
                    for error in final_state['errors']:
                        st.error(f"- {error}")

            except Exception as e:
                st.error(f"Erreur critique lors de l'exécution du graphe : {e}")
                # Afficher l'état partiel si disponible
                if final_state:
                     st.write("État partiel:", final_state)


        # --- Affichage des Résultats (depuis l'état final) ---
        st.header("3. Résultats")
        if final_state and final_state.get('results'):
            df_results = pd.DataFrame(final_state['results'])
            st.dataframe(df_results) # Affichage tabulaire simple

            # Affichage détaillé (optionnel)
            st.subheader("Détails par Produit")
            for index, res_row in df_results.iterrows():
                st.markdown(f"--- **Produit ID: {res_row.get('product_id', 'N/A')}** ---")
                col_res1, col_res2 = st.columns(2)
                with col_res1:
                    st.write("**Titre Généré/Original:**")
                    st.text(res_row.get('generated_title', 'N/A'))
                    st.write("**Description Améliorée:**")
                    st.markdown(res_row.get('enhanced_description', 'N/A').replace('\n', '  \n'))
                    if res_row.get('processing_error'):
                        st.error(f"Erreur spécifique: {res_row['processing_error']}")
                with col_res2:
                    st.write("**Image Originale:**")
                    if pd.notna(res_row.get('image_source')):
                        st.image(res_row['image_source'], width=150)
                    else:
                        st.caption("Pas d'image source")
                    st.write("**Image Traitée:**")
                    if pd.notna(res_row.get('processed_image_path')) and isinstance(res_row.get('processed_image_path'), str) and os.path.exists(res_row['processed_image_path']):
                        st.image(res_row['processed_image_path'], width=150)
                    elif res_row.get('processed_image_path'):
                        st.caption(f"Image non affichable ou erreur: {res_row.get('processed_image_path')}")
                    else:
                        st.caption("Pas d'image traitée")
                st.divider()


            # --- Export CSV ---
            st.header("4. Exporter les Résultats")
            # Re-fetch from DB to ensure consistency or use df_results directly
            csv_export = df_results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Télécharger les résultats en CSV",
                data=csv_export,
                file_name=f"enhanced_products_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )
        else:
            st.warning("Aucun résultat à afficher. Vérifiez les logs ou les erreurs.")

# --- Affichage Données Existantes (Optionnel) ---
st.header("Historique des Données Améliorées (depuis DuckDB)")
if st.checkbox("Afficher l'historique"):
    try:
        history_df = fetch_all_enhanced_data(conn)
        if not history_df.empty:
            st.dataframe(history_df)
        else:
            st.info("Aucune donnée améliorée n'a encore été sauvegardée.")
    except Exception as e:
        st.error(f"Erreur lors de la récupération de l'historique: {e}")


# Note: La connexion DB est fermée automatiquement par Streamlit à la fin de la session grâce à @st.cache_resource
# Si non cachée, il faudrait appeler close_db_connection(conn) quelque part (ex: fin de script, mais complexe avec état Streamlit)

st.sidebar.info("Version 2 - Workflow Agentique")