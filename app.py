# app.py
import streamlit as st
import pandas as pd
import os
import time
from typing import List, Dict, Any, Optional

# Importer depuis les modules src
from src.config import (
    DEFAULT_MODEL_REPO_ID,
    PROMPT_FILE,
    IMAGE_DIR,
    REMOVE_BG_DEFAULT,
    RESIZE_IMAGE_DEFAULT,
    DEFAULT_IMAGE_SIZE,
)
from src.db_handler import (
    get_db_connection,
    close_db_connection,
    fetch_all_enhanced_data,
)
from src.llm_models import initialize_llm
from src.llm_models import initialize_google_llm # Modifier l'import

from src.graph_workflow import (
    build_graph,
    WorkflowState,
)  # On devra peut-être ajuster WorkflowState
from src.data_handler import load_and_combine_csvs, map_columns  # Importer map_columns
from src.utils import is_valid_url  # Pour prévisualisation image

# S'assurer que Pillow/rembg sont bien installés
try:
    from src.image_processor import check_rembg_availability

    REM_BG_AVAILABLE = check_rembg_availability()
except ImportError:
    REM_BG_AVAILABLE = False

# --- Configuration de la Page ---
st.set_page_config(layout="wide", page_title="Amélioration Produit IA V3 (UI Focus)")
st.image("LOGO.png")
st.title("🚀 Amélioration Produit V3 (Interface Améliorée)")

# --- Initialisation de l'État de Session ---
# Nécessaire pour garder les données entre les interactions
if "uploaded_files_list" not in st.session_state:
    st.session_state.uploaded_files_list = []  # Liste des objets fichiers
if "combined_df" not in st.session_state:
    st.session_state.combined_df = None  # DataFrame combiné avant sélection
if "edited_df" not in st.session_state:
    st.session_state.edited_df = None  # DataFrame avec les sélections de l'utilisateur
if "processing_results" not in st.session_state:
    st.session_state.processing_results = None  # Résultats après exécution du graphe
if "loading_errors" not in st.session_state:
    st.session_state.loading_errors = []  # Erreurs lors du chargement CSV

# --- Chargement Configuration & Secrets ---
hf_api_key = st.secrets.get("HUGGINGFACEHUB_API_TOKEN")
google_api_key = st.secrets.get("GOOGLE_API_KEY")
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)

if not hf_api_key:
    st.error(
        "Veuillez configurer `HUGGINGFACEHUB_API_TOKEN` dans `.streamlit/secrets.toml`"
    )
    st.stop()
if not google_api_key: # Priorité à Google pour cet exemple
    st.error("Veuillez configurer `GOOGLE_API_KEY` dans `.streamlit/secrets.toml`")
    st.stop()


# --- Initialisation (Lazy & Cached) ---
@st.cache_resource
def cached_get_db_connection():
    # Assurez-vous que db_handler.py est importé
    from src.db_handler import get_db_connection

    return get_db_connection()


conn = cached_get_db_connection()


@st.cache_resource
def get_compiled_graph():
    # Assurez-vous que graph_workflow.py est importé
    from src.graph_workflow import build_graph

    # IMPORTANT: On devra peut-être adapter le graphe pour qu'il accepte un DF pré-traité
    return build_graph()


app_graph = get_compiled_graph()

# --- Barre Latérale (Configuration Constante) ---
with st.sidebar:
    st.header("⚙️ Configuration Générale")
    # model_repo_id = st.text_input("Modèle Hugging Face Repo ID", DEFAULT_MODEL_REPO_ID)
    google_model_name = st.selectbox(
        "Modèle Google AI",
        ["gemini-1.5-flash", "gemini-1.0-pro", "models/gemini-1.5-pro-latest", "models/gemma-3-27b-it", "models/gemma-3-12b-it"],
        index=0 # Défaut sur Flash
    )

    # Initialisation LLM (Cache basé sur le nom du modèle et la clé)
    @st.cache_resource 
    def get_llm_client_cached(model_name, key):
        print(f"Tentative d'initialisation LLM Google pour {model_name}")
        # Appeler la fonction d'initialisation Google
        return initialize_google_llm(model_name=model_name, api_key=key)

    llm = get_llm_client_cached(google_model_name, google_api_key)

    if not llm:
        st.error("Impossible d'initialiser le LLM Google.")


    
    st.header("📝 Prompt d'Amélioration")
    try:
        with open(PROMPT_FILE, "r", encoding="utf-8") as f:
            default_prompt = f.read()
    except Exception as e:
        st.error(f"Erreur chargement prompt ({PROMPT_FILE}): {e}")
        default_prompt = "Erreur chargement prompt."

    editable_prompt = st.text_area(
        "Modifiez le prompt:", value=default_prompt, height=250
    )

    st.header("🖼️ Options Image")
    harmonize_images = st.checkbox("Harmoniser les images", value=True)
    image_options = {}
    if harmonize_images:
        image_options["remove_bg"] = st.checkbox(
            "Enlever le fond", value=REMOVE_BG_DEFAULT, disabled=not REM_BG_AVAILABLE
        )
        image_options["resize"] = st.checkbox(
            "Redimensionner", value=RESIZE_IMAGE_DEFAULT
        )
        if image_options["resize"]:
            img_width = st.number_input(
                "Largeur max", min_value=50, max_value=2000, value=DEFAULT_IMAGE_SIZE[0]
            )
            img_height = st.number_input(
                "Hauteur max", min_value=50, max_value=2000, value=DEFAULT_IMAGE_SIZE[1]
            )
            image_options["max_size"] = (img_width, img_height)
        if not REM_BG_AVAILABLE and image_options.get("remove_bg"):
            st.warning("'rembg' non trouvé.", icon="⚠️")


# --- Section Principale ---

# 1. Upload des Fichiers
st.header("1. Charger les Fichiers CSV")
uploaded_files = st.file_uploader(
    "Sélectionnez ou glissez-déposez vos fichiers CSV",
    type=["csv"],
    accept_multiple_files=True,
    key="file_uploader",  # Clé pour potentiellement réinitialiser
)

# Détecter si la liste des fichiers a changé
if uploaded_files != st.session_state.uploaded_files_list:
    st.session_state.uploaded_files_list = uploaded_files
    st.session_state.combined_df = None  # Réinitialiser le DF combiné
    st.session_state.edited_df = None  # Réinitialiser le DF édité
    st.session_state.processing_results = None  # Réinitialiser les résultats précédents
    st.session_state.loading_errors = []  # Réinitialiser les erreurs
    if uploaded_files:
        st.info(
            f"{len(uploaded_files)} fichier(s) sélectionné(s). Chargement et combinaison..."
        )
        # Charger et combiner immédiatement
        combined_df, errors = load_and_combine_csvs(uploaded_files)
        st.session_state.combined_df = combined_df
        st.session_state.loading_errors = errors
        if combined_df is not None:
            # Préparer pour l'éditeur: Ajouter la colonne 'Select' si elle n'existe pas
            if "Select" not in combined_df.columns:
                combined_df.insert(0, "Select", False)
            st.session_state.edited_df = (
                combined_df.copy()
            )  # Initialiser l'éditeur avec le DF chargé
        st.rerun()  # Forcer un re-run pour afficher le data_editor

# Afficher les erreurs de chargement s'il y en a
if st.session_state.loading_errors:
    st.warning("Erreurs lors du chargement des fichiers CSV :")
    for error in st.session_state.loading_errors:
        st.error(f"- {error}")

# 2. Prévisualisation et Sélection
st.header("2. Prévisualiser et Sélectionner les Produits")

if st.session_state.edited_df is not None and not st.session_state.edited_df.empty:
    st.info("Cochez les lignes que vous souhaitez traiter.")

    # Boutons de sélection rapide
    col_btn1, col_btn2, _ = st.columns([1, 1, 4])
    with col_btn1:
        if st.button("Tout Sélectionner", key="select_all"):
            st.session_state.edited_df["Select"] = True
            st.rerun()
    with col_btn2:
        if st.button("Tout Désélectionner", key="deselect_all"):
            st.session_state.edited_df["Select"] = False
            st.rerun()

    # --- Configuration dynamique du Data Editor ---
    column_config = {
        "Select": st.column_config.CheckboxColumn(required=True, default=False),
        # Désactiver l'édition des colonnes sources par défaut
        # Trouver une colonne d'image potentielle pour la prévisualisation
    }
    potential_image_cols = [
        "image_source",
        "image_url",
        "image",
        "Image Source",
        "Image URL",
    ]
    image_col_found = None
    for col in potential_image_cols:
        if col in st.session_state.edited_df.columns:
            image_col_found = col
            break

    # Ajouter la config pour l'image si trouvée
    if image_col_found:
        column_config[image_col_found] = st.column_config.ImageColumn(
            label="🖼️ Aperçu Image",
            help="Aperçu de l'image depuis l'URL source",
            width="small",  # ou "medium"
        )
        # On peut aussi vouloir désactiver l'édition de cette colonne URL
        # column_config[image_col_found]['disabled'] = True # Ne semble pas exister pour ImageColumn

    # Désactiver l'édition des autres colonnes sources
    for col in st.session_state.edited_df.columns:
        if (
            col != "Select" and col not in column_config
        ):  # Ne pas écraser la config Select/Image
            column_config[col] = st.column_config.Column(disabled=True)

    # Afficher le data editor
    edited_df_result = st.data_editor(
        st.session_state.edited_df,
        key="data_editor",  # Clé pour accéder à l'état édité
        use_container_width=True,
        hide_index=True,
        column_config=column_config,
        num_rows="dynamic",  # Garder dynamique pour voir toutes les lignes
    )

    # Mettre à jour l'état de session avec les modifications de l'éditeur
    # Vérifier si l'objet retourné est différent (signifie une édition)
    # Note: C'est un peu délicat, parfois il vaut mieux juste relire la clé
    st.session_state.edited_df = (
        edited_df_result  # L'état est mis à jour par st.data_editor lui-même via sa clé
    )

    # Compter les lignes sélectionnées
    selected_rows_df = st.session_state.edited_df[st.session_state.edited_df["Select"]]
    st.info(f"**{len(selected_rows_df)}** produit(s) sélectionné(s) pour traitement.")

else:
    st.warning("Veuillez charger un ou plusieurs fichiers CSV pour commencer.")

# 3. Lancer le Traitement Agentique
st.header("3. Lancer le Traitement IA")

button_disabled = (
    llm is None
    or not editable_prompt
    or editable_prompt == "Erreur chargement prompt."
    or st.session_state.edited_df is None
    or st.session_state.edited_df[st.session_state.edited_df["Select"]].empty
)

if st.button(
    "🚀 Lancer l'Amélioration sur la Sélection",
    type="primary",
    disabled=button_disabled,
):
    selected_df_to_process = st.session_state.edited_df[
        st.session_state.edited_df["Select"]
    ].copy()

    # Préparer l'état initial
    initial_state = {
        "selected_dataframe": selected_df_to_process,
        "mapped_data": [],
        "processing_options": {
            "prompt": editable_prompt,
            "harmonize_images": harmonize_images,
            "image_options": image_options,
        },
        "llm_client": llm,
        "results": [],
        "errors": [],
        # "db_connection": conn,
    }

    st.info("Lancement du workflow agentique...")
    final_state = None
    node_statuses = {}  # Pour suivre l'état de chaque noeud

    # Utiliser st.status pour afficher la progression dynamique
    with st.status(
        "🚀 Initialisation du workflow...", expanded=True
    ) as status_container:
        try:
            # Utiliser app_graph.stream pour obtenir les événements
            event_stream = app_graph.stream(initial_state, stream_mode="values")

            for event in event_stream:
                # La structure de l'événement est un dictionnaire où les clés sont les noms des noeuds
                # et les valeurs sont les sorties de ces noeuds (l'état mis à jour)
                # On peut détecter quel noeud vient de s'exécuter
                final_state = event
                latest_node = list(event.keys())[
                    -1
                ]  # Le noeud le plus récent dans l'événement

                if latest_node not in node_statuses:
                    node_statuses[latest_node] = "running"
                    st.write(f"▶️ Démarrage étape : **{latest_node}**")
                    status_container.update(label=f"⏳ Exécution : {latest_node}...")

                # # Mettre à jour l'état final à chaque événement
                # final_state = event.get(
                #     latest_node
                # )  # L'état après l'exécution du dernier noeud

                # Optionnel: Afficher des détails de l'état pour debug
                # st.write(f"État après {latest_node}: {final_state}")
            print(f"DEBUG: Type of final_state after loop: {type(final_state)}")
            print(f"DEBUG: Keys in final_state after loop: {list(final_state.keys()) if isinstance(final_state, dict) else 'N/A'}")
            # Une fois la boucle terminée, le workflow est fini
            if isinstance(final_state, dict):
                status_container.update(label="✅ Workflow terminé !", state="complete", expanded=False)
                st.session_state.processing_results = final_state # Sauvegarder l'état final correct
                st.success("Traitement terminé avec succès !")

                # Afficher les erreurs globales (maintenant sûr d'utiliser .get())
                if final_state.get('errors'):
                    st.warning("Des erreurs globales sont survenues pendant le traitement :")
                    for error in final_state['errors']:
                        st.error(f"- {error}")
            else:
                # Si final_state n'est pas un dict, il y a eu un problème inattendu
                error_msg = f"Erreur interne: Le workflow s'est terminé mais l'état final est invalide (type: {type(final_state)})."
                print(error_msg)
                st.error(error_msg)
                status_container.update(label="❌ Erreur Workflow (État Final Invalide)", state="error", expanded=True)
                st.session_state.processing_results = None # Ne pas sauvegarder un état invalide

        except Exception as e:
            st.error(f"Erreur critique lors de l'exécution du graphe : {e}")
            st.exception(e)
            status_container.update(label="❌ Erreur Workflow", state="error", expanded=True)
            # Sauvegarder l'état partiel si disponible et si c'est un dict
            if isinstance(final_state, dict):
                st.session_state.processing_results = final_state
            else:
                st.session_state.processing_results = {"errors": [f"Erreur critique: {e}", f"État final invalide: {final_state}"]}

# 4. Affichage des Résultats
st.header("4. Résultats du Traitement")

if isinstance(st.session_state.processing_results, dict) and st.session_state.processing_results.get('results'):
    results_list = st.session_state.processing_results["results"]
    df_results = pd.DataFrame(results_list)

    st.info(f"{len(df_results)} produits traités.")
    # st.dataframe(df_results) # Affichage tabulaire simple pour debug

    # Affichage détaillé par produit
    st.subheader("Détails par Produit Traité")
    for index, res_row in df_results.iterrows():
        # Utiliser l'ID produit comme titre de section si disponible
        product_id_display = res_row.get("product_id", f"Ligne {index+1}")
        with st.expander(f"**Produit ID: {product_id_display}**", expanded=False):
            col_res1, col_res2 = st.columns(2)
            with col_res1:
                st.write("**Titre Généré/Original:**")
                st.caption(res_row.get("generated_title", "N/A"))
                st.write(
                    "**Description Originale (HTML brute):**"
                )  # Afficher l'original pour comparaison
                st.code(res_row.get("original_body_html", "N/A"), language="html")
                st.write("**Image Originale:**")
                img_src = res_row.get("image_source")
                if img_src and pd.notna(img_src) and is_valid_url(img_src):
                    st.image(img_src, width=150)
                else:
                    st.caption("Pas d'image source valide")

            with col_res2:
                st.write("**Description Améliorée:**")
                st.markdown(
                    res_row.get(
                        "enhanced_description", "*Aucune description générée*"
                    ).replace("\n", "  \n")
                )
                st.write("**Image Traitée:**")
                img_processed = res_row.get("processed_image_path")
                if (
                    img_processed
                    and pd.notna(img_processed)
                    and isinstance(img_processed, str)
                    and os.path.exists(img_processed)
                ):
                    st.image(img_processed, width=150)
                elif img_processed:
                    st.caption(f"Image non affichable ({img_processed})")
                else:
                    st.caption("Pas d'image traitée")

                # Afficher l'erreur spécifique au produit si elle existe
                if pd.notna(res_row.get("processing_error")):
                    st.error(
                        f"Erreur spécifique: {res_row['processing_error']}", icon="🚨"
                    )

    # --- Export CSV ---
    st.header("5. Exporter les Résultats")
    csv_export = df_results.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Télécharger les résultats traités en CSV",
        data=csv_export,
        file_name=f"enhanced_products_{time.strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )
elif st.session_state.processing_results:
    st.warning("Le traitement s'est terminé mais aucun résultat structuré n'a été trouvé dans l'état final.")
    # Afficher les erreurs globales si elles existent, même si results est vide/manquant
    if isinstance(st.session_state.processing_results, dict) and st.session_state.processing_results.get('errors'):
        st.warning("Erreurs globales reportées :")
        for error in st.session_state.processing_results['errors']:
            st.error(f"- {error}")


# 5. Historique (Optionnel)
st.header("📚 Historique Complet (depuis DuckDB)")
if st.checkbox("Afficher l'historique complet"):
    try:
        history_df = fetch_all_enhanced_data(conn)
        if not history_df.empty:
            st.dataframe(history_df, use_container_width=True)
        else:
            st.info("Aucune donnée améliorée n'a encore été sauvegardée dans la base.")
    except Exception as e:
        st.error(f"Erreur lors de la récupération de l'historique: {e}")

st.sidebar.info("Version 3 - UI/UX Améliorée")
