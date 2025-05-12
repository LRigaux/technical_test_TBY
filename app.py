# app.py
import streamlit as st
import pandas as pd
import os
import time

import plotly.express as px


# --- Configuration et Utilitaires ---
# On importe nos modules maison pour garder le code organisé
from src.config import (
    DEFAULT_MODEL_REPO_ID,
    DATA_DIR,
    IMAGE_DIR,
    REMOVE_BG_DEFAULT,
    RESIZE_IMAGE_DEFAULT,
    DEFAULT_IMAGE_SIZE,
)
from src.db_handler import fetch_all_enhanced_data

from src.llm_models import initialize_llm # pour huggingface (plus de crédit)
from src.llm_models import initialize_google_llm
from src.data_handler import load_and_combine_csvs
from src.utils import is_valid_url, clean_html, analyze_dataframe 

# S'assurer que Pillow/rembg sont bien installés
try:
    from src.image_processor import check_rembg_availability
    REM_BG_AVAILABLE = check_rembg_availability()
except ImportError:
    REM_BG_AVAILABLE = False

# --- Configuration de la page Streamlit ---
st.set_page_config(layout="wide", page_title="Amélioration Produit IA")
st.image("LOGO.png", width=300)
st.title("🚀 Améliorateur de descriptions produits")

# --- Gestion de l'État de Session ---
# st.session_state permet de conserver des informations (données chargées, sélections)
# entre ces réexécutions, essentiel pour une application interactive.
if "uploaded_files_list" not in st.session_state:
    st.session_state.uploaded_files_list = [] 
if "combined_df" not in st.session_state: st.session_state.combined_df = None
if "analysis_results" not in st.session_state: st.session_state.analysis_results = None
if "user_preprocessing_choices" not in st.session_state: st.session_state.user_preprocessing_choices = {}
if "preprocessed_df" not in st.session_state: st.session_state.preprocessed_df = None
if "edited_df" not in st.session_state: st.session_state.edited_df = None
if "processing_results" not in st.session_state:
    st.session_state.processing_results = None
if "loading_errors" not in st.session_state:
    st.session_state.loading_errors = []

# --- Chargement configuration & secrets ---
# hf_api_key = st.secrets.get("HUGGINGFACEHUB_API_TOKEN")
google_api_key = st.secrets.get("GOOGLE_API_KEY") # Priorité à Google
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)

# --- Vérification des clés API ---
#if not hf_api_key:
#   st.error(
#       "Veuillez configurer `HUGGINGFACEHUB_API_TOKEN` dans `.streamlit/secrets.toml`"
#   )
#   st.stop()

if not google_api_key: 
    st.error("Veuillez configurer `GOOGLE_API_KEY` dans `.streamlit/secrets.toml`")
    st.stop()


# --- Initialisation des Ressources Mises en Cache ---
# @st.cache_resource pour les objets lourds ou non sérialisables
# comme les connexions DB, les graphes LangChain, ou les clients LLM.
# Cela évite de les recréer à chaque interaction.
@st.cache_resource
def cached_get_db_connection():
    from src.db_handler import get_db_connection
    return get_db_connection()

@st.cache_resource
def get_compiled_graph():
    from src.graph_workflow import build_graph
    return build_graph()

@st.cache_resource 
def get_llm_client_cached(model_name, key):
    print(f"Tentative d'initialisation LLM Google pour {model_name}")
    return initialize_google_llm(model_name=model_name, api_key=key)

# obtenir les ressources initialisées (récupérées du cache si déjà créées)
conn = cached_get_db_connection()
app_graph = get_compiled_graph()

# --- Barre Latérale de configuration ---
with st.sidebar:
    st.header("⚙️ Configuration générale")
    # model_repo_id = st.text_input("Modèle Hugging Face Repo ID", DEFAULT_MODEL_REPO_ID)
    google_model_name = st.selectbox(
        "Modèle Google AI",
        ["models/gemma-3-27b-it", "models/gemma-3-12b-it", "gemini-1.5-flash", "gemini-1.0-pro", "models/gemini-1.5-pro-latest"],
        index=0,
        help="Choisissez le modèle d'IA pour la génération."
    )

    # Initialisation LLM (cache basé sur le nom du modèle et la clé)
    llm = get_llm_client_cached(google_model_name, google_api_key)

    if not llm:
        st.error("Impossible d'initialiser le LLM Google.")


    # --- Prompt d'amélioration ---
    st.header("📝 Prompt d'amélioration")
    prompt = st.selectbox("Choisir un prompt", ["prompt.txt", "prompt_v2.txt"], index=0)
    if prompt == "prompt.txt":
        PROMPT_FILE = os.path.join(DATA_DIR, "prompt.txt")
    else:
        PROMPT_FILE = os.path.join(DATA_DIR, "prompt_v2.txt")
    try:
        with open(PROMPT_FILE, "r", encoding="utf-8") as f:
            default_prompt = f.read()
    except Exception as e:
        st.error(f"Erreur chargement prompt ({PROMPT_FILE}): {e}")
        default_prompt = "Erreur chargement prompt."

    editable_prompt = st.text_area(
        "Adaptez le prompt si besoin:", value=default_prompt, height=250
    )

    # --- Options de traitement des images ---
    st.header("🖼️ Options image")
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
        image_options['force_format'] = st.selectbox("Forcer Format Sortie", [None, "PNG", "JPEG", "WEBP"], index=0)
        image_options['overwrite'] = st.checkbox("Écraser images existantes", value=False)
        if REM_BG_AVAILABLE:
            image_options['rembg_model'] = st.selectbox("Modèle Rembg", ["u2net", "u2netp", "silueta"], index=0)
        if image_options['force_format'] == 'JPEG':
            image_options['jpeg_quality'] = st.slider("Qualité JPEG", 50, 100, 90)
        if not REM_BG_AVAILABLE and image_options.get("remove_bg"):
            st.warning("'rembg' non trouvé.", icon="⚠️")


# --- Section principale du streamlit---

# Étape 1 : Upload des fichiers
st.header("1. Charger les descriptions")
uploaded_files = st.file_uploader(
    "Sélectionnez ou glissez-déposez vos fichiers CSV",
    type=["csv"],
    accept_multiple_files=True,
    key="file_uploader",  # clé pour potentiellement réinitialiser
)

# logique pour recharger les données si les fichiers uploadés changent
if uploaded_files != st.session_state.uploaded_files_list:
    st.session_state.uploaded_files_list = uploaded_files
    # réinitialise toutes les étapes suivantes
    st.session_state.combined_df = st.session_state.analysis_results = st.session_state.user_preprocessing_choices = None
    st.session_state.preprocessed_df = st.session_state.edited_df = st.session_state.processing_results = None
    st.session_state.loading_errors = []
    if uploaded_files:
        with st.spinner("Chargement et analyse initiale des fichiers..."):
            combined_df, errors = load_and_combine_csvs(uploaded_files)
            st.session_state.combined_df = combined_df
            st.session_state.loading_errors = errors
            if combined_df is not None:
                # analyse immédiatement après le chargement
                st.session_state.analysis_results = analyze_dataframe(combined_df)
                # initialiser le DataFrame prétraité comme étant le combiné au début
                st.session_state.preprocessed_df = combined_df.copy()
                # préparer l'éditeur avec le DF initial (sera mis à jour après prétraitement)
                df_for_editor = st.session_state.preprocessed_df
                if "Select" not in df_for_editor.columns:
                    df_for_editor.insert(0, "Select", False)
                st.session_state.edited_df = df_for_editor.copy()

        st.rerun()  # forcer un re-run pour afficher le data_editor

# affiche les erreurs de chargement s'il y en a
if st.session_state.loading_errors:
    st.warning("Erreurs lors du chargement des fichiers CSV :")
    for error in st.session_state.loading_errors:
        st.error(f"- {error}")

st.header("1.5 Analyser et Préparer les Données")
if st.session_state.analysis_results and st.session_state.combined_df is not None:
    analysis = st.session_state.analysis_results
    st.info(f"Analyse initiale : {analysis['total_rows']} lignes, {analysis['total_cols']} colonnes.")

    with st.expander("Voir les détails de l'analyse", expanded=False):
        st.subheader("Informations Générales")
        st.code(analysis['df_info'])

        st.subheader("Valeurs Manquantes par Colonne")
        if not analysis['missing_stats'].empty:
            # Visualisation
            fig = px.bar(analysis['missing_stats'], y='Manquants (%)', x=analysis['missing_stats'].index,
                         title="Pourcentage de Valeurs Manquantes", text_auto='.2s',
                         labels={'index': 'Colonne'})
            fig.update_traces(textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
            # Tableau détaillé
            st.dataframe(analysis['missing_stats'])
        else:
            st.success("Aucune valeur manquante détectée !")

        st.subheader("Lignes avec Description Manquante")
        st.write(f"Nombre de lignes où la description ('body_html') est vide ou manquante : {analysis['rows_with_missing_desc_count']}")
        # affiche les index ou un extrait
        if analysis['rows_with_missing_desc_indices']:
            st.write("Index des lignes concernées (premiers 10):", analysis['rows_with_missing_desc_indices'][:10])

    st.subheader("Options de Nettoyage (Optionnel)")
    st.markdown("Vous pouvez choisir de retirer certaines colonnes ou lignes avant de continuer.")

    # Initialiser les choix utilisateur s'ils n'existent pas
    if not st.session_state.user_preprocessing_choices:
        st.session_state.user_preprocessing_choices = {
            'cols_to_drop': [],
            'drop_rows_missing_desc': False
        }

    # Option pour supprimer les colonnes suggérées
    cols_suggestion = analysis.get('cols_to_suggest_dropping', [])
    if cols_suggestion:
        st.session_state.user_preprocessing_choices['cols_to_drop'] = st.multiselect(
            f"Colonnes suggérées pour suppression (plus de 95% de valeurs manquantes) :",
            options=cols_suggestion,
            default=st.session_state.user_preprocessing_choices['cols_to_drop'], # Garder le choix précédent
            help="Sélectionnez les colonnes que vous souhaitez retirer définitivement."
        )
    else:
        st.write("Aucune colonne avec un taux de remplissage très faible n'a été détectée.")
        st.session_state.user_preprocessing_choices['cols_to_drop'] = []


    # Option pour supprimer les lignes sans description
    missing_desc_count = analysis.get('rows_with_missing_desc_count', 0)
    if missing_desc_count > 0:
        st.session_state.user_preprocessing_choices['drop_rows_missing_desc'] = st.checkbox(
            f"Retirer les {missing_desc_count} lignes sans description ('body_html') ?",
            value=st.session_state.user_preprocessing_choices['drop_rows_missing_desc'], # Garder le choix précédent
            help="Ces lignes ne pourront probablement pas être traitées efficacement par l'IA."
        )
    else:
        st.write("Toutes les lignes semblent avoir une description.")
        st.session_state.user_preprocessing_choices['drop_rows_missing_desc'] = False

    # Bouton pour appliquer le nettoyage
    if st.button("Appliquer le Nettoyage Sélectionné", key="apply_preprocessing"):
        with st.spinner("Application du nettoyage..."):
            df_to_clean = st.session_state.combined_df.copy()
            choices = st.session_state.user_preprocessing_choices

            # 1. Supprime les colonnes sélectionnées
            cols_to_drop = choices.get('cols_to_drop', [])
            if cols_to_drop:
                df_to_clean = df_to_clean.drop(columns=cols_to_drop, errors='ignore')
                st.success(f"Colonnes retirées : {', '.join(cols_to_drop)}")

            # 2. Supprime les lignes sans description si demandé
            if choices.get('drop_rows_missing_desc', False):
                desc_indices_to_drop = analysis.get('rows_with_missing_desc_indices', [])
                if desc_indices_to_drop:
                    initial_rows = len(df_to_clean)
                    df_to_clean = df_to_clean.drop(index=desc_indices_to_drop, errors='ignore')
                    rows_dropped = initial_rows - len(df_to_clean)
                    st.success(f"{rows_dropped} lignes sans description retirées.")

            # Met à jour le DataFrame prétraité et celui pour l'éditeur
            st.session_state.preprocessed_df = df_to_clean
            # Prépare l'éditeur avec le DF nettoyé
            df_for_editor = st.session_state.preprocessed_df
            if "Select" not in df_for_editor.columns:
                df_for_editor.insert(0, "Select", False)
            else: # Assurer que la colonne Select est réinitialisée après drop
                df_for_editor['Select'] = False
            st.session_state.edited_df = df_for_editor.copy()
            st.info("Nettoyage appliqué. Le tableau de prévisualisation est mis à jour.")
            # pas besoin de rerun ici, le tableau suivant utilisera le nouvel edited_df

elif not st.session_state.uploaded_files_list:
    st.info("En attente du chargement de fichiers CSV pour l'analyse.")

# Étape 2: prévisualisation et sélection
st.header("2. Prévisualiser et sélectionner les produits")

if st.session_state.edited_df is not None and not st.session_state.edited_df.empty:
    st.info("Cochez les lignes que vous souhaitez traiter.")

    # boutons de sélection/désélection rapide
    col_btn1, col_btn2, _ = st.columns([1, 1, 4])
    with col_btn1:
        if st.button("Tout Sélectionner", key="select_all"):
            st.session_state.edited_df["Select"] = True
            st.rerun()
    with col_btn2:
        if st.button("Tout Désélectionner", key="deselect_all"):
            st.session_state.edited_df["Select"] = False
            st.rerun()

    # configuration des colonnes pour l'éditeur (rend les colonnes sources non modifiables)
    column_config = {
        "Select": st.column_config.CheckboxColumn(required=True, default=False),
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

    # ajouter la config pour l'image si trouvée
    if image_col_found:
        column_config[image_col_found] = st.column_config.ImageColumn(
            label="🖼️ Aperçu Image",
            help="Aperçu de l'image depuis l'URL source",
            width="small",  # ou "medium"
        )
    # désactiver l'édition des autres colonnes sources
    for col in st.session_state.edited_df.columns:
        if (
            col != "Select" and col not in column_config
        ):
            column_config[col] = st.column_config.Column(disabled=True)

    # afficher le data editor
    edited_df_result = st.data_editor(
        st.session_state.edited_df,
        key="data_editor",  # clé pour accéder à l'état édité
        use_container_width=True,
        hide_index=True,
        column_config=column_config,
        num_rows="dynamic",  # garder dynamique pour voir toutes les lignes
    )

    # met à jour l'état de session avec les modifications pas l'utilisateur
    # vérifie si l'objet retourné est différent (signifie une édition)
    st.session_state.edited_df = (
        edited_df_result
    )

    # affiche les lignes sélectionnées
    selected_rows_df = st.session_state.edited_df[st.session_state.edited_df["Select"]]
    st.info(f"**{len(selected_rows_df)}** produit(s) sélectionné(s) pour traitement.")

elif st.session_state.uploaded_files_list:
    st.warning("Les données sont en cours d'analyse ou le pré-traitement n'a pas encore été appliqué.")
else:
    st.warning("Veuillez charger un ou plusieurs fichiers CSV pour commencer.")


# Étape 3: Lancer le traitement agentique de la description
st.header("3. Lancer le traitement IA")

button_disabled = (
    llm is None
    or not editable_prompt
    or editable_prompt == "Erreur chargement prompt."
    or st.session_state.edited_df is None
    or st.session_state.edited_df[st.session_state.edited_df["Select"]].empty
)

if st.button(
    "✨ Lancer l'amélioration sur la sélection",
    type="primary",
    disabled=button_disabled,
    icon="✨"
):
    selected_df_to_process = st.session_state.edited_df[
        st.session_state.edited_df["Select"]
    ].copy()

    # préparer l'état initial pour LangGraph
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

    # utiliser st.status pour afficher la progression dynamique
    with st.status(
        "🚀 Initialisation du workflow...", expanded=True
    ) as status_container:
        try:
            # exécute le graphe en mode streaming pour suivre les étapes
            event_stream = app_graph.stream(initial_state, stream_mode="values")

            for event in event_stream:
                final_state = event # garder l'état complet le plus récent
                latest_node = list(event.keys())[
                    -1
                ]

                if latest_node not in node_statuses:
                    node_statuses[latest_node] = "running"
                    status_container.write(f"▶️ Étape: **{latest_node}**")
                    status_container.update(label=f"⏳ En cours: {latest_node}...")

                # afficher des détails de l'état pour debug
                # st.write(f"État après {latest_node}: {final_state}")

            # une fois la boucle terminée, le workflow est fini
            # Vérifier l'état final après la fin du stream
            if isinstance(final_state, dict):
                status_container.update(label="✅ Workflow terminé !", state="complete", expanded=False)
                st.session_state.processing_results = final_state
                st.success("Traitement terminé !")
                if final_state.get('errors'):
                    st.warning("Des erreurs globales sont survenues :")
                    for error in final_state['errors']: st.error(f"- {error}")
            else:
                error_msg = f"Erreur interne: État final invalide (type: {type(final_state)})."
                print(error_msg) # Log pour debug
                st.error(error_msg)
                status_container.update(label="❌ Erreur Workflow", state="error", expanded=True)
                st.session_state.processing_results = None

        except Exception as e:
            st.error(f"Erreur critique lors de l'exécution du graphe : {e}")
            st.exception(e) # Affiche la trace complète pour le debug
            status_container.update(label="❌ Erreur Critique Workflow", state="error", expanded=True)
            st.session_state.processing_results = final_state if isinstance(final_state, dict) else {"errors": [f"Erreur critique: {e}", f"État final partiel: {final_state}"]}

# Étape 4. affichage des résultats
st.header("4. Résultats du traitement")

if isinstance(st.session_state.processing_results, dict) and st.session_state.processing_results.get('results'):
    results_list = st.session_state.processing_results["results"]
    df_results = pd.DataFrame(results_list)

    st.success(f"{len(df_results)} produits traités avec succès (voir détails ci-dessous).")
    # st.dataframe(df_results)

    # affichage détaillé dans des expanders pour garder l'interface propre
    st.subheader("Détails par produit traité")
    for index, res_row in df_results.iterrows():
        # utiliser l'ID produit comme titre de section si disponible
        product_id_display = res_row.get("product_id", f"Ligne {index+1}")
        with st.expander(f"**Produit ID: {product_id_display}**", expanded=False):
            col_res1, col_res2 = st.columns(2)
            with col_res1:
                st.write("**Titre généré/original:**")
                st.caption(res_row.get("generated_title", "N/A"))
                st.write(
                    "**Description originale (nettoyée):**"
                )
                cleaned_original = clean_html(res_row.get("original_body_html", ""))
                st.text_area("", value=cleaned_original, height=150, disabled=True, key=f"clean_orig_{product_id_display}")
                st.write("**Image originale:**")
                img_src = res_row.get("image_source")
                if img_src and pd.notna(img_src) and is_valid_url(img_src):
                    st.image(img_src, width=150)
                else:
                    st.caption("Pas d'image source valide")

            with col_res2:
                st.write("**Titre généré:**")
                st.caption(res_row.get("generated_title", "N/A"))
                st.write("**Description améliorée:**")
                st.markdown(
                    res_row.get(
                        "enhanced_description", "*Aucune description générée*"
                    ).replace("\n", "  \n")
                )
                st.write("**Image traitée:**")
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

    # Etape 5: export CSV
    st.header("5. Exporter les résultats")
    csv_export = df_results.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Télécharger les résultats traités en CSV",
        data=csv_export,
        file_name=f"enhanced_products_{time.strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )
    # gérer le cas où le traitement a eu lieu mais sans résultats (ex: erreur globale)
elif st.session_state.processing_results:
    st.warning("Le traitement s'est terminé mais aucun résultat structuré n'a été trouvé dans l'état final.")
    # afficher les erreurs globales si elles existent, même si results est vide/manquant
    if isinstance(st.session_state.processing_results, dict) and st.session_state.processing_results.get('errors'):
        st.warning("Erreurs globales reportées :")
        for error in st.session_state.processing_results['errors']:
            st.error(f"- {error}")


# Etape 6: historique
st.header("📚 Historique complet (depuis DuckDB)")
if st.checkbox("Afficher l'historique complet"):
    try:
        history_df = fetch_all_enhanced_data(conn)
        if not history_df.empty:
            st.dataframe(history_df, use_container_width=True)
        else:
            st.info("Aucune donnée améliorée n'a encore été sauvegardée dans la base.")
    except Exception as e:
        st.error(f"Erreur lors de la récupération de l'historique: {e}")

st.sidebar.info("Application d'amélioration produit TheBradery - Louis Rigaux")
