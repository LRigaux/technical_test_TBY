# src/graph_workflow.py
from typing import TypedDict, List, Dict, Any, Optional
import pandas as pd
from langgraph.graph import StateGraph, END
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable # Pour chaîner

# Importer les fonctions des autres modules
from .config import COLUMN_MAPPING, REQUIRED_KEYS, DEFAULT_MODEL_REPO_ID, PROMPT_FILE
from .utils import clean_html
from .data_handler import map_columns # Fonction à créer dans data_handler.py
from .llm_models import initialize_llm
from .image_processor import process_image # Fonction existante (légèrement adaptée)
from .db_handler import save_results_to_db # Fonction à créer dans db_handler.py

# --- Définition de l'État du Graphe ---
class WorkflowState(TypedDict):
    uploaded_files: List[Any] # Liste des objets fichiers uploadés par Streamlit
    raw_dataframe: Optional[pd.DataFrame] # DataFrame combiné avant mapping
    mapped_data: List[Dict[str, Any]] # Liste de dicts avec clés standardisées
    processing_options: Dict[str, Any] # Options UI (prompt, image harm.)
    llm_client: Optional[Runnable] # Instance du LLM initialisé
    results: List[Dict[str, Any]] # Résultats finaux (incluant erreurs)
    errors: List[str] # Erreurs globales du workflow

# --- Nœuds du Graphe (Fonctions) ---

def load_and_map_data(state: WorkflowState) -> WorkflowState:
    """Charge les CSV, les combine et mappe les colonnes."""
    print("--- Node: load_and_map_data ---")
    files = state['uploaded_files']
    all_dfs = []
    errors = state.get('errors', [])

    for file in files:
        try:
            df = pd.read_csv(file)
            all_dfs.append(df)
        except Exception as e:
            errors.append(f"Erreur lecture fichier {file.name}: {e}")

    if not all_dfs:
        errors.append("Aucun fichier CSV n'a pu être lu.")
        state['errors'] = errors
        return state # Pas de données à traiter

    # Combiner les dataframes (simpliste, pourrait nécessiter plus de logique)
    raw_df = pd.concat(all_dfs, ignore_index=True)
    state['raw_dataframe'] = raw_df

    # Mapper les colonnes
    mapped_data_list, mapping_errors = map_columns(raw_df, COLUMN_MAPPING, REQUIRED_KEYS)
    errors.extend(mapping_errors)

    state['mapped_data'] = mapped_data_list
    state['errors'] = errors
    print(f"Données mappées: {len(mapped_data_list)} lignes.")
    return state

def generate_missing_titles(state: WorkflowState) -> WorkflowState:
    """Génère des titres si manquants en utilisant le LLM."""
    print("--- Node: generate_missing_titles ---")
    mapped_data = state.get('mapped_data', [])
    llm = state.get('llm_client')
    errors = state.get('errors', [])

    if not llm or not mapped_data:
        print("LLM ou données mappées manquantes, skip title generation.")
        return state

    title_prompt_template = PromptTemplate.from_template(
        "Basé sur la description suivante d'un produit, génère un titre court et accrocheur (max 10 mots) en français.\n\nDescription:\n{description}\n\nTitre:"
    )
    title_chain = title_prompt_template | llm

    updated_data = []
    for item in mapped_data:
        # Vérifie si 'title' existe et n'est pas vide/NaN
        if pd.isna(item.get('title')) or not item.get('title'):
            cleaned_desc = clean_html(item.get('body_html', ''))
            if cleaned_desc:
                try:
                    print(f"Génération titre pour produit ID: {item.get('product_id', 'Inconnu')}")
                    generated_title = title_chain.invoke({"description": cleaned_desc})
                    # Nettoyer la réponse du LLM (peut ajouter des préfixes/suffixes)
                    item['title'] = generated_title.strip().replace('"', '') # Exemple de nettoyage
                    print(f"  Titre généré: {item['title']}")
                except Exception as e:
                    errors.append(f"Erreur génération titre pour {item.get('product_id', 'Inconnu')}: {e}")
                    item['title'] = "Titre non généré (Erreur)" # Marquer l'erreur
            else:
                item['title'] = "Titre non généré (Pas de description)"
        updated_data.append(item)

    state['mapped_data'] = updated_data
    state['errors'] = errors
    return state

def enhance_descriptions(state: WorkflowState) -> WorkflowState:
    """Améliore les descriptions en utilisant le prompt principal."""
    print("--- Node: enhance_descriptions ---")
    mapped_data = state.get('mapped_data', [])
    llm = state.get('llm_client')
    options = state.get('processing_options', {})
    user_prompt_str = options.get('prompt', '')
    errors = state.get('errors', [])
    results = state.get('results', []) # Initialiser si première passe

    if not llm or not mapped_data or not user_prompt_str:
        print("LLM, données ou prompt manquants, skip description enhancement.")
        return state

    try:
        # Adapter les input_variables selon ce qu'on a VRAIMENT après mapping/génération
        # Le prompt doit être flexible ou les variables garanties
        desc_prompt = PromptTemplate(
            template=user_prompt_str,
            # S'assurer que ces variables existent dans chaque 'item' de mapped_data
            input_variables=["vendor", "product_type", "description", "title"]
        )
        desc_chain = desc_prompt | llm
    except Exception as e:
        errors.append(f"Erreur création chaîne de description: {e}")
        state['errors'] = errors
        return state

    # Initialiser results si vide
    if not results:
        results = [{'product_id': item.get('product_id'), 'enhanced_description': None, 'error': None} for item in mapped_data]


    for i, item in enumerate(mapped_data):
        cleaned_desc = clean_html(item.get('body_html', ''))
        if not cleaned_desc:
            results[i]['enhanced_description'] = "Description originale vide."
            results[i]['error'] = "Skipped - No original description"
            continue

        input_data = {
            "vendor": item.get('vendor', 'Non spécifié'),
            "product_type": item.get('product_type', ''), # Peut être vide
            "description": cleaned_desc,
            "title": item.get('title', '') # Titre généré ou original (peut être vide)
        }

        try:
            print(f"Génération description pour produit ID: {item.get('product_id', 'Inconnu')}")
            enhanced_desc = desc_chain.invoke(input_data)
            results[i]['enhanced_description'] = enhanced_desc.strip()
            print(f"  Description générée.")
        except Exception as e:
            error_msg = f"Erreur génération description pour {item.get('product_id', 'Inconnu')}: {e}"
            print(f"  {error_msg}")
            errors.append(error_msg)
            results[i]['enhanced_description'] = "Erreur lors de la génération."
            results[i]['error'] = str(e)

    state['results'] = results
    state['errors'] = errors
    return state

def process_images_node(state: WorkflowState) -> WorkflowState:
    """Traite les images (peut être un nœud séparé)."""
    print("--- Node: process_images ---")
    mapped_data = state.get('mapped_data', [])
    options = state.get('processing_options', {})
    image_options = options.get('image_options', {})
    harmonize = options.get('harmonize_images', False)
    errors = state.get('errors', [])
    results = state.get('results', []) # Doit exister après enhance_descriptions

    if not harmonize or not mapped_data:
        print("Harmonisation image désactivée ou pas de données.")
        # S'assurer que la clé existe dans les résultats même si non traitée
        for i in range(len(results)):
             if 'processed_image_path' not in results[i]:
                 results[i]['processed_image_path'] = None
        state['results'] = results
        return state

    print(f"Options images: {image_options}")

    for i, item in enumerate(mapped_data):
        img_url = item.get('image_source')
        prod_id = item.get('product_id', f'unknown_{i}')

        if pd.isna(img_url) or not img_url:
            results[i]['processed_image_path'] = None # Pas d'URL
            continue

        try:
            processed_path = process_image(img_url, prod_id, image_options) # Utilise la fonction de image_processor.py
            results[i]['processed_image_path'] = processed_path
            print(f"Image traitée pour {prod_id}: {processed_path}")
        except Exception as e:
            error_msg = f"Erreur traitement image pour {prod_id}: {e}"
            print(f"  {error_msg}")
            errors.append(error_msg)
            results[i]['processed_image_path'] = f"Erreur traitement: {e}"


    state['results'] = results
    state['errors'] = errors
    return state

def aggregate_and_save(state: WorkflowState) -> WorkflowState:
    """Combine toutes les infos et sauvegarde en BDD."""
    print("--- Node: aggregate_and_save ---")
    mapped_data = state.get('mapped_data', [])
    results_desc_img = state.get('results', []) # Contient desc, img_path, errors
    errors = state.get('errors', [])
    db_conn = state.get('db_connection') # La connexion doit être passée dans l'état initial

    if not db_conn:
         errors.append("Connexion DB non disponible. Sauvegarde impossible.")
         state['errors'] = errors
         return state

    final_results_for_db = []
    for i, item in enumerate(mapped_data):
        # Combine les données originales mappées avec les résultats générés
        final_item = {
            'product_id': item.get('product_id'),
            'original_body_html': item.get('body_html_original', item.get('body_html')), # Garder l'original si possible
            'enhanced_description': results_desc_img[i].get('enhanced_description'),
            'generated_title': item.get('title'), # Le titre potentiellement généré
            'vendor': item.get('vendor'),
            'product_type': item.get('product_type'),
            'image_source': item.get('image_source'),
            'processed_image_path': results_desc_img[i].get('processed_image_path'),
            'processing_error': results_desc_img[i].get('error') # Erreur spécifique à ce produit
        }
        final_results_for_db.append(final_item)

    # Sauvegarde en BDD (la fonction doit gérer INSERT OR REPLACE)
    try:
        save_results_to_db(db_conn, final_results_for_db)
        print(f"{len(final_results_for_db)} résultats sauvegardés en BDD.")
    except Exception as e:
        error_msg = f"Erreur lors de la sauvegarde en BDD: {e}"
        print(error_msg)
        errors.append(error_msg)

    # Mettre à jour les résultats finaux dans l'état pour affichage UI
    state['results'] = final_results_for_db
    state['errors'] = errors
    return state


# --- Construction du Graphe ---
def build_graph():
    workflow = StateGraph(WorkflowState)

    # Ajout des noeuds
    workflow.add_node("load_map", load_and_map_data)
    workflow.add_node("gen_titles", generate_missing_titles)
    workflow.add_node("enhance_desc", enhance_descriptions)
    workflow.add_node("proc_images", process_images_node)
    workflow.add_node("save_db", aggregate_and_save)

    # Définition des transitions (edges)
    workflow.set_entry_point("load_map")
    workflow.add_edge("load_map", "gen_titles")
    workflow.add_edge("gen_titles", "enhance_desc")
    # On peut exécuter le traitement d'image en parallèle si on veut, mais ici séquentiel pour la simplicité
    workflow.add_edge("enhance_desc", "proc_images")
    workflow.add_edge("proc_images", "save_db")
    workflow.add_edge("save_db", END) # Fin du workflow

    # Compilation du graphe
    app_graph = workflow.compile()
    print("Graphe LangGraph compilé.")
    return app_graph