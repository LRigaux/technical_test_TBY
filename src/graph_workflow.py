# src/graph_workflow.py
from typing import TypedDict, List, Dict, Any, Optional
import pandas as pd
from langgraph.graph import StateGraph, END
from langchain_core.prompts import PromptTemplate

# from langchain_huggingface import HuggingFaceEndpoint # Type hint pour llm_client
from langchain_core.runnables import Runnable  # Pour type hint llm_client

# Importer les fonctions des autres modules
from .config import COLUMN_MAPPING, REQUIRED_KEYS, DEFAULT_MODEL_REPO_ID, PROMPT_FILE
from .utils import clean_html
from .data_handler import map_columns
from .llm_models import initialize_llm  # Reste nécessaire pour le type hint potentiel
from .image_processor import process_image
from .db_handler import (
    save_results_to_db,
    get_db_connection,
    close_db_connection
)


# --- Définition de l'État du Graphe (Modifiée) ---
class WorkflowState(TypedDict):
    # Supprimé: uploaded_files, raw_dataframe
    selected_dataframe: Optional[
        pd.DataFrame
    ]  # DataFrame sélectionné par l'utilisateur
    mapped_data: List[
        Dict[str, Any]
    ]  # Liste de dicts avec clés standardisées (sortie du mapping)
    processing_options: Dict[str, Any]  # Options UI (prompt, image harm.)
    llm_client: Optional[
        Runnable
    ]  # Instance du LLM initialisé (HuggingFaceEndpoint ou autre Runnable)
    results: List[Dict[str, Any]]  # Résultats finaux (incluant erreurs spécifiques)
    errors: List[str]  # Erreurs globales du workflow
    # db_connection: Optional[
    #     Any
    # ]  # Connexion DB passée depuis l'app (duckdb.DuckDBPyConnection)


# --- Nœuds du Graphe (Fonctions) ---


# NOUVEAU Noeud: Remplace load_and_map_data
def map_selected_data(state: WorkflowState) -> WorkflowState:
    """
    Prend le DataFrame sélectionné depuis l'état et mappe ses colonnes aux clés standard.
    """
    print("--- Node: map_selected_data ---")
    selected_df = state.get("selected_dataframe")
    errors = state.get("errors", [])
    mapped_data_list = []  # Initialiser au cas où

    if selected_df is None or selected_df.empty:
        errors.append("Aucun DataFrame sélectionné n'a été fourni au workflow.")
        print("Avertissement: DataFrame sélectionné vide ou manquant.")
        state["errors"] = errors
        state["mapped_data"] = []  # Assurer que mapped_data est une liste vide
        return state  # Ne peut pas continuer sans données

    # Utiliser la fonction map_columns importée depuis data_handler
    try:
        mapped_data_list, mapping_errors = map_columns(
            selected_df, COLUMN_MAPPING, REQUIRED_KEYS
        )
        errors.extend(mapping_errors)  # Ajouter les erreurs/warnings de mapping
        print(f"Mapping terminé. {len(mapped_data_list)} lignes mappées.")
    except Exception as e:
        error_msg = f"Erreur critique lors du mapping des colonnes: {e}"
        print(error_msg)
        errors.append(error_msg)
        # En cas d'erreur critique ici, on met mapped_data à vide pour potentiellement arrêter le flux
        mapped_data_list = []

    state["mapped_data"] = mapped_data_list
    state["errors"] = errors
    return state

# --- Noeud: generate_missing_titles  ---
def generate_missing_titles(state: WorkflowState) -> WorkflowState:
    print("--- Node: generate_missing_titles ---")
    mapped_data = state.get("mapped_data", [])
    llm = state.get("llm_client")
    errors = state.get("errors", [])

    if not mapped_data: return state
    if not llm: return state

    # --- Prompt encore plus directif (à essayer !) ---

    prompt_string_option_1 = """**TACHE STRICTE : GENERATION DE TITRE PRODUIT**
        **ENTREE :** Une description de produit de prêt-à-porter.
        **SORTIE ATTENDUE :** UN SEUL titre court (5-10 mots), accrocheur et descriptif en français.
        **REGLES ABSOLUES :**
        1. Ta réponse doit contenir **UNIQUEMENT LE TITRE GÉNÉRÉ**.
        2. **AUCUN** texte avant ou après le titre.
        3. **AUCUNE** explication, justification, commentaire, question, ou suggestion alternative.
        4. **AUCUNE** mise en forme spéciale (pas de guillemets, pas d'astérisques, pas de listes).
        5. Le titre doit être basé **UNIQUEMENT** sur la description fournie.
        6. Ne termine pas par un point.
        **Description du Produit :**
        {description}
        **Titre (Réponse Unique et Directe) :**"""
    
    prompt_string_option_2 = """**Rôle :** Générateur de titres produits E-commerce.
        **Objectif :** Créer UN titre optimisé pour un produit de mode.

        **Instructions :**
        1.  Analyse la `{description}` produit fournie.
        2.  Génère UN SEUL titre en français :
            *   Court : 5 à 10 mots maximum.
            *   Accrocheur : Donne envie de cliquer.
            *   Descriptif : Mentionne la caractéristique principale (type, couleur, matière ou motif).
        3.  **Format de Réponse :** Ta réponse doit être **UNIQUEMENT LE TITRE**, sans aucun autre texte, formatage, ou commentaire.
        **REGLES ABSOLUES :**
        1. Ta réponse doit contenir **UNIQUEMENT LE TITRE GÉNÉRÉ**.
        2. **AUCUN** texte avant ou après le titre.
        3. **AUCUNE** explication, justification, commentaire, question, ou suggestion alternative.
        4. **AUCUNE** mise en forme spéciale (pas de guillemets, pas d'astérisques, pas de listes).
        5. Le titre doit être basé **UNIQUEMENT** sur la description fournie.
        6. Ne termine pas par un point.
        
        **Exemple :**
        Description : Pull doux en cachemire bleu marine avec un col rond classique. Parfait pour l'hiver. Tricoté en Italie.
        Titre : Pull Cachemire Bleu Marine Col Rond Classique

        **Description du Produit :**
        {description}

        **Titre Généré :**"""

    title_prompt_template = PromptTemplate.from_template(prompt_string_option_2)
    # -------------------------------------------------
    title_chain = title_prompt_template | llm

    updated_data = []
    conversational_starts = ["voici", "je recommande", "quel titre", "*", "-", "titre:", "option"] # Mots/symboles à ignorer au début des lignes

    for item in mapped_data:
        if pd.isna(item.get("title")) or not str(item.get("title", "")).strip():
            cleaned_desc = clean_html(item.get("body_html", ""))
            if cleaned_desc:
                try:
                    print(f"Génération titre pour produit ID: {item.get('product_id', 'Inconnu')}")
                    response_message = title_chain.invoke({"description": cleaned_desc})

                    extracted_content = ""
                    if hasattr(response_message, 'content'):
                        extracted_content = str(response_message.content).strip()
                    else:
                        extracted_content = str(response_message).strip()

                    # --- Logique de Parsing Améliorée ---
                    possible_title = "Titre non extrait" # Default
                    lines = extracted_content.splitlines()
                    for line in lines:
                        clean_line = line.strip().replace('"', '').replace("'", "")
                        # Ignorer les lignes vides ou trop courtes
                        if not clean_line or len(clean_line) < 3:
                            continue
                        # Ignorer les lignes commençant par des mots conversationnels (insensible à la casse)
                        if any(clean_line.lower().startswith(start) for start in conversational_starts):
                            continue
                        # Prendre la première ligne qui semble valide
                        possible_title = clean_line
                        break # On a trouvé notre titre potentiel

                    # Limiter la longueur si nécessaire
                    max_title_length = 80 # Un peu plus long pour être sûr
                    if len(possible_title) > max_title_length:
                         possible_title = possible_title[:max_title_length] + "..."
                    # ------------------------------------

                    item['title'] = possible_title
                    print(f"  Titre extrait (parsing amélioré): {item['title']}")
                    # Logguer la réponse brute si le parsing a pu être nécessaire
                    if len(lines) > 1 or possible_title == "Titre non extrait":
                        print(f"  Réponse brute du LLM (titre): {response_message}")

                except Exception as e:
                    error_msg = f"Erreur génération titre pour {item.get('product_id', 'Inconnu')}: {e}"
                    print(error_msg)
                    errors.append(error_msg)
                    item['title'] = "Titre non généré (Erreur)"
            else:
                item['title'] = "Titre non généré (Pas de description)"
        updated_data.append(item)

    state["mapped_data"] = updated_data
    state["errors"] = errors
    return state


# --- Noeud: enhance_descriptions (Peu de changements requis) ---
def enhance_descriptions(state: WorkflowState) -> WorkflowState:
    """Améliore les descriptions en utilisant le prompt principal."""
    print("--- Node: enhance_descriptions ---")
    mapped_data = state.get("mapped_data", [])
    llm = state.get("llm_client")
    options = state.get("processing_options", {})
    user_prompt_str = options.get("prompt", "")
    errors = state.get("errors", [])
    results = state.get("results", [])
    if (
        not results and mapped_data
    ):  # Initialiser seulement si pas déjà fait et s'il y a des données
        results = [
            {
                "product_id": item.get("product_id"),
                "enhanced_description": None,
                "processed_image_path": None,
                "error": None,
            }
            for item in mapped_data
        ]

    if not mapped_data:
        print("Aucune donnée mappée disponible, skip description enhancement.")
        state["results"] = []  # Assurer que results est vide si pas de données
        return state
    if not llm:
        print("LLM non disponible, skip description enhancement.")
        errors.append("LLM non initialisé, impossible d'améliorer les descriptions.")
        # Marquer l'erreur dans chaque résultat potentiel
        for i in range(len(results)):
            results[i]["error"] = "LLM non disponible"
            results[i]["enhanced_description"] = "Erreur: LLM non disponible"
        state["errors"] = errors
        state["results"] = results
        return state
    if not user_prompt_str:
        print("Prompt utilisateur vide, skip description enhancement.")
        errors.append("Prompt utilisateur vide.")
        for i in range(len(results)):
            results[i]["error"] = "Prompt vide"
            results[i]["enhanced_description"] = "Erreur: Prompt vide"
        state["errors"] = errors
        state["results"] = results
        return state

    # Le reste de la logique est largement identique...
    try:
        # S'assurer que les input_variables correspondent aux clés DANS mapped_data
        # (title, vendor, product_type, description nettoyée)
        desc_prompt = PromptTemplate(
            template=user_prompt_str,
            input_variables=["vendor", "product_type", "description", "title"],
        )
        desc_chain = desc_prompt | llm
    except Exception as e:
        error_msg = f"Erreur création chaîne de description: {e}"
        print(error_msg)
        errors.append(error_msg)
        for i in range(len(results)):
            results[i]["error"] = f"Erreur prompt: {e}"
            results[i]["enhanced_description"] = f"Erreur: {e}"
        state["errors"] = errors
        state["results"] = results
        return state

    for i, item in enumerate(mapped_data):
        # Assurer que l'index existe dans results (devrait si initialisé correctement)
        if i >= len(results):
            print(
                f"Erreur interne: Index {i} hors limites pour results (taille {len(results)})"
            )
            continue

        cleaned_desc = clean_html(item.get("body_html", ""))
        if not cleaned_desc:
            results[i]["enhanced_description"] = "Description originale vide."
            results[i]["error"] = "Skipped - No original description"
            continue

        input_data = {
            "vendor": item.get("vendor", "Non spécifié"),
            "product_type": item.get("product_type", ""),  # Peut être vide
            "description": cleaned_desc,
            "title": item.get("title", ""),  # Titre généré ou original (peut être vide)
        }

        try:
            print(f"Génération description pour produit ID: {item.get('product_id', 'Inconnu')}")
            response_message = desc_chain.invoke(input_data)

            # --- Extraction et Vérification ---
            enhanced_desc_str = "Erreur extraction contenu" # Default en cas de problème
            if hasattr(response_message, 'content'):
                # Forcer la conversion en string et nettoyer
                enhanced_desc_str = str(response_message.content).strip()
            else:
                # Fallback (moins probable avec ChatGoogleGenerativeAI)
                enhanced_desc_str = str(response_message).strip()
                print(f"  AVERTISSEMENT: Réponse LLM n'avait pas d'attribut 'content'. Réponse brute utilisée: {response_message}")

            # --- Debugging: Vérifier type et valeur avant assignation ---
            print(f"  DEBUG (enhance_desc): Type avant assignation: {type(enhanced_desc_str)}")
            print(f"  DEBUG (enhance_desc): Valeur avant assignation (premiers 100 chars): {enhanced_desc_str[:100]}")
            # -------------------------------------------------------------

            # Assigner la chaîne de caractères extraite
            results[i]['enhanced_description'] = enhanced_desc_str
            print(f"  Description traitée et assignée.")

        except Exception as e:
            error_msg = f"Erreur génération description pour {item.get('product_id', 'Inconnu')}: {e}"
            print(f"  {error_msg}")
            errors.append(error_msg)
            results[i]['enhanced_description'] = "Erreur lors de la génération."
            results[i]['error'] = str(e)

    state["results"] = results
    state["errors"] = errors
    return state


# --- Noeud: process_images_node (Peu de changements requis) ---
def process_images_node(state: WorkflowState) -> WorkflowState:
    """Traite les images (peut être un nœud séparé)."""
    print("--- Node: process_images ---")
    mapped_data = state.get("mapped_data", [])
    options = state.get("processing_options", {})
    image_options = options.get("image_options", {})
    harmonize = options.get("harmonize_images", False)
    errors = state.get("errors", [])
    results = state.get("results", [])  # Doit exister après enhance_descriptions

    if not mapped_data:
        print("Aucune donnée mappée, skip image processing.")
        # Assurer que results est cohérent (probablement vide si mapped_data est vide)
        state["results"] = results if results else []
        return state

    # Assurer que `results` a la bonne taille si enhance_desc a échoué avant init
    if len(results) != len(mapped_data):
        print(
            f"Avertissement: Incohérence de taille entre mapped_data ({len(mapped_data)}) et results ({len(results)}). Réinitialisation partielle de results."
        )
        results = [
            {
                "product_id": item.get("product_id"),
                "enhanced_description": (
                    results[i].get("enhanced_description")
                    if i < len(results)
                    else "Erreur pré-image"
                ),
                "processed_image_path": None,
                "error": (
                    results[i].get("error") if i < len(results) else "Erreur pré-image"
                ),
            }
            for i, item in enumerate(mapped_data)
        ]

    if not harmonize:
        print("Harmonisation image désactivée.")
        # Assurer que la clé existe dans les résultats même si non traitée
        for i in range(len(results)):
            if "processed_image_path" not in results[i]:
                results[i]["processed_image_path"] = None
        state["results"] = results
        return state

    print(f"Options images: {image_options}")

    for i, item in enumerate(mapped_data):
        # Assurer que l'index existe dans results
        if i >= len(results):
            continue

        img_url = item.get("image_source")
        prod_id = item.get("product_id", f"unknown_{i}")

        if pd.isna(img_url) or not img_url:
            results[i]["processed_image_path"] = None  # Pas d'URL
            continue

        try:
            processed_path = process_image(
                img_url, prod_id, image_options
            )  # Utilise la fonction de image_processor.py
            results[i]["processed_image_path"] = processed_path
            print(f"Image traitée pour {prod_id}: {processed_path}")
        except Exception as e:
            error_msg = f"Erreur traitement image pour {prod_id}: {e}"
            print(f"  {error_msg}")
            errors.append(error_msg)
            # Stocker l'erreur spécifique à l'image si possible, sans écraser une erreur précédente
            if not results[i].get("error"):  # Ne pas écraser une erreur LLM
                results[i]["error"] = f"Erreur Image: {e}"
            results[i]["processed_image_path"] = f"Erreur traitement"

    state["results"] = results
    state["errors"] = errors
    return state


def aggregate_results(state: WorkflowState) -> WorkflowState:
    """Combine les données mappées et les résultats des étapes précédentes."""
    print("--- Node: aggregate_results ---")
    mapped_data = state.get('mapped_data', [])
    results_desc_img = state.get('results', []) # Contient desc, img_path, errors spécifiques
    errors = state.get('errors', []) # Erreurs globales

    if not mapped_data:
        print("Aucune donnée mappée à agréger.")
        state['results'] = [] # Assurer que les résultats finaux sont vides
        return state

    if len(mapped_data) != len(results_desc_img):
        error_msg = f"Erreur critique: Incohérence de taille entre mapped_data ({len(mapped_data)}) et results ({len(results_desc_img)}) avant agrégation."
        print(error_msg)
        errors.append(error_msg)
        state['errors'] = errors
        # Mettre les résultats partiels pour l'UI, même si incohérents
        state['results'] = results_desc_img # Ou une version marquée
        return state

    final_results_for_ui_and_db = []
    for i, item_mapped in enumerate(mapped_data):
        item_result = results_desc_img[i]
        final_item = {
            'product_id': item_mapped.get('product_id'),
            'original_body_html': item_mapped.get('body_html'),
            'generated_title': item_mapped.get('title'),
            'vendor': item_mapped.get('vendor'),
            'product_type': item_mapped.get('product_type'),
            'image_source': item_mapped.get('image_source'),
            'enhanced_description': item_result.get('enhanced_description'),
            'processed_image_path': item_result.get('processed_image_path'),
            'processing_error': item_result.get('error')
        }
        final_results_for_ui_and_db.append(final_item)

    # Mettre les résultats agrégés dans l'état pour le noeud suivant et l'UI
    state['results'] = final_results_for_ui_and_db
    state['errors'] = errors # Reporter les erreurs accumulées
    print(f"Agrégation terminée pour {len(final_results_for_ui_and_db)} enregistrements.")
    return state


# --- Nouveau Noeud pour la persistance ---
def persist_to_db(state: WorkflowState) -> WorkflowState:
    """Sauvegarde les résultats agrégés en base de données."""
    print("--- Node: persist_to_db ---")
    final_results = state.get('results', [])
    # db_conn = state.get('db_connection')
    errors = state.get('errors', [])

    if not final_results:
        print("Aucun résultat agrégé à sauvegarder.")
        return state # Retourner l'état tel quel

    local_conn = None # Initialiser à None
    try:
        print("Ouverture d'une connexion DB locale pour la sauvegarde...")
        # Utiliser la fonction standard pour obtenir une connexion
        local_conn = get_db_connection() # Utilise DB_FILE depuis config par défaut
        if not local_conn:
             # Gérer le cas où get_db_connection échoue (devrait lever une exception)
             raise ConnectionError("Impossible d'obtenir une connexion DB locale.")

        print(f"Tentative de sauvegarde de {len(final_results)} enregistrements agrégés...")
        save_results_to_db(local_conn, final_results) # Appel à la fonction du db_handler
        print(f"{len(final_results)} enregistrements sauvegardés avec succès via connexion locale.")

    except Exception as e:
        error_msg = f"Erreur lors de la sauvegarde en BDD (persist_to_db node): {e}"
        print(error_msg)
        errors.append(error_msg) # Ajouter l'erreur de sauvegarde aux erreurs globales

    finally:
        # --- Toujours tenter de fermer la connexion locale ---
        if local_conn:
            print("Fermeture de la connexion DB locale après sauvegarde.")
            close_db_connection(local_conn)
        # ----------------------------------------------------

    state['errors'] = errors # Assurer que les erreurs (y compris DB) sont dans l'état final
    print("Fin du noeud persist_to_db, retour de l'état.")
    return state # Retourner l'état final


# --- Construction du Graphe (Modifiée) ---
def build_graph():
    workflow = StateGraph(WorkflowState)

    # Ajout des noeuds (avec le nouveau nom et le nouveau noeud)
    workflow.add_node("map_selected", map_selected_data)
    workflow.add_node("gen_titles", generate_missing_titles)
    workflow.add_node("enhance_desc", enhance_descriptions)
    workflow.add_node("proc_images", process_images_node)
    workflow.add_node("aggregate_results", aggregate_results) # Nouveau nom
    workflow.add_node("persist_db", persist_to_db)       # Nouveau noeud

    # Définition des transitions (edges)
    workflow.set_entry_point("map_selected")
    workflow.add_edge("map_selected", "gen_titles")
    workflow.add_edge("gen_titles", "enhance_desc")
    workflow.add_edge("enhance_desc", "proc_images")
    workflow.add_edge("proc_images", "aggregate_results") # Transition vers l'agrégation
    workflow.add_edge("aggregate_results", "persist_db")  # Transition vers la persistance
    workflow.add_edge("persist_db", END)                  # Fin après la persistance

    # Compilation du graphe
    app_graph = workflow.compile()
    print("Graphe LangGraph compilé (avec noeuds séparés pour agrégation et persistance).")
    return app_graph