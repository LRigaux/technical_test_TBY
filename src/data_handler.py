# src/data_handler.py

import pandas as pd
import streamlit as st
from typing import List, Dict, Any, Tuple, Optional

# Importation de la configuration pour le mapping
# Assurez-vous que config.py est dans le même dossier (ou ajuste l'import)
from .config import COLUMN_MAPPING, REQUIRED_KEYS

def load_and_combine_csvs(uploaded_files: List[st.runtime.uploaded_file_manager.UploadedFile]) -> Tuple[Optional[pd.DataFrame], List[str]]:
    """
    Charge plusieurs fichiers CSV uploadés via Streamlit, les combine en un seul DataFrame.

    Args:
        uploaded_files: Liste des objets fichiers uploadés par Streamlit.

    Returns:
        Un tuple contenant:
        - Le DataFrame combiné (ou None si aucun fichier n'est valide).
        - Une liste de messages d'erreur rencontrés lors de la lecture.
    """
    all_dfs: List[pd.DataFrame] = []
    errors: List[str] = []

    if not uploaded_files:
        errors.append("Aucun fichier fourni.")
        return None, errors

    for file in uploaded_files:
        try:
            # Force l'encodage en UTF-8, souvent nécessaire pour les CSV web
            df = pd.read_csv(file, encoding='utf-8')
            # Optionnel: Nettoyer les noms de colonnes (enlever espaces, etc.)
            df.columns = df.columns.str.strip()
            all_dfs.append(df)
            print(f"Fichier '{file.name}' chargé avec {len(df)} lignes.")
        except pd.errors.EmptyDataError:
            errors.append(f"Avertissement : Le fichier '{file.name}' est vide.")
        except Exception as e:
            errors.append(f"Erreur lors de la lecture du fichier '{file.name}': {e}")

    if not all_dfs:
        errors.append("Aucun des fichiers fournis n'a pu être lu ou ils étaient tous vides.")
        return None, errors

    # Combiner tous les DataFrames lus avec succès
    try:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        print(f"Total de {len(combined_df)} lignes après combinaison.")
        # Optionnel: Supprimer les doublons basés sur product_id si nécessaire
        # if 'product_id' in combined_df.columns:
        #     initial_rows = len(combined_df)
        #     combined_df = combined_df.drop_duplicates(subset=['product_id'], keep='first')
        #     print(f"{initial_rows - len(combined_df)} doublons supprimés basés sur product_id.")

        return combined_df, errors
    except Exception as e:
        errors.append(f"Erreur lors de la combinaison des DataFrames : {e}")
        return None, errors


def map_columns(df: pd.DataFrame, mapping_config: Dict, required_keys: List[str]) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Mappe les colonnes d'un DataFrame aux clés standard définies dans la config.

    Args:
        df: Le DataFrame d'entrée.
        mapping_config: Dictionnaire de configuration du mapping (depuis config.py).
        required_keys: Liste des clés standard qui DOIVENT être trouvées.

    Returns:
        Un tuple contenant:
        - Une liste de dictionnaires, chaque dict représentant une ligne avec des clés standardisées.
        - Une liste de messages d'erreur ou d'avertissements de mapping.
    """
    errors: List[str] = []
    mapped_data_list: List[Dict[str, Any]] = []
    actual_columns = df.columns.tolist()
    standard_keys = mapping_config.get('standard_keys', [])
    possible_names_map = mapping_config.get('possible_names', {})

    found_mapping: Dict[str, str] = {} # {standard_key: actual_column_name}

    print("Début du mapping des colonnes...")
    print(f"Colonnes trouvées dans le CSV: {actual_columns}")
    print(f"Clés standard attendues: {standard_keys}")

    # 1. Trouver le mapping entre clé standard et colonne réelle
    for std_key in standard_keys:
        possible_names = possible_names_map.get(std_key, [])
        found = False
        for potential_name in possible_names:
            if potential_name in actual_columns:
                found_mapping[std_key] = potential_name
                print(f"  Mapping trouvé: '{std_key}' -> '{potential_name}'")
                found = True
                break # Prend le premier nom possible trouvé
        if not found:
            print(f"  Avertissement: Aucune colonne trouvée pour la clé standard '{std_key}'.")
            # On ne met pas d'erreur ici, mais on le note si c'est requis plus bas

    # 2. Vérifier si les clés requises ont été mappées
    missing_required = []
    for req_key in required_keys:
        if req_key not in found_mapping:
            missing_required.append(req_key)

    if missing_required:
        errors.append(f"Erreur critique: Les colonnes requises suivantes n'ont pas pu être mappées : {', '.join(missing_required)}. Le traitement risque d'échouer.")
        # Selon la criticité, on pourrait retourner une liste vide ici:
        # return [], errors

    # 3. Transformer le DataFrame en liste de dictionnaires avec les clés standard
    # Utiliser to_dict('records') pour itérer sur les lignes comme des dictionnaires
    try:
        for index, row in enumerate(df.to_dict('records')):
            new_row_dict: Dict[str, Any] = {}
            # Ajouter toutes les clés standard prévues, même si non mappées (avec None)
            for std_key in standard_keys:
                if std_key in found_mapping:
                    actual_col_name = found_mapping[std_key]
                    # Utiliser row.get pour gérer les valeurs potentiellement manquantes dans la ligne originale
                    new_row_dict[std_key] = row.get(actual_col_name)
                else:
                    # La clé standard n'a pas été trouvée dans le CSV
                    new_row_dict[std_key] = None # Ou une autre valeur par défaut (ex: '')

            # Conversion spéciale pour product_id en entier si possible
            pid_key = 'product_id'
            if pid_key in new_row_dict:
                try:
                    # Gérer les NaN/None avant conversion
                    if pd.notna(new_row_dict[pid_key]):
                        new_row_dict[pid_key] = int(new_row_dict[pid_key])
                    else:
                         # Que faire si product_id est manquant? Erreur? ID temporaire?
                         errors.append(f"Avertissement: product_id manquant ou invalide à la ligne {index+2} du CSV combiné.")
                         new_row_dict[pid_key] = f"invalid_id_{index}" # Ou None, mais la PK DB n'aimera pas

                except (ValueError, TypeError) as e:
                    errors.append(f"Avertissement: Impossible de convertir product_id '{new_row_dict[pid_key]}' en entier à la ligne {index+2}: {e}")
                    new_row_dict[pid_key] = f"invalid_id_{index}" # Marquer comme invalide

            # Ajouter la ligne transformée à la liste
            mapped_data_list.append(new_row_dict)

    except Exception as e:
         errors.append(f"Erreur lors de la transformation des lignes en dictionnaires mappés: {e}")
         return [], errors # Retourner vide en cas d'erreur majeure ici

    print(f"Mapping terminé. {len(mapped_data_list)} lignes prêtes pour le traitement.")
    if errors:
        print("Erreurs/Avertissements de mapping:")
        for err in errors:
            print(f"- {err}")

    return mapped_data_list, errors