# src/db_handler.py
# gère la connexion et l'interaction avec la base de données
import duckdb
import pandas as pd
from typing import List, Dict, Any, Optional
import datetime
import streamlit as st  # Utilisé uniquement pour le décorateur @st.cache_resource

# Importer la configuration pour le chemin de la base de données
from .config import DB_FILE

# Nom de la table principale
TABLE_NAME = "enhanced_descriptions"

# --- Fonctions de Gestion de la Connexion et Initialisation ---


# l@st.cache_resource est généralement appliqué dans app.py
# où la connexion est initialisée pour la session Streamlit.
# Cette fonction est fournie pour être appelée par celle décorée dans app.py.
def get_db_connection(db_path: str = DB_FILE) -> duckdb.DuckDBPyConnection:
    """
    Établit une connexion à la base de données DuckDB et initialise la table si nécessaire.

    Args:
        db_path: Chemin vers le fichier de la base de données DuckDB.

    Returns:
        Un objet de connexion DuckDB.
    """
    print(f"Connexion à la base de données DuckDB : {db_path}")
    try:
        conn = duckdb.connect(database=db_path, read_only=False)
        _initialize_database(conn)
        print("Connexion réussie et base de données initialisée.")
        return conn
    except Exception as e:
        print(f"Erreur critique lors de la connexion ou initialisation de DuckDB: {e}")
        # Dans un contexte Streamlit, on pourrait utiliser st.error ici,
        # mais gardons ce module indépendant de l'UI.
        raise  # Propage l'erreur pour que l'appelant la gère


def _initialize_database(conn: duckdb.DuckDBPyConnection):
    """
    Crée la table des descriptions améliorées si elle n'existe pas déjà.
    Utilise le schéma défini implicitement par le workflow.

    Args:
        conn: La connexion DuckDB active.
    """
    create_table_sql = f"""
    CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
        product_id BIGINT PRIMARY KEY,           -- Identifiant unique du produit
        original_body_html TEXT,                 -- Description HTML originale
        enhanced_description TEXT,               -- Description générée par l'IA
        generated_title VARCHAR,                 -- Titre généré/original utilisé
        vendor VARCHAR,                          -- Vendeur/Marque
        product_type VARCHAR,                    -- Type de produit
        image_source VARCHAR,                    -- URL de l'image originale
        processed_image_path VARCHAR,            -- Chemin local de l'image traitée
        processing_error TEXT,                   -- Erreur spécifique lors du traitement de ce produit
        last_updated TIMESTAMP                   -- Horodatage de la dernière mise à jour
    );
    """
    try:
        conn.execute(create_table_sql)
        conn.commit()  # s'assurer que la création de table est persistée
        print(f"Table '{TABLE_NAME}' vérifiée/créée avec succès.")
    except Exception as e:
        print(
            f"Erreur lors de la création/vérification de la table '{TABLE_NAME}': {e}"
        )
        raise  # propage pour indiquer un problème d'initialisation


def close_db_connection(conn: Optional[duckdb.DuckDBPyConnection]):
    """
    Ferme la connexion à la base de données DuckDB si elle est ouverte.

    Args:
        conn: L'objet de connexion DuckDB (peut être None).
    """
    if conn:
        try:
            conn.close()
            print("Connexion DuckDB fermée.")
        except Exception as e:
            print(f"Erreur lors de la fermeture de la connexion DuckDB: {e}")


# --- Fonctions d'Interaction avec les Données ---


def save_results_to_db(conn: duckdb.DuckDBPyConnection, results: List[Dict[str, Any]]):
    """
    Sauvegarde une liste de résultats de produits dans la base de données.
    Utilise INSERT OR REPLACE pour insérer de nouvelles lignes ou mettre à jour
    les lignes existantes basées sur product_id.

    Args:
        conn: La connexion DuckDB active.
        results: Une liste de dictionnaires, où chaque dictionnaire représente
                 un produit traité et contient les clés correspondant aux colonnes
                 de la table (ex: 'product_id', 'enhanced_description', etc.).
    """
    if not results:
        print("Aucun résultat à sauvegarder en base de données.")
        return

    # Préparer les données pour l'insertion par lots
    data_to_insert = []
    current_timestamp = datetime.datetime.now()

    # Définir les colonnes attendues dans l'ordre de la table (sauf last_updated qui est ajouté)
    # IMPORTANT: L'ordre ici DOIT correspondre à l'ordre des '?' dans la requête SQL
    expected_columns = [
        "product_id",
        "original_body_html",
        "enhanced_description",
        "generated_title",
        "vendor",
        "product_type",
        "image_source",
        "processed_image_path",
        "processing_error",
    ]

    for record in results:
        # Créer un tuple avec les valeurs dans le bon ordre, gérant les clés manquantes
        record_tuple = tuple(record.get(col) for col in expected_columns) + (
            current_timestamp,
        )  # Ajouter le timestamp à la fin
        data_to_insert.append(record_tuple)

    # Construire la requête SQL INSERT OR REPLACE
    # Le nombre de '?' doit correspondre au nombre de colonnes + timestamp
    placeholders = ", ".join(["?"] * (len(expected_columns) + 1))
    sql = f"""
    INSERT OR REPLACE INTO {TABLE_NAME} (
        product_id, original_body_html, enhanced_description, generated_title,
        vendor, product_type, image_source, processed_image_path,
        processing_error, last_updated
    ) VALUES ({placeholders});
    """

    try:
        print(
            f"Tentative de sauvegarde de {len(data_to_insert)} enregistrements dans '{TABLE_NAME}'..."
        )
        # Utiliser executemany pour une insertion/remplacement efficace par lots
        conn.executemany(sql, data_to_insert)
        conn.commit()  # Persister les changements
        print(
            f"{len(data_to_insert)} enregistrements sauvegardés/mis à jour avec succès."
        )
    except Exception as e:
        print(f"Erreur lors de la sauvegarde des résultats en base de données: {e}")
        # On pourrait tenter un rollback ici, mais DuckDB gère souvent bien les transactions atomiques
        # conn.rollback()
        # Propage l'erreur pour que le workflow puisse la logger
        raise


def fetch_all_enhanced_data(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Récupère toutes les données de la table des descriptions améliorées.

    Args:
        conn: La connexion DuckDB active.

    Returns:
        Un DataFrame Pandas contenant toutes les données, ou un DataFrame vide en cas d'erreur.
    """
    try:
        print(f"Récupération de toutes les données depuis '{TABLE_NAME}'...")
        # Utiliser fetchdf() pour obtenir directement un DataFrame Pandas
        df = conn.table(TABLE_NAME).fetchdf()
        print(f"{len(df)} enregistrements récupérés.")
        return df
    except Exception as e:
        print(f"Erreur lors de la récupération des données depuis '{TABLE_NAME}': {e}")
        # Retourner un DataFrame vide pour éviter de planter l'UI
        return pd.DataFrame()


# Exemple d'une fonction pour récupérer un produit spécifique (non utilisée actuellement mais utile)
def fetch_product_by_id(
    conn: duckdb.DuckDBPyConnection, product_id: int
) -> Optional[Dict[str, Any]]:
    """
    Récupère les données d'un produit spécifique par son ID.

    Args:
        conn: La connexion DuckDB active.
        product_id: L'ID du produit à récupérer.

    Returns:
        Un dictionnaire représentant le produit, ou None si non trouvé ou en cas d'erreur.
    """
    try:
        print(f"Recherche du produit ID {product_id} dans '{TABLE_NAME}'...")
        # Utiliser fetchone() ou fetchall() après une requête WHERE
        result = conn.execute(
            f"SELECT * FROM {TABLE_NAME} WHERE product_id = ?", (product_id,)
        ).fetchone()
        if result:
            # Convertir le tuple résultat en dictionnaire (si nécessaire)
            # Obtenir les noms de colonnes pour mapper les valeurs
            columns = [desc[0] for desc in conn.description]
            product_dict = dict(zip(columns, result))
            print(f"Produit ID {product_id} trouvé.")
            return product_dict
        else:
            print(f"Produit ID {product_id} non trouvé.")
            return None
    except Exception as e:
        print(f"Erreur lors de la récupération du produit ID {product_id}: {e}")
        return None
