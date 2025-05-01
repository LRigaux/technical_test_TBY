# src/config.py
import os

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_FILE = os.path.join(DATA_DIR, "product_data.duckdb")
IMAGE_DIR = os.path.join(DATA_DIR, "processed_images")
# PROMPT_FILE = os.path.join(DATA_DIR, "prompt.txt")
PROMPT_FILE = os.path.join(DATA_DIR, "prompt_v2.txt")


# --- LLM ---
# Modèle par défaut (peut être surchargé dans l'UI)
# DEFAULT_MODEL_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.1"
# DEFAULT_MODEL_REPO_ID = "google/flan-t5-large"
# DEFAULT_MODEL_REPO_ID = "Qwen/Qwen2.5-Coder-32B-Instruct"
DEFAULT_MODEL_REPO_ID = "tiiuae/falcon-7b-instruct"

# --- Image Processing ---
DEFAULT_IMAGE_SIZE = (800, 800)
REMOVE_BG_DEFAULT = True
RESIZE_IMAGE_DEFAULT = True

# --- Data Handling ---
# Mappage possible des noms de colonnes CSV vers les clés standard
COLUMN_MAPPING = {
    "standard_keys": [
        "product_id",
        "vendor",
        "product_type",
        "body_html",
        "image_source",
        "title",
    ],
    "possible_names": {
        "product_id": ["product_id", "id", "sku"],
        "vendor": ["vendor", "brand", "marque"],
        "product_type": ["product_type", "type", "category"],
        "body_html": [
            "body_html",
            "description",
            "desc",
            "html_description",
            "content",
        ],
        "image_source": ["image_source", "image_url", "image", "picture"],
        "title": ["title", "name", "product_name", "nom"],
    },
}

# Clés essentielles attendues après mapping
REQUIRED_KEYS = ["product_id", "body_html"]  # Minimum requis pour traitement
