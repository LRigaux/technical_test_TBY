# src/llm_models.py
from langchain_huggingface import HuggingFaceEndpoint # Nouvelle version
import streamlit as st # Pour accéder aux secrets

def initialize_llm(repo_id, api_key):
    """Initialise et retourne le modèle LLM via HuggingFaceHub."""
    if not api_key:
        print("Erreur: Clé API Hugging Face non fournie.")
        return None
    try:
        llm = HuggingFaceEndpoint(
            repo_id=repo_id,
            huggingfacehub_api_token=api_key,
            temperature=0.6,          
            max_new_tokens=512,       # préféré à max_length pour beaucoup de modèles
        )
        print(f"LLM initialisé: {repo_id}")
        return llm
    except Exception as e:
        print(f"Erreur lors de l'initialisation du modèle LLM ({repo_id}): {e}")
        return None
