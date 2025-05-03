# src/llm_models.py
# gère l'initialisation et la configuration des modèles LLM
from langchain_huggingface import HuggingFaceEndpoint
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
from typing import Optional
import os

# Initialisation du modèle LLM via HuggingFaceEndpoint
def initialize_llm(repo_id: str, api_key: str) -> Optional[HuggingFaceEndpoint]:
    """Initialise et retourne le modèle LLM via HuggingFaceEndpoint."""
    if not api_key:
        print("Erreur: Clé API Hugging Face non fournie.")
        return None
    try:
        llm = HuggingFaceEndpoint(
            repo_id=repo_id,
            huggingfacehub_api_token=api_key,
            task="text-generation",
            temperature=0.6,
            max_new_tokens=512,
        )
        print(f"LLM HuggingFaceEndpoint initialisé pour text-generation: {repo_id}")
        return llm
    except Exception as e:
        print(f"Erreur lors de l'initialisation du modèle LLM ({repo_id}): {e}")
        return None

# Initialisation du modèle LLM via Google Generative AI
def initialize_google_llm(model_name: str = "gemini-1.5-flash", api_key: Optional[str] = None) -> Optional[ChatGoogleGenerativeAI]:
    """Initialise et retourne un LLM Google Generative AI."""
    if not api_key:
        print("Erreur: Clé API Google non fournie.")
        return None
    try:
        st.secrets.get("HUGGINGFACEHUB_API_TOKEN")
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.6, # Ajuster si nécessaire
            # max_output_tokens=512, # Nom du paramètre peut varier, vérifier doc
            convert_system_message_to_human=True # Souvent utile pour les modèles Gemini/Gemma
        )
        print(f"LLM Google AI initialisé: {model_name}")
        return llm
    except Exception as e:
        print(f"Erreur lors de l'initialisation du modèle Google AI ({model_name}): {e}")
        return None