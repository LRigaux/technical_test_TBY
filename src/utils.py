# src/utils.py

import html2text
import pandas as pd
from typing import Optional

def clean_html(raw_html: Optional[str]) -> str:
    """
    Nettoie une chaîne de caractères HTML pour en extraire le texte brut.

    Args:
        raw_html: La chaîne HTML d'entrée, ou None.

    Returns:
        Le texte nettoyé, ou une chaîne vide si l'entrée est invalide ou vide.
    """
    if not raw_html or pd.isna(raw_html):
        return ""

    try:
        h = html2text.HTML2Text()
        # Configuration pour ignorer les éléments non textuels et préserver les paragraphes
        h.ignore_links = True
        h.ignore_images = True
        h.ignore_emphasis = False # Garder le gras/italique si pertinent ? À ajuster.
        h.body_width = 0  # Désactive le retour à la ligne automatique basé sur la largeur
        h.unicode_snob = True # Pour une meilleure gestion de l'unicode
        h.escape_snob = True # Échapper les caractères spéciaux Markdown

        text = h.handle(str(raw_html))

        # Nettoyages supplémentaires pour enlever les excès d'espaces/sauts de ligne
        lines = [line.strip() for line in text.splitlines()]
        # Filtre les lignes vides et joint avec un seul saut de ligne
        cleaned_text = "\n".join(filter(None, lines))

        # Remplacer les multiples sauts de ligne consécutifs par un seul (optionnel)
        # import re
        # cleaned_text = re.sub(r'\n\s*\n', '\n', cleaned_text)

        return cleaned_text.strip()

    except Exception as e:
        print(f"Erreur lors du nettoyage HTML : {e}")
        # Retourner une chaîne vide ou une indication d'erreur selon le besoin
        return ""

# Potentiellement d'autres fonctions utilitaires ici...
# Par exemple, une fonction pour valider une URL :
import validators

def is_valid_url(url: Optional[str]) -> bool:
    """Vérifie si une chaîne est une URL valide."""
    if not url or pd.isna(url):
        return False
    return validators.url(url) == True