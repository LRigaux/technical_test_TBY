# src/utils.py
# contient des fonctions utilitaires
import html2text
import pandas as pd
from typing import Optional
import io
import validators
from typing import List, Dict, Any, Optional

def analyze_dataframe(df: pd.DataFrame, missing_threshold: float = 95.0) -> Dict[str, Any]:
    """
    Analyse un DataFrame pour les statistiques de base et les problèmes potentiels.

    Args:
        df: Le DataFrame à analyser.
        missing_threshold: Pourcentage de valeurs manquantes au-delà duquel
                           une colonne est suggérée pour suppression.

    Returns:
        Un dictionnaire contenant les résultats de l'analyse:
        - 'shape': Tuple (lignes, colonnes)
        - 'total_rows': Nombre total de lignes
        - 'total_cols': Nombre total de colonnes
        - 'df_info': Informations df.info() sous forme de chaîne
        - 'missing_stats': DataFrame des valeurs manquantes (count, %)
        - 'cols_to_suggest_dropping': Liste des colonnes avec > threshold% de manquants
        - 'rows_with_missing_desc_indices': Index des lignes où 'body_html' est manquant
        - 'rows_with_missing_desc_count': Nombre de ces lignes
    """
    if df is None or df.empty:
        return {"error": "DataFrame vide ou non fourni."}

    analysis = {}
    analysis['shape'] = df.shape
    analysis['total_rows'] = df.shape[0]
    analysis['total_cols'] = df.shape[1]

    # Capturer df.info()
    buf = io.StringIO()
    df.info(buf=buf)
    analysis['df_info'] = buf.getvalue()

    # Statistiques sur les valeurs manquantes
    missing_counts = df.isnull().sum()
    missing_percentages = (missing_counts / analysis['total_rows']) * 100
    missing_stats_df = pd.DataFrame({
        'Manquants (Nb)': missing_counts,
        'Manquants (%)': missing_percentages.round(2) # Arrondi pour affichage
    })
    # Trier par pourcentage décroissant
    analysis['missing_stats'] = missing_stats_df[missing_stats_df['Manquants (Nb)'] > 0].sort_values(by='Manquants (%)', ascending=False)

    # Identifier les colonnes potentiellement à supprimer
    analysis['cols_to_suggest_dropping'] = analysis['missing_stats'][analysis['missing_stats']['Manquants (%)'] >= missing_threshold].index.tolist()

    # Identifier les lignes avec description manquante (en supposant que 'body_html' est la clé après mapping)
    # Utiliser la colonne mappée si elle existe, sinon la colonne brute si le mapping n'a pas eu lieu
    desc_col = 'body_html' if 'body_html' in df.columns else None
    # Essayer de trouver une colonne de description brute si le mapping n'a pas encore eu lieu
    if not desc_col:
         potential_desc_cols = ['body_html', 'description', 'desc', 'html_description', 'content']
         desc_col = next((col for col in potential_desc_cols if col in df.columns), None)

    analysis['rows_with_missing_desc_indices'] = []
    analysis['rows_with_missing_desc_count'] = 0
    if desc_col:
        missing_desc_mask = df[desc_col].isnull() | (df[desc_col].astype(str).str.strip() == '')
        analysis['rows_with_missing_desc_indices'] = df[missing_desc_mask].index.tolist()
        analysis['rows_with_missing_desc_count'] = len(analysis['rows_with_missing_desc_indices'])
    else:
        print("Avertissement: Colonne de description non trouvée pour l'analyse des lignes vides.")


    return analysis
    

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
        h.ignore_emphasis = False  # Garder le gras/italique si pertinent ? À ajuster.
        h.body_width = (
            0  # Désactive le retour à la ligne automatique basé sur la largeur
        )
        h.unicode_snob = True  # Pour une meilleure gestion de l'unicode
        h.escape_snob = True  # Échapper les caractères spéciaux Markdown

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


def is_valid_url(url: Optional[str]) -> bool:
    """
    Vérifie si une chaîne est une URL valide.

    Args:
        url: La chaîne URL d'entrée, ou None.

    Returns:
        True si l'URL est valide, False sinon.
    """
    if not url or pd.isna(url):
        return False
    return validators.url(url) == True
