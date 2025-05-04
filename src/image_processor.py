# src/image_processor.py
# gère le téléchargement, la validation, le traitement et la sauvegarde des images

import os
import requests
from PIL import Image, UnidentifiedImageError, ImageOps # ImageOps pour padding potentiel
from io import BytesIO
from typing import Dict, Any, Optional, Tuple, Union
import pandas as pd
import validators # validation d'URL
from .config import IMAGE_DIR, DEFAULT_IMAGE_SIZE

# Gestion conditionnelle de rembg
try:
    from rembg import remove, new_session # Utiliser new_session pour potentiellement choisir le modèle
    REM_BG_AVAILABLE = True
    print("Librairie 'rembg' trouvée et importée.")
    # réer une session rembg par défaut
    # modèles possibles: "u2net", "u2netp", "u2net_human_seg", "silueta", etc.
    DEFAULT_REMBG_MODEL = "u2net"
    try:
        rembg_session = new_session(model_name=DEFAULT_REMBG_MODEL)
        print(f"Session rembg créée avec le modèle '{DEFAULT_REMBG_MODEL}'.")
    except Exception as rembg_init_e:
        print(f"Avertissement: Impossible de créer la session rembg avec le modèle '{DEFAULT_REMBG_MODEL}': {rembg_init_e}. Tentera sans session.")
        rembg_session = None # Fallback
except ImportError:
    REM_BG_AVAILABLE = False
    rembg_session = None
    print(
        "Avertissement: Librairie 'rembg' non trouvée. La suppression de fond sera désactivée."
    )


def check_rembg_availability() -> bool:
    """Vérifie si la librairie rembg est installée.
    
    Returns:
        True si rembg est installé, False sinon.
    """
    return REM_BG_AVAILABLE


def process_image(
    image_url: Optional[str],
    product_id: Union[str, int],
    harmonise_options: Dict[str, Any]
) -> Optional[str]:
    """
    Télécharge, valide, traite (optionnel: supprime fond, redimensionne) et sauvegarde une image.

    Args:
        image_url: L'URL de l'image source.
        product_id: L'identifiant du produit (utilisé pour nommer le fichier).
        harmonise_options: Dictionnaire contenant les options de traitement:
            - 'remove_bg': bool (si True et rembg dispo, supprime le fond)
            - 'resize': bool (si True, redimensionne)
            - 'max_size': tuple (largeur_max, hauteur_max) si resize est True
            - 'force_format': str | None (ex: 'PNG', 'JPEG', 'WEBP') pour forcer un format de sortie.
            - 'jpeg_quality': int (qualité pour sauvegarde JPEG, défaut 90)
            - 'overwrite': bool (si True, écrase fichier existant, défaut False)
            - 'rembg_model': str (nom du modèle rembg à utiliser, défaut 'u2net')

    Returns:
        Le chemin d'accès au fichier de l'image traitée si elle a été modifiée et sauvegardée,
        sinon None. Renvoie None aussi en cas d'erreur.
    """
    # --- 1. Validation Initiale ---
    if not image_url or pd.isna(image_url):
        print(f"[{product_id}] Pas d'URL d'image fournie.")
        return None
    if not validators.url(image_url):
        print(f"[{product_id}] URL fournie invalide: {image_url}")
        return None

    processed_image_path: Optional[str] = None
    processed: bool = False # flag pour savoir si une modification a eu lieu

    # --- 2. Téléchargement et Validation HTTP ---
    try:
        print(f"[{product_id}] Téléchargement de l'image depuis: {image_url}")
        # Timeouts séparés: connect=5s, read=15s
        response = requests.get(image_url, timeout=(5, 15), stream=True) # stream=True pour vérifier Content-Type
        response.raise_for_status() # Lève une exception pour les codes HTTP 4xx/5xx

        # Vérifier le Content-Type (optionnel mais recommandé)
        content_type = response.headers.get('Content-Type', '').lower()
        if not content_type.startswith('image/'):
            print(f"[{product_id}] Erreur: L'URL ne pointe pas vers une image (Content-Type: {content_type}).")
            response.close() # Fermer le stream
            return None

        # Lire les données de l'image
        img_data = response.content # Attention: charge tout en mémoire
        response.close() # Fermer le stream après lecture
        print(f"[{product_id}] Image téléchargée ({len(img_data)} bytes), Content-Type: {content_type}.")

    except requests.exceptions.Timeout:
        print(f"[{product_id}] Erreur: Timeout lors du téléchargement de {image_url}.")
        return None
    except requests.exceptions.HTTPError as e:
        print(f"[{product_id}] Erreur HTTP {e.response.status_code} lors du téléchargement: {e}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"[{product_id}] Erreur réseau lors du téléchargement de {image_url}: {e}")
        return None
    except Exception as e: # Autres erreurs potentielles
        print(f"[{product_id}] Erreur inattendue pendant le téléchargement: {e}")
        return None

    # --- 3. Traitement avec Pillow et Rembg ---
    try:
        try:
            img = Image.open(BytesIO(img_data))
            original_format = img.format
            original_mode = img.mode
            print(f"[{product_id}] Image ouverte. Format: {original_format}, Mode: {original_mode}, Taille: {img.size}")
        except UnidentifiedImageError:
            print(f"[{product_id}] Erreur Pillow: Impossible d'identifier le format de l'image.")
            return None

        # Convertir en RGBA pour la plupart des traitements (suppression fond, transparence)
        # Garde une copie originale au cas où la conversion échoue ou n'est pas nécessaire
        try:
            img_rgba = img.convert("RGBA")
            current_img = img_rgba # Travailler sur la version RGBA par défaut
            print(f"[{product_id}] Image convertie en RGBA pour traitement.")
        except Exception as e:
            print(f"[{product_id}] Avertissement: Impossible de convertir en RGBA ({e}). Utilisation du mode original {original_mode}.")
            current_img = img # Fallback sur l'image originale

        # 3.a Suppression du fond
        remove_bg_option = harmonise_options.get("remove_bg", False)
        if remove_bg_option and REM_BG_AVAILABLE:
            print(f"[{product_id}] Tentative de suppression du fond...")
            try:
                # Utiliser la session pré-créée ou le modèle spécifié
                active_rembg_session = rembg_session
                custom_model = harmonise_options.get('rembg_model')
                if custom_model and custom_model != DEFAULT_REMBG_MODEL:
                    try:
                        print(f"[{product_id}] Utilisation du modèle rembg personnalisé: {custom_model}")
                        active_rembg_session = new_session(model_name=custom_model)
                    except Exception as custom_model_e:
                        print(f"[{product_id}] Avertissement: Impossible de charger le modèle rembg '{custom_model}': {custom_model_e}. Utilisation du modèle par défaut.")
                        active_rembg_session = rembg_session # Revenir au défaut

                # Appliquer remove sur les données binaires originales
                output_bytes = remove(img_data, session=active_rembg_session)
                # Ré-ouvrir depuis les bytes traités pour obtenir l'image avec fond supprimé
                current_img = Image.open(BytesIO(output_bytes))
                print(f"[{product_id}] Suppression du fond réussie. Nouveau mode: {current_img.mode}")
                processed = True
            except Exception as e:
                print(f"[{product_id}] Avertissement: Erreur lors de la suppression du fond: {e}. On continue sans.")
                # Pas besoin de recharger l'image ici, current_img est toujours l'image convertie (ou originale)
        elif remove_bg_option and not REM_BG_AVAILABLE:
            print(f"[{product_id}] Option 'remove_bg' activée mais rembg n'est pas disponible.")

        # 3.b Redimensionnement
        resize_option = harmonise_options.get("resize", False)
        if resize_option:
            max_size = harmonise_options.get("max_size", DEFAULT_IMAGE_SIZE)
            print(f"[{product_id}] Redimensionnement (max size: {max_size})...")
            try:
                # Utiliser thumbnail pour garder les proportions
                current_img.thumbnail(max_size, Image.Resampling.LANCZOS)
                print(f"[{product_id}] Redimensionnement réussi. Nouvelle taille: {current_img.size}")
                processed = True
                # Optionnel: Ajouter un padding pour atteindre exactement max_size?
                # Exemple: add_padding(current_img, max_size, background_color=(255,255,255,0)) # Transp.
            except Exception as e:
                 print(f"[{product_id}] Erreur lors du redimensionnement: {e}")

        # --- 4. Sauvegarde ---
        if processed:
            # Déterminer le format de sauvegarde final
            force_format = harmonise_options.get('force_format')
            if force_format and force_format.upper() in ['PNG', 'JPEG', 'WEBP']:
                save_format = force_format.upper()
                print(f"[{product_id}] Format de sortie forcé: {save_format}")
            elif 'A' in current_img.mode: # Si transparence (même si pas de remove_bg mais conversion RGBA)
                save_format = 'PNG'
            else: # Pas de transparence, utiliser original ou JPEG
                save_format = original_format if original_format in ['JPEG', 'PNG', 'GIF', 'WEBP'] else 'JPEG'

            filename = f"{product_id}_processed.{save_format.lower()}"
            processed_image_path = os.path.join(IMAGE_DIR, filename)

            # Vérifier si on doit écraser
            overwrite = harmonise_options.get('overwrite', False)
            if not overwrite and os.path.exists(processed_image_path):
                print(f"[{product_id}] Fichier traité existant trouvé et overwrite=False. Pas de sauvegarde.")
                # Retourner le chemin existant car l'image traitée existe déjà
                return processed_image_path

            # Préparer l'image pour la sauvegarde (gestion des modes)
            img_to_save = current_img
            if save_format == 'JPEG':
                jpeg_quality = harmonise_options.get('jpeg_quality', 90)
                save_options = {'quality': jpeg_quality}
                # JPEG ne supporte pas la transparence, convertir en RGB
                if img_to_save.mode == 'RGBA' or img_to_save.mode == 'LA':
                    print(f"[{product_id}] Conversion {img_to_save.mode} -> RGB pour sauvegarde JPEG...")
                    # Créer un fond blanc (ou autre couleur) avant de convertir
                    background = Image.new("RGB", img_to_save.size, (255, 255, 255))
                    background.paste(img_to_save, mask=img_to_save.split()[-1]) # Utiliser le canal alpha comme masque
                    img_to_save = background
                elif img_to_save.mode == 'P': # Palette-based, convertir aussi
                     print(f"[{product_id}] Conversion P -> RGB pour sauvegarde JPEG...")
                     img_to_save = img_to_save.convert('RGB')

            elif save_format == 'PNG':
                save_options = {'optimize': True}
                # S'assurer que le mode est compatible (RGBA ou RGB sont ok, L aussi)
                if img_to_save.mode == 'P':
                    print(f"[{product_id}] Conversion P -> RGBA pour sauvegarde PNG...")
                    img_to_save = img_to_save.convert('RGBA') # Convertir en RGBA pour garder potentielle transp.
            elif save_format == 'WEBP':
                 save_options = {'quality': harmonise_options.get('webp_quality', 85)} # Qualité WebP
                 # WebP supporte RGBA et RGB
                 if img_to_save.mode == 'P':
                     print(f"[{product_id}] Conversion P -> RGBA pour sauvegarde WEBP...")
                     img_to_save = img_to_save.convert('RGBA')
            else: # Autres formats (GIF?) - pas d'options spécifiques
                save_options = {}

            # Sauvegarder
            try:
                os.makedirs(IMAGE_DIR, exist_ok=True)
                print(f"[{product_id}] Sauvegarde vers: {processed_image_path} (Format: {save_format}, Mode: {img_to_save.mode})")
                img_to_save.save(processed_image_path, format=save_format, **save_options)
                print(f"[{product_id}] Sauvegarde réussie.")
            except Exception as e:
                print(f"[{product_id}] Erreur lors de la sauvegarde de l'image: {e}")
                processed_image_path = None # Échec
        else:
             print(f"[{product_id}] Aucune modification d'image effectuée, pas de sauvegarde.")
             # Si aucune modif, on ne retourne pas de chemin traité
             processed_image_path = None

        return processed_image_path

    except Exception as e:
        # Erreur générale pendant le traitement Pillow/Rembg
        print(f"[{product_id}] Erreur inattendue lors du traitement de l'image: {e}")
        return None

# --- Fonction utilitaire pour padding (optionnel) ---
def add_padding(img: Image.Image, target_size: Tuple[int, int], background_color: Tuple[int, int, int, int]) -> Image.Image:
    """
    Ajoute du padding à une image pour atteindre une taille cible.
    
    Args:
        img: L'image à paddinger.
        target_size: La taille cible (largeur, hauteur).
        background_color: La couleur de fond (R, G, B, A).

    Returns:
        L'image paddingée.
    """
    original_width, original_height = img.size
    target_width, target_height = target_size

    # Calculer le padding nécessaire
    padding_left = (target_width - original_width) // 2
    padding_top = (target_height - original_height) // 2
    padding_right = target_width - original_width - padding_left
    padding_bottom = target_height - original_height - padding_top
    padding = (padding_left, padding_top, padding_right, padding_bottom)

    # Appliquer le padding
    return ImageOps.expand(img, border=padding, fill=background_color)