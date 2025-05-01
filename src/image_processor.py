# src/image_processor.py

import os
import requests
from PIL import Image, UnidentifiedImageError
from io import BytesIO
from typing import Dict, Any, Optional, Tuple, Union
import pandas as pd 

# Importer la config pour les chemins et options par défaut
from .config import IMAGE_DIR, DEFAULT_IMAGE_SIZE

# Gestion conditionnelle de rembg
try:
    from rembg import remove

    REM_BG_AVAILABLE = True
    print("Librairie 'rembg' trouvée et importée.")
except ImportError:
    REM_BG_AVAILABLE = False
    print(
        "Avertissement: Librairie 'rembg' non trouvée. La suppression de fond sera désactivée."
    )


def check_rembg_availability() -> bool:
    """Vérifie si la librairie rembg est installée."""
    return REM_BG_AVAILABLE


def process_image(
    image_url: Optional[str],
    product_id: Union[str, int],
    harmonise_options: Dict[str, Any]
) -> Optional[str]:
    """
    Télécharge, traite (optionnel: supprime fond, redimensionne) et sauvegarde une image.

    Args:
        image_url: L'URL de l'image source.
        product_id: L'identifiant du produit (utilisé pour nommer le fichier).
        harmonise_options: Dictionnaire contenant les options de traitement:
            - 'remove_bg': bool (si True et rembg dispo, supprime le fond)
            - 'resize': bool (si True, redimensionne)
            - 'max_size': tuple (largeur_max, hauteur_max) si resize est True

    Returns:
        Le chemin d'accès au fichier de l'image traitée si elle a été modifiée et sauvegardée,
        sinon None. Renvoie None aussi en cas d'erreur de téléchargement ou traitement.
    """
    if not image_url or pd.isna(image_url):
        print(f"[{product_id}] Pas d'URL d'image fournie.")
        return None

    processed_image_path: Optional[str] = None
    processed: bool = False  # Flag pour savoir si une modification a eu lieu

    try:
        print(f"[{product_id}] Téléchargement de l'image depuis: {image_url}")
        response = requests.get(image_url, timeout=20)  # Timeout augmenté
        response.raise_for_status()  # Lève une exception pour les codes HTTP 4xx/5xx
        img_data = response.content
        print(f"[{product_id}] Image téléchargée ({len(img_data)} bytes).")

        # Ouvrir l'image avec Pillow
        try:
            img = Image.open(BytesIO(img_data))
            original_format = img.format  # Garde le format original (JPEG, PNG, etc.)
            print(
                f"[{product_id}] Image ouverte avec Pillow. Format original: {original_format}, Mode: {img.mode}"
            )
        except UnidentifiedImageError:
            print(
                f"[{product_id}] Erreur: Impossible d'identifier le format de l'image depuis l'URL."
            )
            return None  # Ne peut pas traiter si non reconnu par Pillow

        # 1. Suppression du fond (si activé et possible)
        remove_bg_option = harmonise_options.get("remove_bg", False)
        if remove_bg_option and REM_BG_AVAILABLE:
            print(f"[{product_id}] Tentative de suppression du fond...")
            try:
                # rembg fonctionne mieux avec les bytes originaux
                output_bytes = remove(img_data)
                # Ré-ouvrir l'image depuis les bytes traités
                img = Image.open(BytesIO(output_bytes))
                # Si rembg réussit, l'image est maintenant RGBA (avec transparence)
                print(
                    f"[{product_id}] Suppression du fond réussie. Nouveau mode: {img.mode}"
                )
                processed = True
            except Exception as e:
                print(
                    f"[{product_id}] Avertissement: Erreur lors de la suppression du fond avec rembg: {e}. On continue sans."
                )
                # Revenir à l'image originale si rembg a échoué
                img = Image.open(BytesIO(img_data))
        elif remove_bg_option and not REM_BG_AVAILABLE:
            print(
                f"[{product_id}] Option 'remove_bg' activée mais rembg n'est pas disponible."
            )

        # 2. Redimensionnement (si activé)
        resize_option = harmonise_options.get("resize", False)
        if resize_option:
            max_size = harmonise_options.get("max_size", DEFAULT_IMAGE_SIZE)
            print(f"[{product_id}] Redimensionnement (max size: {max_size})...")
            try:
                img.thumbnail(max_size, Image.Resampling.LANCZOS)
                print(
                    f"[{product_id}] Redimensionnement réussi. Nouvelle taille: {img.size}"
                )
                processed = True
            except Exception as e:
                print(f"[{product_id}] Erreur lors du redimensionnement: {e}")

        # 3. Sauvegarde de l'image (UNIQUEMENT si elle a été traitée)
        if processed:
            # Déterminer le format de sauvegarde
            # Si le mode contient 'A' (alpha/transparence), sauvegarder en PNG
            if "A" in img.mode:
                save_format = "PNG"
            # Sinon, utiliser le format original si connu, sinon JPEG par défaut
            else:
                save_format = (
                    original_format
                    if original_format in ["JPEG", "PNG", "GIF"]
                    else "JPEG"
                )

            filename = f"{product_id}_processed.{save_format.lower()}"
            processed_image_path = os.path.join(IMAGE_DIR, filename)

            # Créer le dossier de destination s'il n'existe pas (redondant si fait au début, mais sûr)
            os.makedirs(IMAGE_DIR, exist_ok=True)

            try:
                # Gérer les options de sauvegarde spécifiques au format
                save_options = {}
                if save_format == "JPEG":
                    save_options["quality"] = 90  # Bonne qualité, taille raisonnable
                    # Si l'image venait d'un PNG transparent et qu'on sauve en JPEG, il faut la convertir
                    if img.mode == "RGBA":
                        print(
                            f"[{product_id}] Conversion RGBA -> RGB pour sauvegarde JPEG..."
                        )
                        img = img.convert("RGB")
                elif save_format == "PNG":
                    save_options["optimize"] = True

                print(
                    f"[{product_id}] Sauvegarde de l'image traitée vers: {processed_image_path} (Format: {save_format})"
                )
                img.save(processed_image_path, format=save_format, **save_options)
                print(f"[{product_id}] Sauvegarde réussie.")

            except Exception as e:
                print(
                    f"[{product_id}] Erreur lors de la sauvegarde de l'image traitée: {e}"
                )
                processed_image_path = None  # Échec de sauvegarde, retourne None
        else:
            print(
                f"[{product_id}] Aucune modification d'image effectuée, pas de sauvegarde."
            )

        return processed_image_path  # Retourne le chemin si sauvegardé, sinon None

    except requests.exceptions.RequestException as e:
        print(
            f"[{product_id}] Erreur réseau lors du téléchargement de l'image {image_url}: {e}"
        )
        return None
    except UnidentifiedImageError:  # Attrapé plus haut mais redondance ok
        print(f"[{product_id}] Erreur: Format d'image non identifié pour {image_url}.")
        return None
    except Exception as e:
        # Erreur générale pendant le processus
        print(
            f"[{product_id}] Erreur inattendue lors du traitement de l'image {image_url}: {e}"
        )
        return None
