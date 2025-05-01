# 🚀 Améliorateur agentic de descriptions produits - The Bradery

![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Auteur: Louis Rigaux

Bonjour ! Ce projet propose une application Streamlit conçue pour aider The Bradery à **améliorer et standardiser les descriptions de produits** de manière efficace et cohérente, en tirant parti de l'Intelligence Artificielle.

L'objectif est de transformer des données produits brutes (issues d'exports CSV) en **descriptions marketing structurées, factuelles et optimisées** pour notre plateforme, tout en offrant des outils pour **harmoniser les visuels**.

---

## ✨ Pourquoi cet outil ?

*   **Efficacité :** Automatise la réécriture fastidieuse des descriptions en appliquant des règles de style définies.
*   **Cohérence :** Assure une structure et un ton uniformes sur l'ensemble des fiches produits traitées.
*   **Flexibilité :** Permet de traiter des lots de produits via upload CSV et de sélectionner précisément les éléments à modifier.
*   **Personnalisation :** Le "cerveau" de l'IA (le prompt) est facilement modifiable pour s'adapter à l'évolution de nos besoins marketing.
*   **Harmonisation Visuelle :** Offre des options pour nettoyer et standardiser les images produits (fond transparent, redimensionnement).
*   **Gratuit & Local :** Utilise des modèles d'IA accessibles gratuitement (via l'API Google AI) et stocke les données localement (DuckDB), sans coûts d'infrastructure externes majeurs pour cette version.

---

## 🛠️ Comment ça marche ?

L'application combine plusieurs technologies modernes et open-source :

*   **Interface Utilisateur :** Streamlit (pour une application web interactive simple)
*   **Orchestration IA :** LangChain & LangGraph (pour définir et exécuter le workflow de traitement)
*   **Modèle de Langage (LLM) :** Google AI API (Gemini/Gemma via `langchain-google-genai`) - *Nécessite une clé API gratuite.*
*   **Traitement de Données :** Pandas
*   **Traitement d'Images :** Pillow, Rembg (optionnel, pour la suppression de fond)
*   **Base de Données Locale :** DuckDB (pour stocker l'historique des traitements)
*   **Langage :** Python 3.12+

---

## 📂 Structure du projet

Le code est organisé de manière modulaire pour faciliter la maintenance et l'évolution :

```
technical_test_TBY/
│
├── .streamlit/
│ └── secrets.toml          # Stockage sécurisé des clés API (Google AI, HuggingFace...)
│
├── data/
│ ├── product_data.duckdb   # Base de données locale (créée automatiquement)
│ ├── processed_images/     # Dossier pour les images traitées
│ ├── prompt.txt            # Prompt par défaut pour l'IA (modifiable dans l'app)
│ └── prompt_v2.txt         # Prompt par amélioré
│
├── src/                    # Code source principal
│ ├── init.py
│ ├── config.py             # Paramètres généraux (modèle par défaut, chemins...)
│ ├── data_handler.py       # Chargement et préparation des données CSV
│ ├── db_handler.py         # Interaction avec la base DuckDB
│ ├── graph_workflow.py     # Définition du workflow LangGraph (les étapes IA)
│ ├── image_processor.py    # Logique de traitement d'image
│ ├── llm_models.py         # Initialisation du client LLM (Google AI)
│ └── utils.py              # Fonctions utilitaires (nettoyage HTML...)
│
├── .gitignore              # Fichier pour Git (ne pas commiter les fichiers secrets)
├── app.py                  # Script principal de l'application Streamlit
├── LOGO.png                # Logo de TheBradery
├── mermaid.md              # Diagramme de workflow
├── README.md               # Ce fichier
└── requirements.txt        # Dépendances Python
```

---

## ⚙️ Mise en Place (Installation & Configuration)

Pour lancer l'application sur votre poste :

1.  **Prérequis :** Assurez-vous d'avoir Python 3.12+ et `pip` installés.
2.  **Cloner le Dépôt :**
    ```bash
    git clone https://github.com/lrigaux/technical_test_TBY.git
    cd technical_test_TBY
    ```
3.  **Créer un Environnement Virtuel** (recommandé) :
    ```bash
    python -m venv ai_venv
    ```
4.  **Activer l'Environnement :**
    *   Windows : `.\ai_venv\Scripts\activate`
    *   macOS/Linux : `source ai_venv/bin/activate`
5.  **Installer les Dépendances :**
    ```bash
    pip install -r requirements.txt
    ```

6.  **Configurer les Clés API (Très Important) :**
    *   Obtenez une clé API gratuite depuis [Google AI Studio](https://aistudio.google.com/).
    *   Créez le dossier `.streamlit` à la racine du projet (s'il n'existe pas).
    *   Créez un fichier `secrets.toml` dans ce dossier.
    *   Ajoutez votre clé Google AI dans ce fichier :
        ```toml
        # .streamlit/secrets.toml
        GOOGLE_API_KEY = "VOTRE_CLE_API_GOOGLE_AI_ICI"
        # HUGGINGFACEHUB_API_TOKEN = "VOTRE_CLE_HF_SI_UTILISEE" # Optionnel
        ```
7.  **(Optionnel) Vérifier la Configuration :**
    *   Le fichier `src/config.py` contient le modèle Google AI par défaut et les options d'image.
    *   Le fichier `prompt.txt` contient le prompt par défaut utilisé par l'IA.

---

## ▶️ Comment Utiliser l'Application

1.  **Activer l'environnement virtuel** (si ce n'est pas déjà fait) : `source ai_venv/bin/activate` (ou équivalent Windows).
2.  **Lancer l'Application :** Depuis la racine du projet (`ai_product_enhancer/`), exécutez :
    ```bash
    streamlit run app.py
    ```
3.  **Accéder à l'application :** Ouvrez votre navigateur et accédez à l'URL indiquée dans la console (ex: http://localhost:8501)

---

## 🔀 Dans l'application

**Barre Latérale**: Configuration générale
    *   Choisissez le Modèle Google AI à utiliser.
    *   Visualisez et modifiez le Prompt si vous souhaitez ajuster le style de la description générée.
    *   Configurez les Options d'Image (activer/désactiver, suppression de fond, redimensionnement, etc.).
1. **Charger les descriptions**
    * Cliquez pour uploader un ou plusieurs fichiers CSV contenant les informations produits. Les colonnes attendues (même si les noms varient un peu) sont typiquement product_id, vendor, product_type, body_html, image_source.
2. **Prévisualiser et sélectionner les produits**
    * Un tableau affiche les données chargées avec une case à cocher "Traiter?".
    * Utilisez les cases ou les boutons "Tout Sélectionner"/"Tout Désélectionner".
    * Un aperçu de l'image source est affiché si une colonne d'URL est détectée.
3.  **Lancer le traitement** : 
    * Cliquez sur le bouton "✨ Lancer l'amélioration".
    * Une boîte de statut apparaîtra, montrant la progression à travers les différentes étapes du workflow (Mapping, Génération Titre, Amélioration, Description, Traitement Image, Sauvegarde).
4.  **Consulter les résultats** :
    * Une fois terminé, les résultats s'affichent produit par produit dans des sections dépliables (st.expander).
    * Vous y verrez : Titre, Description Originale (nettoyée), Description Améliorée, Image Originale, Image Traitée, et les erreurs éventuelles.
5.  **Exporter** : 
    * Téléchargez un fichier CSV contenant uniquement les produits traités et leurs nouvelles informations.
6.  **Historique** : 
    * Cochez la case pour voir l'ensemble des données traitées précédemment et stockées dans la base locale DuckDB.

---

## 🧠 Le workflow d'amélioration (LangGraph)
Sous le capot, un "graphe" LangGraph orchestre les étapes suivantes pour chaque produit sélectionné :
* map_selected : Identifie et standardise les colonnes du CSV (même si les noms varient).
* gen_titles : Si le produit n'a pas de titre, l'IA en génère un basé sur la description.
* enhance_desc : L'IA principale réécrit la description en suivant les règles du prompt configuré.
* proc_images : Si activé, télécharge et applique les transformations d'image demandées.
* aggregate_results : Rassemble toutes les informations (originales, générées, erreurs).
* persist_db : Sauvegarde les résultats finaux dans la base de données locale DuckDB.

---

## 🚀 Pistes d'amélioration possibles

* Mapping de Colonnes Assisté par IA : Pour gérer des formats CSV encore plus variés.
* Traitement d'Image Avancé : Ajout d'options (recadrage, ajout de fond blanc uniforme...).
* Gestion d'Erreurs Plus Fine : Permettre de relancer facilement les produits en erreur.
* Traitement par Lots (Batch Processing) : Pour une meilleure performance sur de très gros volumes.
* A/B Testing de Prompts : Intégrer une fonctionnalité pour comparer les résultats de différents prompts.
* Intégration Directe : Connecter l'outil à d'autres systèmes internes.
