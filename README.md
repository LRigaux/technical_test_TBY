# 🚀 Améliorateur agentic de descriptions produits - The Bradery

![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Auteur: Louis Rigaux

Ce projet propose une application Streamlit conçue pour aider The Bradery à **améliorer et standardiser les descriptions de produits** de manière efficace et cohérente, en tirant parti de l'IA Générative (via l'API gratuite de Google AI).

L'objectif est de transformer des données produits brutes (issues d'exports CSV) en **descriptions marketing structurées, factuelles et optimisées** pour notre plateforme, tout en offrant des outils pour **analyser la qualité des données** et **harmoniser les visuels**.

---

## ✨ Pourquoi cet outil ? Bénéfices clés

*   **Analyse de données intégrée:** fournit un aperçu statistique des fichiers CSV chargés (taux de remplissage, types de données) pour identifier rapidement les problèmes potentiels.
*   **Nettoyage interactif:** permet de retirer facilement les colonnes jugées inutiles (ex: trop de valeurs manquantes) ou les lignes sans description avant le traitement IA.
*   **Efficacité IA:** automatise la réécriture des descriptions en appliquant des règles de style définies, en se concentrant sur les données pertinentes.
*   **Cohérence:** assure une structure et un ton uniformes sur l'ensemble des fiches produits traitées.
*   **Flexibilité & personnalisation:** traite des lots via upload CSV, permet une sélection fine des produits, et le prompt IA est modifiable pour s'adapter aux besoins marketing.
*   **Harmonisation visuelle:** offre des options pour nettoyer (fond transparent) et standardiser (redimensionnement) les images produits.
*   **Gratuit & local:** utilise l'API gratuite de Google AI et stocke les données localement (DuckDB), limitant les coûts.

---

## 🛠️ Comment ça marche ?

L'application combine plusieurs technologies modernes et open-source :

*   **Interface Utilisateur:** Streamlit (pour une application web interactive simple)
*   **Analyse de données & visualisation:** Pandas, Plotly Express
*   **Orchestration IA:** LangChain & LangGraph (pour définir et exécuter le workflow de traitement)
*   **Modèle de langage (LLM):** Google AI API (Gemini/Gemma via `langchain-google-genai`) - *Nécessite une clé API gratuite.*
*   **Traitement de données:** Pandas
*   **Traitement d'images:** Pillow, Rembg (optionnel, pour la suppression de fond)
*   **Base de données locale:** DuckDB (pour stocker l'historique des traitements)
*   **Langage:** Python 3.12+

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

## ⚙️ Mise en place (installation & configuration)

Pour lancer l'application sur votre poste :

1.  **Prérequis:** Assurez-vous d'avoir Python 3.12+ et `pip` installés.
2.  **Cloner le dépôt:**
    ```bash
    git clone https://github.com/lrigaux/technical_test_TBY.git
    cd technical_test_TBY
    ```
3.  **Créer un environnement virtuel** (recommandé):
    ```bash
    python -m venv ai_venv
    ```
4.  **Activer l'environnement:**
    *   Windows : `.\ai_venv\Scripts\activate`
    *   macOS/Linux : `source ai_venv/bin/activate`
5.  **Installer les dépendances:**
    ```bash
    pip install -r requirements.txt
    ```

6.  **Configurer les clés API:**
    *   Obtenez une clé API gratuite depuis [Google AI Studio](https://aistudio.google.com/).
    *   Créez le dossier `.streamlit` à la racine du projet (s'il n'existe pas).
    *   Créez un fichier `secrets.toml` dans ce dossier.
    *   Ajoutez votre clé Google AI dans ce fichier :
        ```toml
        # .streamlit/secrets.toml
        GOOGLE_API_KEY = "VOTRE_CLE_API_GOOGLE_AI_ICI"
        # HUGGINGFACEHUB_API_TOKEN = "VOTRE_CLE_HF_SI_UTILISEE" # Optionnel
        ```
7.  **(Optionnel) Vérifier la configuration:**
    *   Le fichier `src/config.py` contient le modèle Google AI par défaut et les options d'image.
    *   Le fichier `prompt.txt` contient le prompt par défaut utilisé par l'IA.

---

## ▶️ Comment utiliser l'application

1.  **Activer l'environnement virtuel** (si ce n'est pas déjà fait) : `source ai_venv/bin/activate` (ou équivalent Windows).
2.  **Lancer l'application :** Depuis la racine du projet (`ai_product_enhancer/`), exécutez :
    ```bash
    streamlit run app.py
    ```
3.  **Accéder à l'application :** Ouvrez votre navigateur et accédez à l'URL indiquée dans la console (ex: http://localhost:8501)

---

## 🔀 Flux de travail dans l'application

**(A) Configuration (barre latérale)**
*   Choisissez le **Modèle Google AI**.
*   Visualisez/Modifiez le **Prompt** pour la génération de texte.
*   Configurez les **Options d'image** (harmonisation, format, etc.).

**(B) Traitement principal**

1.  **Charger les données (section 1)**
    *   Uploadez un ou plusieurs fichiers CSV. L'application les combine automatiquement.

2.  **Analyser et préparer (section 1.5)**
    *   Visualisez les **statistiques clés** (lignes, colonnes, valeurs manquantes via graphique).
    *   **Optionnel:** Choisissez de **retirer les colonnes** jugées peu utiles (ex: >95% vides).
    *   **Optionnel:** Choisissez de **retirer les lignes** où la description produit est manquante.
    *   Cliquez sur **"Appliquer le Nettoyage"** pour valider vos choix. Le tableau de prévisualisation sera mis à jour.

3.  **Prévisualiser et sélectionner (section 2)**
    *   Le tableau affiche les données (potentiellement nettoyées).
    *   Utilisez les **cases à cocher** ou les boutons "Tout sélectionner/désélectionner" pour choisir les produits à traiter par l'IA.

4.  **Lancer le traitement IA (section 3)**
    *   Cliquez sur **"✨ Lancer l'amélioration"**.
    *   Suivez la **progression** des différentes étapes (mapping, génération titre, description, image, sauvegarde) dans la boîte de statut.

5.  **Consulter les résultats (section 4)**
    *   Les résultats s'affichent produit par produit (titre, description originale nettoyée, description améliorée, images, erreurs éventuelles).

6.  **Exporter (section 5)**
    *   Téléchargez un fichier CSV contenant les données des produits traités.

7.  **Historique (section 📚)**
    *   Consultez l'historique complet des traitements stockés localement.

---

## 🧠 Le workflow d'amélioration (LangGraph)
Sous le capot, un "graphe" LangGraph orchestre les étapes suivantes pour chaque produit sélectionné :
* map_selected : identifie et standardise les colonnes du CSV (même si les noms varient).
* gen_titles : si le produit n'a pas de titre, l'IA en génère un basé sur la description.
* enhance_desc : l'IA principale réécrit la description en suivant les règles du prompt configuré.
* proc_images : si activé, télécharge et applique les transformations d'image demandées.
* aggregate_results : rassemble toutes les informations (originales, générées, erreurs).
* persist_db : sauvegarde les résultats finaux dans la base de données locale DuckDB.

---

## Pistes d'amélioration possibles

*   **Mapping de colonnes assisté par IA.**
*   **Traitement d'image avancé** (recadrage, fond blanc...).
*   **Gestion d'erreurs plus fine** (relance facile).
*   **Traitement par lots** (performance gros volumes).
*   **A/B Testing de prompts.**
*   **Intégration directe** (API PIM...).
