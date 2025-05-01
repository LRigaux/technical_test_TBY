# RAG Application for The Bradery Product Descriptions

This project implements a Streamlit application that uses a Retrieval-Augmented Generation (RAG) pipeline to generate optimized product descriptions for The Bradery.

## Project Structure

```
rag_thebradery/
├── .env                  # (Optional) For storing API keys locally if not using secrets.toml
├── .gitignore            # Specifies intentionally untracked files that Git should ignore
├── .streamlit/
│   └── secrets.toml      # Stores secrets like API keys for Streamlit Cloud or local use
├── data/
│   └── descriptions.csv  # Input product data
├── prompts/
│   └── prompt.txt        # Base prompt for description generation
├── chroma_db/            # Local vector database storage for ChromaDB
├── descriptions.duckdb   # Local SQL database for product data and generated descriptions
├── app.py                # Main Streamlit application file (UI)
├── rag_pipeline.py       # Core RAG pipeline logic (LangChain)
├── embedding.py          # Handles text embedding using sentence-transformers
├── utils.py              # Utility functions (data loading, cleaning, DB interaction)
├── requirements.txt      # Project dependencies
└── README.md             # This file
```

## Features

- **Load Data:** Ingests product descriptions from a CSV file.
- **Vector Storage:** Uses ChromaDB for local vector storage of product descriptions.
- **Embeddings:** Utilizes `sentence-transformers/all-MiniLM-L6-v2` for generating text embeddings locally.
- **RAG Pipeline:** Leverages LangChain to implement the RAG pipeline, combining retrieval from ChromaDB with generation using a Hugging Face model.
- **LLM Integration:** Uses the `mistralai/Mistral-7B-Instruct-v0.2` model via the free Hugging Face Inference API for text generation.
- **Data Persistence:** Stores original product data and generated descriptions in a local DuckDB database.
- **Streamlit UI:** Provides an intuitive web interface for:
    - Selecting products.
    - Viewing original details.
    - Editing the generation prompt.
    - Generating new descriptions.
    - Viewing generated descriptions.
    - Exporting results to CSV.
- **History Tracking:** Keeps a record of generated descriptions in DuckDB.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/LRigaux/technical_test_TBY.git
    cd technical_test_tby
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv ai_venv
    # On Windows
    .\ai_venv\Scripts\activate
    # On macOS/Linux
    source ai_venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up API Key:**
    - Create a file named `secrets.toml` inside a `.streamlit` directory (`technical_test_tby/.streamlit/secrets.toml`).
    - Add your Hugging Face API token to it:
      ```toml
      HF_API_TOKEN="hf_YOUR_HUGGINGFACE_TOKEN"
      ```
    - *Alternatively*, you can create a `.env` file in the root directory and add `HF_API_TOKEN=hf_YOUR_HUGGINGFACE_TOKEN` if you prefer using `python-dotenv` directly (though `secrets.toml` is standard for Streamlit).

5.  **Place Data:**
    - Ensure your `descriptions.csv` file is in the `data/` directory.
    - Ensure your `prompt.txt` file is in the `prompts/` directory.

## Usage

1.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

2.  **Interact with the UI:**
    - Select a product from the dropdown.
    - View the original description and image.
    - (Optional) Modify the prompt in the text area.
    - Click "Generate New Description" to run the RAG pipeline.
    - View the generated description.
    - Click "Export Generated Descriptions to CSV" to save the latest generated descriptions for all products.

## Technical Details

- **Database:** DuckDB is used for its speed and ease of use as an embedded SQL database.
- **Vector Store:** ChromaDB provides efficient local vector storage and retrieval.
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` is chosen for its balance of performance and quality for local execution.
- **LLM:** `mistralai/Mistral-7B-Instruct-v0.2` is used via the Hugging Face Inference API, offering powerful generation capabilities without requiring local GPU resources.
- **Frameworks:** Streamlit for the UI and LangChain for orchestrating the RAG pipeline.

## Potential Improvements

- Implement asynchronous operations for faster UI responsiveness during generation.
- Add more robust error handling and logging.
- Integrate LangGraph for more complex agentic workflows.
- Add functionality for batch processing of descriptions.
- Incorporate image harmonization features.
- Allow selection of different embedding or LLM models via the UI.
