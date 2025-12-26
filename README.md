# PlantXBot - AI-Powered Genomics Chatbot Platform

**PlantXBot** is a flexible, AI-driven conversational interface designed to query specialized databases using natural language. It combines **Large Language Models (LLMs)** with **SQL generation** and **Retrieval-Augmented Generation (RAG)** to provide accurate, data-backed answers for scientific research.

Originally designed for plant genomics (tRFs, peptides, fusion transcripts), this codebase has been generalized to serve as a robust template for any domain-specific database chatbot.

##  Key Features

*   **Natural Language to SQL:** Automatically converts user questions (e.g., *"Show me peptides with high stability"*) into optimized SQL queries.
*   **Dual-Layer Intelligence:**
    *   **Intent Classification:** Distinguishes between data lookup, metadata questions, and general conversation.
    *   **Orchestration:** Uses LangChain to manage context and conversation history.
*   **RAG Capabilities:** Integrates with **ChromaDB** to answer unstructured questions using embedded knowledge bases (e.g., research papers or documentation).
*   **Dynamic UI:** A React-based chat interface embedded in PHP, featuring:
    *   Streaming responses.
    *   Dynamic data grid previews.
    *   CSV download generation.
*   **High Performance:** Built on Groq's inference engine (Llama-3/Mixtral) for sub-second latency.

##  Repository Structure

| File | Description |
| :--- | :--- |
| `demo_app.py` | The Flask API server that acts as the backend dispatcher. |
| `demo_bot.py` | The core AI logic (LangChain agents, SQL generation, Prompt Templates). |
| `setup.sh` | Automated script to create the Conda environment and install dependencies. |
| `requirements.yaml` | Detailed dependency definitions for Conda/Pip. |
| `chat.php` | The main Chatbot UI (React + PHP). |
| `about.php` | Project information page. |
| `team.php` | Team/Credits page. |
| `all_database_2.db` | The SQLite database file (Main data source). |

## 🛠️ Prerequisites

Before installing, ensure you have the following:

1.  **Operating System:** Linux (Ubuntu 20.04+ recommended), macOS, or Windows (via WSL2).
2.  **Package Manager:** [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
3.  **Git:** Installed and configured.
4.  **API Key:** A free **Groq API Key** (Get one at [console.groq.com](https://console.groq.com)).

## Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/skbinfo/PlantXBot.git
cd PlantXBot


