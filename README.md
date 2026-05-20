# SeSMap

SeSMap is a visual analytics system for exploring semantic maps built from scientific papers. It organizes fine-grained semantic units (MSUs) into subspaces, lets users inspect semantic relationships across papers, and supports LLM-assisted search, summarization, and interaction.

The system is designed around two linked views:

- A semantic map view for browsing subspaces, hexagonal semantic units, routes, and selected MSUs.
- A source gallery and stepwise analysis view for reviewing papers, saved paths, selected MSUs, and LLM summaries.

## Main Features

- Multi-case semantic maps for different research domains.
- Interactive subspace visibility control through natural language commands.
- Source Gallery for displaying paper thumbnails by topic, such as air pollution and scramjet combustion.
- Stepwise Analysis View for saving selected routes and summarizing checked MSUs.
- RAG-based paper question answering over project PDFs.
- Centralized LLM configuration for chat, intent parsing, RAG, MSU summary, context compression, and embeddings.

## Project Structure

```text
SeSMap/
  SeSMap-backend/      Flask backend, APIs, RAG, prompts, data, PDF indexes
  SeSMap-frontend/     Vue + Vite frontend, semantic map UI, gallery, analysis panels
```

Important backend files:

```text
SeSMap-backend/app.py                    Main Flask app and API routes
SeSMap-backend/services/llm_config.py    Central LLM configuration
SeSMap-backend/rag.py                    PDF RAG helper
SeSMap-backend/prompts.py                Prompt definitions
SeSMap-backend/data/                     Semantic map data, PDFs, FAISS indexes
```

Important frontend files:

```text
SeSMap-frontend/src/components/MainView.vue   Semantic map container
SeSMap-frontend/src/components/LeftPane.vue   Chat, source gallery, controls
SeSMap-frontend/src/components/RightPane.vue  Stepwise analysis view
SeSMap-frontend/src/components/LinkCard.vue   MSU selection and summary cards
SeSMap-frontend/src/lib/semanticMap.js        Main semantic map rendering logic
SeSMap-frontend/src/lib/api.js                Frontend API helpers
```

## Quick Start

### 1. Start the Backend

```bash
cd SeSMap-backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

The backend runs on:

```text
http://127.0.0.1:5000
```

### 2. Start the Frontend

```bash
cd SeSMap-frontend
npm install
npm run dev
```

The frontend usually runs on:

```text
http://localhost:5173
```

## LLM Configuration

LLM settings live in:

```text
SeSMap-backend/.env
```

The project uses task-based model roles:

```env
LLM_DEFAULT_MODEL=gpt-4o
LLM_CHAT_MODEL=gpt-4o
LLM_INTENT_MODEL=gpt-4o
LLM_RAG_MODEL=gpt-4o
LLM_SUMMARY_MODEL=gpt-4o
LLM_CONDENSE_MODEL=gpt-4o
LLM_EMBEDDING_MODEL=text-embedding-3-small
```

These roles are resolved by:

```text
SeSMap-backend/services/llm_config.py
```

Use the backend config endpoint to check the active model settings:

```text
GET /api/llm/config
```

Do not commit real API keys to a public repository.

## Common Commands

### Source Gallery

```text
show air related papers in gallery
show combust related papers in gallery
Scramjet Combustion
clear gallery
```

Air-related commands show air pollution papers. Scramjet or combustion-related commands show the case1 combustion papers.

### Subspace Control

```text
show background
show method and result
show all subspaces
hide all subspaces
list subspaces
subspace count
```

### RAG and Paper Questions

```text
list projects
build index for case1
ask case2: compare the main methods across papers
```

## MSU Summary Workflow

In the Stepwise Analysis View:

1. Save or select a semantic route.
2. Open a LinkCard.
3. Check the MSUs you want to summarize.
4. Click `Summarize`.

The selected MSUs are grouped by HSU and ordered by route path. The backend uses the `LLM_SUMMARY_MODEL` role to generate a compact `RouteSummary`.

## Data

The project currently includes multiple cases under:

```text
SeSMap-backend/data/
SeSMap-backend/case1/
SeSMap-backend/case2/
SeSMap-backend/case3/
```

PDFs and gallery assets are mirrored in the frontend for presentation:

```text
SeSMap-frontend/src/assets/pdf/
SeSMap-frontend/src/assets/pictures/
```

## Notes

- If you change the embedding model, rebuild the FAISS indexes.
- If the LLM provider changes, update `.env` rather than editing business code.
- The Source Gallery is mostly handled on the frontend through local keyword matching.
- RAG indexes are stored locally under `SeSMap-backend/data/indexes/`.
