# PDF Loading Guide

This guide has been modified to use Ollama models for local inference.

## Prerequisites

- Ollama running locally with the required models:
  - `gpt-oss:120b-cloud` (LLM)
  - `embeddinggemma` (embeddings)
- Neo4j database instance
- Python environment with dependencies installed: `pip install -r requirements.txt`

## Setup Instructions

### 1. Environment Configuration

Create a `.env` file based on the `.env.example`:

```bash
cp .env.example .env
```

Edit the `.env` file with your specific configuration values:
- `NEO4J_URI`: Your Neo4j database URI
- `NEO4J_USERNAME`: Your Neo4j username (typically "neo4j")
- `NEO4J_PASSWORD`: Your Neo4j password
- `OPENAI_API_KEY`: Required only for the text2cypher example

### 2. Load Cypher Data

Use the CSV load script to populate the graph database with initial company and asset manager data:

```bash
./workshop-genai/financial-documents/csv-load/load.sh
```

This script will:
- Load the `.env` file from the project root
- Execute the `load.cypher` script to create the initial graph structure
- Verify the data load by counting nodes

### 3. Load PDF Data

Run the PDF loader notebook to process and load PDF documents into Neo4j GraphRAG:

```bash
jupyter notebook workshop-genai/financial-documents/pdf-build/01_PDF_Loader_for_Neo4j_GraphRAG.ipynb
```

Or open the notebook in your preferred IDE (VS Code, JupyterLab, etc.) and execute all cells.

This notebook will:
- Extract entities and relationships from PDF documents
- Create a knowledge graph in Neo4j
- Generate embeddings for semantic search
- Create a vector index on Chunk nodes

### 4. Run Vector Cypher Retriever

Execute the vector cypher retriever example to perform semantic search combined with graph traversal:

```bash
python workshop-genai/solutions/01_02_vector_cypher_retriever.py
```

This script demonstrates:
- Vector similarity search using embeddings
- Cypher queries to retrieve related entities and context
- Multiple retrieval patterns (risk factors, asset managers, shared risks)
- All using Ollama models for local inference

## Graph Visualization

For examples of graph schema visualization, see:

```bash
python workshop-genai/solutions/01_03_text2cypher_retriever.py
```

This script demonstrates:
- Text-to-Cypher query generation
- **Graph schema visualization** - creates an interactive HTML visualization (`my_schema.html`) showing all node types, relationship types, and properties
- Natural language to database queries

**Note:** This script uses OpenAI models (not Ollama) for text-to-Cypher generation.
