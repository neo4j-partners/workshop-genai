#!/usr/bin/env python3
"""
PDF Loader for Neo4j GraphRAG

This script loads PDF files into a Neo4j knowledge graph using the GraphRAG pipeline.
It extracts entities and relationships from company filings and creates a vector index
for semantic search.
"""

import argparse
import asyncio
import csv
import glob
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j_graphrag.embeddings import OllamaEmbeddings
from neo4j_graphrag.experimental.components.schema import GraphSchema
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.generation.prompts import ERExtractionTemplate
from neo4j_graphrag.indexes import create_vector_index
from neo4j_graphrag.llm import OllamaLLM


def load_approved_companies(csv_path):
    """
    Load approved company names from CSV file.

    Args:
        csv_path: Path to CSV file containing company names

    Returns:
        Set of approved company names (uppercase)
    """
    approved_companies = set()

    if not os.path.exists(csv_path):
        print(f"Warning: Company CSV file not found at {csv_path}")
        return approved_companies

    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            name = row.get('name')
            if name:
                approved_companies.add(name.strip().upper())

    print(f"Loaded {len(approved_companies)} approved companies")
    return approved_companies


def create_prompt_template(approved_companies):
    """
    Create custom prompt template with approved company list.

    Args:
        approved_companies: Set of approved company names

    Returns:
        ERExtractionTemplate instance
    """
    joined_names = '\n'.join(f'- {name}' for name in approved_companies)

    company_instruction = (
        "Extract only information about the following companies. "
        "If a company is mentioned but is not in this list, ignore it. "
        "When extracting, the company name must match exactly as shown below. "
        "Do not generate or include any company not on this list or an alternate name for any company on this list. "
        "ONLY USE THE COMPANY NAME EXACTLY AS SHOWN IN THE LIST. "
        "If the text refers to 'the Company', 'the Registrant', or uses a pronoun or generic phrase instead of a company name, "
        "you MUST look up and use the exact company name from the allowed list based on context (such as the file being processed). "
        "UNDER NO CIRCUMSTANCES should you output 'the Company', 'the Registrant', or any generic phrase as a company name. "
        "Only use the exact allowed company name.\n\n"
        f"Allowed Companies (match exactly):\n{joined_names}\n\n"
    )

    custom_template = company_instruction + ERExtractionTemplate.DEFAULT_TEMPLATE
    return ERExtractionTemplate(template=custom_template)


def get_node_types():
    """
    Define node type schema for knowledge graph.

    Returns:
        List of node type definitions
    """
    return [
        {"label": "Executive", "properties": [{"name": "name", "type": "STRING"}]},
        {"label": "Product", "properties": [{"name": "name", "type": "STRING"}]},
        {"label": "FinancialMetric", "properties": [{"name": "name", "type": "STRING"}]},
        {"label": "RiskFactor", "properties": [{"name": "name", "type": "STRING"}]},
        {"label": "StockType", "properties": [{"name": "name", "type": "STRING"}]},
        {"label": "Transaction", "properties": [{"name": "name", "type": "STRING"}]},
        {"label": "TimePeriod", "properties": [{"name": "name", "type": "STRING"}]},
        {"label": "Company", "properties": [{"name": "name", "type": "STRING"}]}
    ]


def get_relationship_types():
    """
    Define relationship types for knowledge graph.

    Returns:
        List of relationship type definitions
    """
    return [
        {"label": "HAS_METRIC"},
        {"label": "FACES_RISK"},
        {"label": "ISSUED_STOCK"},
        {"label": "MENTIONS"}
    ]


def get_patterns():
    """
    Define valid patterns (triplets) for knowledge graph.

    Returns:
        List of (source_label, relationship_label, target_label) tuples
    """
    return [
        ("Company", "HAS_METRIC", "FinancialMetric"),
        ("Company", "FACES_RISK", "RiskFactor"),
        ("Company", "ISSUED_STOCK", "StockType"),
        ("Company", "MENTIONS", "Product")
    ]


async def process_pdf_file(file_path, pipeline, delay=0):
    """
    Process a single PDF file through the pipeline.

    Args:
        file_path: Path to PDF file
        pipeline: SimpleKGPipeline instance
        delay: Delay in seconds after processing (for rate limiting)
    """
    print(f"\nProcessing: {file_path}")
    try:
        await pipeline.run_async(file_path=file_path)
        print(f"Successfully processed: {file_path}")

        if delay > 0:
            print(f"Waiting {delay} seconds before next file...")
            time.sleep(delay)
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        raise


async def process_all_pdfs(pdf_directory, pipeline, delay=21):
    """
    Process all PDF files in a directory.

    Args:
        pdf_directory: Directory containing PDF files
        pipeline: SimpleKGPipeline instance
        delay: Delay in seconds between files (for rate limiting)
    """
    pdf_files = list(Path(pdf_directory).glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {pdf_directory}")
        return

    print(f"Found {len(pdf_files)} PDF files to process")

    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"\n[{i}/{len(pdf_files)}]")
        try:
            await process_pdf_file(str(pdf_file), pipeline, delay if i < len(pdf_files) else 0)
        except Exception as e:
            print(f"Continuing with next file...")
            continue

    print("\n=== Processing complete ===")


def create_vector_index_if_not_exists(driver, dimensions=768):
    """
    Create vector index in Neo4j for semantic search.

    Args:
        driver: Neo4j driver instance
        dimensions: Embedding dimensions (default: 768 for embeddinggemma)
    """
    print("\nCreating vector index...")
    try:
        create_vector_index(
            driver,
            name="chunkEmbeddings",
            label="Chunk",
            embedding_property="embedding",
            dimensions=dimensions,
            similarity_fn="cosine"
        )
        print("Vector index created successfully")
    except Exception as e:
        if "already exists" in str(e).lower():
            print("Vector index already exists")
        else:
            print(f"Error creating vector index: {e}")
            raise


def main():
    """Main entry point for the PDF loader application."""
    parser = argparse.ArgumentParser(
        description="Load PDF files into Neo4j GraphRAG knowledge graph"
    )
    parser.add_argument(
        "--pdf-dir",
        default="data/form10k-sample",
        help="Directory containing PDF files (default: data/form10k-sample)"
    )
    parser.add_argument(
        "--company-csv",
        default="data/Company_Filings.csv",
        help="CSV file with approved company names (default: data/Company_Filings.csv)"
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=21,
        help="Delay in seconds between processing files (default: 21)"
    )
    parser.add_argument(
        "--ollama-host",
        default="http://localhost:11434",
        help="Ollama server host (default: http://localhost:11434)"
    )
    parser.add_argument(
        "--llm-model",
        default="gpt-oss:120b-cloud",
        help="LLM model name (default: gpt-oss:120b-cloud)"
    )
    parser.add_argument(
        "--embedding-model",
        default="embeddinggemma",
        help="Embedding model name (default: embeddinggemma)"
    )
    parser.add_argument(
        "--embedding-dimensions",
        type=int,
        default=768,
        help="Embedding vector dimensions (default: 768)"
    )
    parser.add_argument(
        "--skip-index",
        action="store_true",
        help="Skip vector index creation"
    )

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()
    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_user = os.getenv("NEO4J_USERNAME")
    neo4j_password = os.getenv("NEO4J_PASSWORD")

    if not all([neo4j_uri, neo4j_user, neo4j_password]):
        print("Error: Missing Neo4j credentials in environment variables")
        print("Please ensure NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD are set")
        sys.exit(1)

    print("=== PDF Loader for Neo4j GraphRAG ===")
    print(f"Neo4j URI: {neo4j_uri}")
    print(f"Ollama host: {args.ollama_host}")
    print(f"LLM model: {args.llm_model}")
    print(f"Embedding model: {args.embedding_model}")

    # Connect to Neo4j
    print("\nConnecting to Neo4j...")
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

    try:
        # Verify connection
        driver.verify_connectivity()
        print("Connected to Neo4j successfully")
    except Exception as e:
        print(f"Error connecting to Neo4j: {e}")
        sys.exit(1)

    # Load approved companies
    approved_companies = load_approved_companies(args.company_csv)
    if not approved_companies:
        print("Warning: No approved companies loaded. Extraction may not work as expected.")

    # Initialize LLM and embeddings
    print("\nInitializing LLM and embeddings...")
    llm = OllamaLLM(
        model_name=args.llm_model,
        model_params={"options": {"temperature": 0}},
        host=args.ollama_host
    )
    embedder = OllamaEmbeddings(model=args.embedding_model)

    # Get schema and prompt template
    node_types = get_node_types()
    relationship_types = get_relationship_types()
    patterns = get_patterns()
    prompt_template = create_prompt_template(approved_companies)

    # Initialize pipeline
    print("Initializing knowledge graph pipeline...")
    try:
        # Create schema with strict enforcement
        schema = GraphSchema(
            node_types=node_types,
            relationship_types=relationship_types,
            patterns=patterns,
            additional_node_types=False,
            additional_relationship_types=False,
            additional_patterns=False
        )

        pipeline = SimpleKGPipeline(
            driver=driver,
            llm=llm,
            embedder=embedder,
            schema=schema,
            prompt_template=prompt_template
        )
        print("Pipeline initialized successfully")
    except Exception as e:
        print(f"Pipeline initialization failed: {e}")
        driver.close()
        sys.exit(1)

    # Process PDF files
    try:
        asyncio.run(process_all_pdfs(args.pdf_dir, pipeline, args.delay))
    except KeyboardInterrupt:
        print("\n\nProcessing interrupted by user")
    except Exception as e:
        print(f"\nError during processing: {e}")
        driver.close()
        sys.exit(1)

    # Create vector index
    if not args.skip_index:
        try:
            create_vector_index_if_not_exists(driver, args.embedding_dimensions)
        except Exception as e:
            print(f"Warning: Could not create vector index: {e}")

    # Close connection
    driver.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
