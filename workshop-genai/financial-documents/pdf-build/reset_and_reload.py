#!/usr/bin/env python3
"""
Reset Neo4j database and reload with correct embeddings.

This script:
1. Drops the existing vector index
2. Clears all nodes and relationships
3. Allows you to reload the data with the correct embedder
"""

import os
import sys
from dotenv import load_dotenv
from neo4j import GraphDatabase

def clear_database(driver):
    """Clear all nodes and relationships from the database."""
    print("Clearing all nodes and relationships...")
    with driver.session() as session:
        # Delete all nodes and relationships
        result = session.run("MATCH (n) DETACH DELETE n")
        summary = result.consume()
        print(f"Deleted all nodes and relationships")

def drop_vector_index(driver, index_name="chunkEmbeddings"):
    """Drop the vector index if it exists."""
    print(f"Dropping vector index '{index_name}'...")
    with driver.session() as session:
        try:
            session.run(f"DROP INDEX {index_name} IF EXISTS")
            print(f"Vector index '{index_name}' dropped successfully")
        except Exception as e:
            print(f"Note: {e}")

def main():
    # Load environment variables
    load_dotenv()
    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_user = os.getenv("NEO4J_USERNAME")
    neo4j_password = os.getenv("NEO4J_PASSWORD")

    if not all([neo4j_uri, neo4j_user, neo4j_password]):
        print("Error: Missing Neo4j credentials in environment variables")
        sys.exit(1)

    print("=== Neo4j Database Reset ===")
    print(f"Neo4j URI: {neo4j_uri}")

    # Connect to Neo4j
    print("\nConnecting to Neo4j...")
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

    try:
        driver.verify_connectivity()
        print("Connected successfully")
    except Exception as e:
        print(f"Error connecting to Neo4j: {e}")
        sys.exit(1)

    # Drop index and clear data
    try:
        drop_vector_index(driver)
        clear_database(driver)
        print("\n✅ Database reset complete!")
        print("\nNext steps:")
        print("1. Run the PDF loader to reload data with correct embeddings:")
        print("   python workshop-genai/financial-documents/pdf-build/pdf_loader.py")
        print("\n   Or run the notebook:")
        print("   workshop-genai/financial-documents/pdf-build/01_PDF_Loader_for_Neo4j_GraphRAG.ipynb")
    except Exception as e:
        print(f"Error during reset: {e}")
        sys.exit(1)
    finally:
        driver.close()

if __name__ == "__main__":
    main()
