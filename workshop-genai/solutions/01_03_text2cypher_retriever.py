"""
Text2Cypher Retriever Solution

This script demonstrates how to use the Text2CypherRetriever to convert natural language
questions into Cypher queries that can be executed against a Neo4j knowledge graph.

Key Concepts:
-------------
1. Text2Cypher: Automatically converts natural language to Cypher queries using an LLM
2. Schema Extraction: Gets the graph structure (nodes, relationships, properties)
3. Schema Visualization: Creates an interactive HTML visualization of the graph schema
4. GraphRAG: Combines retrieval and generation for natural language responses

How It Works:
-------------
The Text2CypherRetriever uses a Large Language Model (LLM) to:
1. Analyze the user's natural language question
2. Review the Neo4j graph schema (node types, relationships, properties)
3. Generate an appropriate Cypher query to answer the question
4. Execute the query and return results
5. (Optional) Use GraphRAG to generate a natural language answer from the results
"""

from neo4j import GraphDatabase
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.retrievers import Text2CypherRetriever
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.schema import get_schema
from neo4j_graphrag.experimental.components.schema import SchemaFromExistingGraphExtractor
from neo4j_graphrag.experimental.utils.schema import schema_visualization
import asyncio

# ============================================================================
# SECTION 1: Environment Setup
# ============================================================================
# Load environment variables from .env file
# These include database credentials and API keys
import os
from dotenv import load_dotenv
load_dotenv()
NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USER = os.getenv('NEO4J_USERNAME')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
NEO4J_DATABASE = os.getenv('NEO4J_DATABASE')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Create a connection to the Neo4j database
# The driver manages the connection pool and sessions
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# ============================================================================
# SECTION 2: Initialize LLM and Embeddings
# ============================================================================
# Initialize the Large Language Model (LLM) for:
# - Converting natural language to Cypher queries
# - Generating natural language responses from query results
llm = OpenAILLM(model_name='gpt-4o', api_key=OPENAI_API_KEY)

# Initialize embeddings (not used in this example, but commonly needed for vector search)
embedder = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

# ============================================================================
# SECTION 3: Extract and Display Graph Schema
# ============================================================================
# Get the schema as a formatted string
# This includes node labels, relationship types, and their properties
# The schema is essential for the LLM to understand the graph structure
schema = get_schema(driver)
print(schema)

# ============================================================================
# SECTION 4: Visualize the Graph Schema
# ============================================================================
async def visualize_schema():
    """
    Create an interactive HTML visualization of the graph schema.

    This function:
    1. Extracts the structured schema from the Neo4j database
    2. Creates an interactive visualization using neo4j-viz
    3. Saves the visualization as an HTML file

    The visualization shows:
    - All node types (labels) as nodes
    - All relationship types as connections between nodes
    - Properties for each node and relationship type
    """
    # Extract structured schema from the existing graph
    # This queries the database to get all node types, relationship types, and properties
    schema_extractor = SchemaFromExistingGraphExtractor(driver=driver)
    graph_schema = await schema_extractor.run()

    # Create the visualization
    # VG is a VisualizationGraph object that contains nodes and relationships
    VG = schema_visualization(graph_schema)
    html = VG.render()

    # Save the generated HTML file
    # This creates an interactive graph that can be opened in a web browser
    with open("my_schema.html", "w") as f:
        f.write(html.data)
    print("Schema visualization saved to my_schema.html")

# Run the async function to generate the visualization
asyncio.run(visualize_schema())

# ============================================================================
# SECTION 5: Text2Cypher Retriever - Direct Query Execution
# ============================================================================
# Create a Text2CypherRetriever instance
# This retriever automatically converts natural language to Cypher queries
text2cypher_retriever = Text2CypherRetriever(
    driver=driver,           # Neo4j connection
    llm=llm,                 # LLM for generating Cypher from natural language
    neo4j_schema=schema      # Graph schema to guide Cypher generation
)

# Example 1: Find companies owned by a specific asset manager
# --------------------------------------------------------
query = "What companies are owned by BlackRock Inc."

# The retriever will:
# 1. Send the query + schema to the LLM
# 2. LLM generates a Cypher query (e.g., MATCH (am:AssetManager)-[:OWNS]->(c:Company)...)
# 3. Execute the Cypher query against Neo4j
# 4. Return the results
cypher_query = text2cypher_retriever.get_search_results(query)

# Display the original question
print("Original Query:", query)

# Display the generated Cypher query
# The LLM creates this based on understanding the question and the schema
print("Generated Cypher:", cypher_query.metadata["cypher"])

# Display the query results
# These are the raw Neo4j records returned by executing the Cypher query
print("Cypher Query Results:")
for record in cypher_query.records:
    print(record)


# ============================================================================
# SECTION 6: GraphRAG - Natural Language Responses
# ============================================================================
# GraphRAG combines retrieval (Text2Cypher) with generation (LLM) to produce
# natural language answers to questions about the graph

# Example 2: Using GraphRAG for natural language responses
# --------------------------------------------------------
query = "Who are the assets managers?"

# Create a GraphRAG instance
# GraphRAG orchestrates the entire pipeline:
# 1. Uses the retriever to get data from the graph (via Text2Cypher)
# 2. Formats the results as context
# 3. Uses the LLM to generate a natural language answer
rag = GraphRAG(llm=llm, retriever=text2cypher_retriever)

# Execute the search
# return_context=True includes the raw query results in the response
response = rag.search(
    query,
    return_context=True
)

# The response contains:
# - answer: Natural language answer generated by the LLM
# - retriever_result: The raw results from the Text2Cypher query
print(response.answer)

# Show the generated Cypher query
# This reveals how the question was translated to a graph query
print("Generated Cypher:", response.retriever_result.metadata["cypher"])

# Show the context (raw data) used to generate the answer
# This is the actual data retrieved from Neo4j
print("Context:", *response.retriever_result.items, sep="\n")


# ============================================================================
# Additional Examples
# ============================================================================
"""
Try these example queries:

1. Summarise the products mentioned in the company filings.
2. What risk factors does Apple face?
3. How many stock types are there in total?
4. Which companies have transactions?
5. What executives are mentioned in the documents?

The Text2CypherRetriever will automatically generate appropriate Cypher queries
for each question based on the graph schema.
"""