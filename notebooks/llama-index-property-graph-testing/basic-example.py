from llama_index.core import PropertyGraphIndex

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
import os
import logging
import sys

logging.basicConfig(
    stream=sys.stdout, level=logging.INFO
)  # logging.DEBUG for more verbose output
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from dotenv import load_dotenv
load_dotenv("../.env")
os.environ["OPENAI_API_VERSION"] = "2024-02-15-preview"
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = "2024-02-15-preview"

from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding

# You need to deploy your own embedding model as well as your own chat completion model
embed_model = AzureOpenAIEmbedding(
    model="text-embedding-ada-002",
    deployment_name="text-embedding-ada-002",
)

llm = AzureOpenAI(
    engine="gpt-4o", model="gpt-4o", temperature=0.0
)

from llama_index.core import Settings

Settings.llm = llm
Settings.embed_model = embed_model


response = llm.complete("The sky is a beautiful blue and")
print(response)
# Load documents and build index
documents = SimpleDirectoryReader(
    "../../data/2-interim/"
).load_data()

import nest_asyncio

nest_asyncio.apply()

# create
index = PropertyGraphIndex.from_documents(
    documents,
    llm= llm,
    embed_model=embed_model,
    show_progress = True,
)
index.property_graph_store.save_networkx_graph(name="./kg_free_form1.html")

# use
retriever = index.as_retriever(
    include_text=True,  # include source chunk with matching paths
    similarity_top_k=2,  # top k for vector kg node retrieval
)
nodes = retriever.retrieve("Borrower will")
from pprint import pprint
for node in nodes:
    pprint(node.text)
query_engine = index.as_query_engine(
    include_text=True,  # include source chunk with matching paths
    similarity_top_k=2,  # top k for vector kg node retrieval
)
response = query_engine.query("Borrower will")
response.response

from llama_index.core import VectorStoreIndex
rag_index = VectorStoreIndex.from_documents(documents)
# use
rag_retriever = rag_index.as_retriever(
    include_text=True,  # include source chunk with matching paths
    similarity_top_k=2,  # top k for vector kg node retrieval
)
nodes = rag_retriever.retrieve("Borrower")

rag_query_engine = rag_index.as_query_engine(
    include_text=True,  # include source chunk with matching paths
    similarity_top_k=2,  # top k for vector kg node retrieval
)
rag_response = rag_query_engine.query("Borrower will")

import pprint
pprint.pprint(f"Graph RAG response: {response.response}")
pprint.pprint(f"Vector RAG response: {rag_response.response}")


# Function to extract and print document information
def extract_documents_from_response(response):
    for node_with_score in response.source_nodes:
        node = node_with_score.node
        pprint.pprint(f"Document ID: {node.id_}")
        pprint.pprint(f"Document Content: {node.text}")
        for key, value in node.metadata.items():
            pprint.pprint(f"{key}: {value}")
        pprint.pprint("\n")

# Call the function with the response
extract_documents_from_response(response)
extract_documents_from_response(rag_response)

pprint.pprint(f"Graph RAG text retrieved: {response.text}")
pprint.pprint(f"Vector RAG text retrieved: {rag_response.text}")
response.metadata
#the Borrower will consider alter -\nnatives and implement technically and financially feasible and cost-effective\n11 options12 to avoid or \nminimize project-related air emissions during the design, construction and operation of the project\n 

# First Attempt to create a predefined schema

from typing import Literal
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor
from typing import Literal

# Define entities
entities = Literal["DOCUMENT", "SECTION", "SUBSECTION", "PARAGRAPH", "TERM"]

# Define relationships
relations = Literal["HAS_SECTION", "HAS_SUBSECTION", "HAS_PARAGRAPH", "CONTAINS_TERM", "NEXT"]

# Define which entities can have which relations
validation_schema = {
    "DOCUMENT": ["HAS_SECTION", "CONTAINS_TERM"],
    "SECTION": ["HAS_SUBSECTION", "NEXT", "CONTAINS_TERM"],
    "SUBSECTION": ["HAS_PARAGRAPH", "NEXT", "CONTAINS_TERM"],
    "PARAGRAPH": ["CONTAINS_TERM", "NEXT"],
    "TERM": []  # Terms do not have outgoing relationships in this schema
}

kg_extractor = SchemaLLMPathExtractor(
    llm=llm,
    possible_entities=entities,
    possible_relations=relations,
    kg_validation_schema=validation_schema,
    strict=True,
)

# create
index_with_predefined_schema = PropertyGraphIndex.from_documents(
    documents,
    llm= llm,
    embed_model=embed_model,
    kg_extractors=[kg_extractor],
    show_progress = True,

)
index_with_predefined_schema.property_graph_store.save_networkx_graph(name="./kg_predefined_schema.html")


# Second Attempt to create a predefined schema

# Define entities
entities = Literal["DOCUMENT", "SECTION", "INTRODUCTION", "OBJECTIVES", "SCOPE_OF_APPLICATION", "REQUIREMENTS", "SUBREQUIREMENTS"]

# Define relationships
relations = Literal["HAS_SECTION", "HAS_INTRODUCTION", "HAS_OBJECTIVES", "HAS_SCOPE_OF_APPLICATION", "HAS_REQUIREMENTS", "HAS_SUBREQUIREMENTS", "NEXT"]

# Define which entities can have which relations
validation_schema = {
    "DOCUMENT": ["HAS_SECTION", "NEXT"],
    "SECTION": ["HAS_INTRODUCTION", "HAS_OBJECTIVES", "HAS_SCOPE_OF_APPLICATION", "HAS_REQUIREMENTS", "NEXT"],
    "INTRODUCTION": ["NEXT"],
    "OBJECTIVES": ["NEXT"],
    "SCOPE_OF_APPLICATION": ["NEXT"],
    "REQUIREMENTS": ["HAS_SUBREQUIREMENTS", "NEXT"],
    "SUBREQUIREMENTS": ["NEXT"],
}

kg_extractor = SchemaLLMPathExtractor(
    llm=llm,
    possible_entities=entities,
    possible_relations=relations,
    kg_validation_schema=validation_schema,
    strict=True,
)

# create
index_with_predefined_schema2 = PropertyGraphIndex.from_documents(
    documents,
    llm= llm,
    embed_model=embed_model,
    kg_extractors=[kg_extractor],
    show_progress = True,

)
index_with_predefined_schema2.property_graph_store.save_networkx_graph(name="./kg_predefined_schema2.html")



# -------------------- Graph Store --------------------
# Pending I have to install docker image of neo4j
# docker run \
#     -p 7474:7474 -p 7687:7687 \
#     -v $PWD/data:/data -v $PWD/plugins:/plugins \
#     --name neo4j-apoc \
#     -e NEO4J_apoc_export_file_enabled=true \
#     -e NEO4J_apoc_import_file_enabled=true \
#     -e NEO4J_apoc_import_file_use__neo4j__config=true \
#     -e NEO4JLABS_PLUGINS=\[\"apoc\"\] \
#     neo4j:latest

from llama_index.graph_stores.neo4j import Neo4jPGStore

graph_store = Neo4jPGStore(
    username="neo4j",
    password="<password>",
    url="bolt://localhost:7687",
)







# ---------------------------- Retrieval ----------------------------

# use
retriever = index.as_retriever(
    include_text=True,  # include source chunk with matching paths
    similarity_top_k=2,  # top k for vector kg node retrieval
    embed_model=embed_model,
)
nodes = retriever.retrieve("Stakeholder")

query_engine = index.as_query_engine(
    include_text=True,  # include source chunk with matching paths
    similarity_top_k=2,  # top k for vector kg node retrieval
    embed_model=embed_model,
)
response = query_engine.query("Stakeholder")

# save and load
index.storage_context.persist(persist_dir="./storage")

from llama_index.core import StorageContext, load_index_from_storage

index = load_index_from_storage(
    StorageContext.from_defaults(persist_dir="./storage")
)

# loading from existing graph store (and optional vector store)
# load from existing graph/vector store
index = PropertyGraphIndex.from_existing(
    property_graph_store=graph_store, vector_store=vector_store,# ...
)