from llama_index.core import PropertyGraphIndex
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import Settings
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding

# Utilities
import os
import sys
from loguru import logger

# Local modules
from src.config import config


# You need to deploy your own embedding model as well as your own chat completion model
embed_model = AzureOpenAIEmbedding(
    model="text-embedding-ada-002",
    deployment_name="text-embedding-ada-002",
)

llm = AzureOpenAI(
    engine="gpt-4o", model="gpt-4o", temperature=0.0
)

Settings.llm = llm
Settings.embed_model = embed_model

