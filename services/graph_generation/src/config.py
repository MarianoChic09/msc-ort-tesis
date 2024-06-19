from typing import List, Optional
from dotenv import load_dotenv, find_dotenv
from pydantic import field_validator
from pydantic_settings import BaseSettings

# load my .env file variables as environment variables so I can access them
# with os.environ[] statements
load_dotenv(find_dotenv())


class Config(BaseSettings):

    azure_openai_api_key : str
    azure_openai_endpoint : str
    azure_openai_api_version : str


config = Config()
