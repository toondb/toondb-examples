import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class ToonDBConfig:
    db_path: str = os.getenv("TOONDB_AZURE_PATH", "./toondb_azure_data")
    
@dataclass
class AzureConfig:
    api_key: str = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint: str = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_version: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    deployment: str = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    embedding_deployment: str = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")

def get_toondb_config() -> ToonDBConfig:
    return ToonDBConfig()

def get_azure_config() -> AzureConfig:
    return AzureConfig()
