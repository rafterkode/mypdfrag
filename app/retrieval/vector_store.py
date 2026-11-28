import chromadb

from app.core.config import settings

client = chromadb.PersistentClient(path=settings.database_path)

collection = client.get_or_create_collection(name=settings.collection_name)
