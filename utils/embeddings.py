from typing import Optional
from dotenv import load_dotenv

from openai import OpenAI

from constants import EMBEDDING_MODEL

load_dotenv()

client = OpenAI()


def get_embeddings(text: str) -> Optional[list[float]]:
    """Get embeddings for a text using OpenAI's embedding API."""
    try:
        response = client.embeddings.create(
            input=[text],
            model=EMBEDDING_MODEL,
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embeddings for {text}: {e}")
        return None
