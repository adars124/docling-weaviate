import os
import time
from dotenv import load_dotenv
import contextlib

import weaviate
from weaviate.exceptions import WeaviateConnectionError

load_dotenv()


def connect_to_weaviate(retries=3, delay=2):
    """Connect to Weaviate with retries"""
    for attempt in range(retries):
        try:
            print(f"Connecting to Weaviate (attempt {attempt+1}/{retries})...")
            client = weaviate.connect_to_local(
                headers={"X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")},
                skip_init_checks=True,
            )
            print("Successfully connected to Weaviate!")
            return client
        except WeaviateConnectionError as e:
            print(f"Connection error: {e}")
            if attempt < retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print("Failed to connect to Weaviate after multiple attempts.")
                print(
                    "Please ensure Weaviate is running with: docker run -d -p 8080:8080 -p 50051:50051 -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true -e DEFAULT_VECTORIZER_MODULE=none -e GRPC_ENABLED=true semitechnologies/weaviate:latest"
                )
                raise


def create_weaviate_client():
    """Create and return a Weaviate client without context management.

    The caller is responsible for closing the connection when done.
    This is useful for long-running applications that need to maintain
    a persistent connection.
    """
    return connect_to_weaviate()


@contextlib.contextmanager
def weaviate_client(retries=3, delay=2):
    """Context manager for Weaviate client to ensure connection is closed properly."""
    client = None
    try:
        client = connect_to_weaviate(retries=retries, delay=delay)
        yield client
    finally:
        if client:
            print("Closing Weaviate connection...")
            client.close()
