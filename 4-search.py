import pandas as pd

from weaviate.classes.query import MetadataQuery
from utils.connection import weaviate_client
from utils.embeddings import get_embeddings
from constants import COLLECTION_NAME

with weaviate_client() as wb_client:
    collection = wb_client.collections.get(COLLECTION_NAME)

    query = "quantum physics"
    query_embedding = get_embeddings(query)

    # Perform similarity search
    response = collection.query.near_vector(
        near_vector=query_embedding,
        limit=5,
        return_properties=["text", "filename", "pageNumbers", "title"],
        return_metadata=MetadataQuery(distance=True),
    )

    # Convert to a format you can work with
    results = []
    for obj in response.objects:
        results.append({"uuid": obj.uuid, **obj.properties})

    # Convert to pandas DataFrame
    df = pd.DataFrame(results)
    print(df)
