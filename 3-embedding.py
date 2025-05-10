from dotenv import load_dotenv

from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from weaviate.classes.query import MetadataQuery
from weaviate.classes.config import Configure, Property, DataType

from utils.connection import weaviate_client
from utils.tokenizer import OpenAITokenizerWrapper
from utils.embeddings import get_embeddings
from constants import MAX_TOKENS, COLLECTION_NAME


load_dotenv()

converter = DocumentConverter()
tokenizer = OpenAITokenizerWrapper()
result = converter.convert("https://arxiv.org/pdf/2408.09869")

chunker = HybridChunker(tokenizer=tokenizer, max_tokens=MAX_TOKENS, merge_peers=True)
chunks = list(chunker.chunk(dl_doc=result.document))

with weaviate_client() as wb_client:
    collections = wb_client.collections.list_all()
    if COLLECTION_NAME not in collections:
        print(f"Creating collection {COLLECTION_NAME}...")
        wb_client.collections.create(
            name=COLLECTION_NAME,
            vectorizer_config=Configure.Vectorizer.none(),
            properties=[
                Property(name="text", data_type=DataType.TEXT),
                Property(name="filename", data_type=DataType.TEXT),
                Property(name="pageNumbers", data_type=DataType.INT_ARRAY),
                Property(name="title", data_type=DataType.TEXT),
            ],
        )

    collection = wb_client.collections.get(COLLECTION_NAME)

    # TODO: Uncomment this to create embeddings of the chunks
    # i = 0
    # for chunk in chunks:
    #     text = chunk.text
    #     properties = {
    #         "filename": chunk.meta.origin.filename,
    #         "pageNumbers": sorted(
    #             {prov.page_no for item in chunk.meta.doc_items for prov in item.prov}
    #         )
    #         or [],
    #         "title": (chunk.meta.headings[0] if chunk.meta.headings else ""),
    #         "text": text,
    #     }

    #     embedding = get_embeddings(text)

    #     # Insert data with vector
    #     collection.data.insert(
    #         properties=properties, vector=embedding, uuid=str(uuid.uuid4())
    #     )
    #     i += 1
    #     print(f"Processed {i}/{len(chunks)} embeddings...")

    print("Embeddings creation complete!")

    # Querying with similarity search
    query = "quantum physics"
    query_embedding = get_embeddings(query)
    response = collection.query.near_vector(
        near_vector=query_embedding,
        limit=5,
        return_properties=["text", "filename"],
        return_metadata=MetadataQuery(distance=True),
    )

    # Print results
    for obj in response.objects:
        print(f"Text: {obj.properties['text'][:500]}...")
        print(f"Filename: {obj.properties['filename']}")
        print(f"Distance: {obj.metadata.distance}")
        print("---")

print("Script completed successfully")
