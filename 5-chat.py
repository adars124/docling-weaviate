import streamlit as st

# import weaviate
import weaviate.classes as wvc
from openai import OpenAI
from dotenv import load_dotenv

from utils.connection import create_weaviate_client
from utils.embeddings import get_embeddings
from constants import COLLECTION_NAME

load_dotenv()

client = OpenAI()


# Initialize Weaviate connection
@st.cache_resource
def init_db():
    """Initialize database connection.

    Returns:
        Weaviate collection object
    """
    wb_client = create_weaviate_client()
    return wb_client.collections.get(COLLECTION_NAME)


def get_context(query: str, collection, num_results: int = 5) -> str:
    """Search the database for relevant context."""
    query_embedding = get_embeddings(query)
    results = collection.query.near_vector(
        near_vector=query_embedding,
        limit=num_results,
        return_metadata=wvc.query.MetadataQuery(distance=True),
    )

    contexts = []
    for obj in results.objects:
        text = obj.properties.get("text", "")
        source = obj.properties.get("source", "")
        title = obj.properties.get("title", "")
        page_numbers = obj.properties.get("pageNumbers", [])
        filename = obj.properties.get("filename", "")

        # Build source citation
        source_parts = []
        if filename:
            source_parts.append(filename)
        if page_numbers:
            source_parts.append(f"p. {', '.join(str(p) for p in page_numbers)}")

        source = f"\nSource: {' - '.join(source_parts)}"
        if title:
            source += f"\nTitle: {title}"

        contexts.append(f"{text}{source}")

    return "\n\n".join(contexts)


def get_chat_response(messages, context: str) -> str:
    """Get streaming response from OpenAI API."""
    system_prompt = f"""You are a helpful assistant that answers questions based on the provided context.
    Use only the information from the context to answer questions. If you're unsure or the context
    doesn't contain the relevant information, say so.
    
    Context:
    {context}
    """
    messages_with_context = [{"role": "system", "content": system_prompt}, *messages]

    # Create the streaming response
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages_with_context,
        temperature=0.7,
        stream=True,
    )

    # Use Streamlit's built-in streaming capability
    response = st.write_stream(stream)
    return response


st.title("📚 Docling Q&A")

if "messages" not in st.session_state:
    st.session_state.messages = []

collection = init_db()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question..."):
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.status("Searching document...", expanded=False) as status:
        context = get_context(prompt, collection)
        st.markdown(
            """
            <style>
            .search-result {
                margin: 10px 0;
                padding: 10px;
                border-radius: 4px;
                background-color: #f0f2f6;
            }
            .search-result summary {
                cursor: pointer;
                color: #0f52ba;
                font-weight: 500;
            }
            .search-result summary:hover {
                color: #1e90ff;
            }
            .metadata {
                font-size: 0.9em;
                color: #666;
                font-style: italic;
            }
            </style>
        """,
            unsafe_allow_html=True,
        )

        st.write("Found relevant sections:")
        for chunk in context.split("\n\n"):
            # Split into text and metadata parts
            parts = chunk.split("\n")
            text = parts[0]
            metadata = {
                line.split(": ")[0]: line.split(": ")[1]
                for line in parts[1:]
                if ": " in line
            }

            source = metadata.get("Source", "Unknown source")
            title = metadata.get("Title", "Untitled section")

            st.markdown(
                f"""
                <div class="search-result">
                    <details>
                        <summary>{source}</summary>
                        <div class="metadata">Section: {title}</div>
                        <div style="margin-top: 8px;">{text}</div>
                    </details>
                </div>
            """,
                unsafe_allow_html=True,
            )

    with st.chat_message("assistant"):
        response = get_chat_response(st.session_state.messages, context)

    st.session_state.messages.append({"role": "assistant", "content": response})
