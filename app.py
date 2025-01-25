"""
Streamlit app for GitHub Semantic Search with Weaviate.
It supports the following search modes:
- Near Text
- BM25
- Hybrid
The user's OpenAI API key is used to generate vector embeddings for the search query.

Author:
    @dcarpintero : https://github.com/dcarpintero
"""
from langchain_huggingface import HuggingFaceEmbeddings
from llama_index.legacy.embeddings.langchain import LangchainEmbedding
import pandas as pd
import streamlit as st
from pinecone import Pinecone
import logging
from dotenv import load_dotenv
import os


from datetime import datetime
from typing import Optional


def load_environment_vars() -> dict:
    """Load required environment variables. Raise an exception if any are missing."""

    # Uncomment below if loading from .env
    load_dotenv()
    # openapi_key = os.getenv("OPENAI_API_KEY")
    # weaviate_url = os.getenv("WEAVIATE_URL")
    # weaviate_api_key = os.getenv("WEAVIATE_API_KEY")

    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_environment = "us-east-1"

    if not pinecone_api_key:
        raise EnvironmentError("PINECONE_API_KEY environment variable not set.")
    if not pinecone_environment:
        raise EnvironmentError("PINECONE_ENVIRONMENT environment variable not set.")

    logging.info("Environment variables loaded.")
    return {"PINECONE_API_KEY": pinecone_api_key, "PINECONE_ENV": pinecone_environment}


@st.cache_resource(show_spinner=False)
def pinecone_client(pinecone_api_key: str, pinecone_environment: str):
    logging.info(f"Initializing Pinecone Client: '{pinecone_environment}'")
    pc = Pinecone(api_key=pinecone_api_key)
    index_name = "github-issues"
    index = pc.Index(index_name)
    return index


@st.cache_data
def query_with_pinecone(_pc_client: Pinecone, query, index_name, embed_model, max_results=10) -> pd.DataFrame:
    """
    Search GitHub Issues in Pinecone with vector similarity (Cosine/Euclidean).
    """
    embed_model = LangchainEmbedding(
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    )

    # Convert the query into an embedding vector
    query_vector = embed_model.get_text_embedding(query)

    # Perform the similarity search in Pinecone
    index = _pc_client  # Access the Pinecone index
    response = index.query(vector=query_vector, top_k=max_results, include_metadata=True)

    # Format the response into a DataFrame
    data = []
    for match in response["matches"]:
        data.append(
            {
                "title": match["metadata"]["title"],
                "url": match["metadata"]["url"],
                "labels": match["metadata"]["labels"],
                "description": match["metadata"]["description"],
                "created_at": match["metadata"]["created_at"],
                "state": match["metadata"]["state"],
                "score": match["score"],
            }
        )

    return pd.DataFrame.from_dict(data, orient="columns")


def onchange_with_cosine_similarity():
    if st.session_state.with_cosine_similarity:
        st.session_state.with_bm25 = False
        st.session_state.with_hybrid = False


def format_date(date_string: str) -> Optional[str]:
    try:
        date = datetime.strptime(date_string, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return None
    return date.strftime("%d %B %Y")


# Initialize session state variables
if "with_cosine_similarity" not in st.session_state:
    st.session_state.with_cosine_similarity = False
if "with_bm25" not in st.session_state:
    st.session_state.with_bm25 = False
if "with_hybrid" not in st.session_state:
    st.session_state.with_hybrid = False

env_vars = load_environment_vars()
pinecone_client = pinecone_client(
    env_vars["PINECONE_API_KEY"], env_vars["PINECONE_ENV"]
)

st.header("🦜 Semantic Search on Langchain Issues 🔍")

with st.sidebar.expander("🐙 GITHUB-REPOSITORY", expanded=True):
    st.text_input(
        label="GITHUB-REPOSITORY",
        key="github_repo",
        label_visibility="hidden",
        value="langchain-ai/langchain",
        disabled=True,
    )

with st.sidebar.expander("🔧 WEAVIATE-SETTINGS", expanded=True):
    st.toggle(
        "Cosine Similarity Search",
        key="with_cosine_similarity",
        on_change=onchange_with_cosine_similarity,
    )

with st.sidebar.expander("🔍 SEARCH-RESULTS", expanded=True):
    bm25_score = st.slider("BM25 Score", min_value=1.0, max_value=4.0, value=1.9, step=0.1)
    hybrid_score = st.slider(
        "Hybrid Score (Scaled)", min_value=1.0, max_value=3.0, value=1.1, step=0.05
    )
    max_results = st.slider("Max Results", min_value=0, max_value=100, value=10, step=1)

with st.sidebar:
    "[![Weaviate Docs](https://img.shields.io/badge/Weaviate%20Docs-gray)](https://weaviate.io/developers/weaviate)"

query = st.text_input("Search in 'langchain-ai/langchain'", "")

if query:
    if st.session_state.with_cosine_similarity:
        st.subheader("Cosine Similarity Search")
        df = query_with_pinecone(pinecone_client, query, "github-issues", max_results)
    else:
        st.info("ℹ️ Select your preferred Search Mode (Near Text, BM25 or Hybrid)!")
        st.stop()

    tab_list, tab_raw = st.tabs([f'Issues with "{query}"', "Raw"])

    with tab_list:
        if df is None:
            st.info("No GitHub Issues found.")
        else:
            for i in range(len(df)):
                issue = df.iloc[i]

                title = issue["title"]
                url = issue["url"]
                createdAt = format_date(issue["created_at"])
                score = issue["score"]

                if score >= 0.4:
                    st.markdown(f"[{title}]({url}) ({createdAt})")

    with tab_raw:
        if df is None:
            st.info("No GitHub Issues found.")
        else:
            st.dataframe(df, hide_index=True)


