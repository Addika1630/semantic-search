import pandas as pd
import logging
from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings
from llama_index.legacy.embeddings.langchain import LangchainEmbedding
import os


def load_environment_vars() -> dict:
    """Load required environment variables. Raise an exception if any are missing."""

    pinecone_api_key = "pcsk_2sz2n4_DHSRzSoheEkp9PPph2CAbVuoJWGNjUPHCcvjS9xBwuEi5k7WaDgnzfmqEyM9Gfs"
    pinecone_environment = "us-east-1"

    if not pinecone_api_key:
        raise EnvironmentError("PINECONE_API_KEY environment variable not set.")
    if not pinecone_environment:
        raise EnvironmentError("PINECONE_ENVIRONMENT environment variable not set.")

    logging.info("Environment variables loaded.")
    return {"PINECONE_API_KEY": pinecone_api_key, "PINECONE_ENV": pinecone_environment}


def generate_embeddings(df: pd.DataFrame, text_columns: list) -> pd.DataFrame:
    """
    Generate embeddings using HuggingFace model via Langchain.
    Embeddings are generated from a combination of specified text columns.
    """
    logging.info(f"Generating embeddings using columns: {text_columns}")
    embed_model = LangchainEmbedding(
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    )

    # Combine the specified text columns into a single string
    combined_texts = df[text_columns].apply(lambda row: " ".join(row.dropna().astype(str)), axis=1)

    embeddings = []
    for text in combined_texts:
        try:
            embedding = embed_model.get_text_embedding(text)
            embeddings.append(embedding)
        except Exception as e:
            logging.error(f"Failed to generate embedding for text: {text}. Error: {e}")
            embeddings.append(None)

    df["embedding"] = embeddings
    return df

def index_data(pinecone_api_key: str, pinecone_environment: str):
    """Index Data into Pinecone with Batching."""
    file_name = "langchain-github-issues-2023-09-18.pkl"

    logging.info(f"Loading data from '{file_name}'")
    df = pd.read_pickle(file_name)

    # Debugging: Check DataFrame structure
    logging.info(f"DataFrame columns: {list(df.columns)}")
    logging.info(f"First few rows of DataFrame: \n{df.head()}")

    # Generate embeddings if the embedding column is missing
    if "embedding" not in df.columns:
        logging.info("Generating embeddings for the DataFrame.")
        df = generate_embeddings(df, text_columns=["title", "description"])
        df = df.dropna(subset=["embedding"])  # Drop rows with failed embeddings

    logging.info("Initializing Pinecone Client")
    pc = Pinecone(api_key=pinecone_api_key)

    index_name = "github-issues"
    dimension = len(df["embedding"].iloc[0])  # Dynamic dimension based on embedding size

    # Delete index if it exists
    pc.delete_index(index_name)

    # Create an index if it doesn't exist
    if index_name not in pc.list_indexes():
        logging.info(f"Creating Pinecone index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",  # Change to 'euclidean' or 'dotproduct' if needed
            spec=ServerlessSpec(cloud="aws", region=pinecone_environment)
        )

    # Connect to the index
    index = pc.Index(index_name)

    logging.info(f"Indexing data to Pinecone: '{index_name}'")
    try:
        # Batch upload vectors
        batch_size = 100  # Adjust batch size to fit within 4MB limit
        vectors = []
        for item in df.itertuples():
            vector_id = item.url
            vector = item.embedding
            metadata = {
                "title": item.title,
                "description": item.description,
                "url": item.url,
                "labels": item.labels,
                "state": item.state,
                "creator": item.creator,
                "created_at": item.created_at,
            }
            vectors.append({"id": vector_id, "values": vector, "metadata": metadata})

            # If batch size is reached, upsert to Pinecone
            if len(vectors) >= batch_size:
                index.upsert(vectors)
                vectors = []  # Reset batch

        # Upsert remaining vectors if any
        if vectors:
            index.upsert(vectors)

        logging.info("Data successfully indexed.")
    except Exception as ex:
        logging.error(f"Unexpected Error: {ex}")
        raise
        

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    try:
        env_vars = load_environment_vars()
        index_data(env_vars["PINECONE_API_KEY"], env_vars["PINECONE_ENV"])
    except EnvironmentError as ee:
        logging.error(f"Environment Error: {ee}")
        raise
    except Exception as ex:
        logging.error(f"Unexpected Error: {ex}")
        raise


