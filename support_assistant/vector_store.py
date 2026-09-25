from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / "chroma_db"

MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "zepto_policies"


# Load the local embedding model
model = SentenceTransformer(MODEL_NAME)


def load_policy_documents():
    """Load all 8 policy markdown files."""
    documents = []

    for file_path in sorted(BASE_DIR.glob("doc_*.md")):
        text = file_path.read_text(encoding="utf-8").strip()

        if text:
            documents.append(
                {
                    "source": file_path.name,
                    "text": text,
                }
            )

    return documents


def chunk_text(text):
    """
    Split each policy document into section-level chunks.

    Each ## heading and its following content becomes one chunk.
    This gives the retriever smaller and more relevant pieces of text.
    """

    sections = []
    current_section = []

    for line in text.splitlines():
        line = line.strip()

        if not line:
            continue

        # Start a new chunk whenever a level-2 heading appears.
        if line.startswith("## "):
            if current_section:
                sections.append("\n".join(current_section).strip())

            current_section = [line]

        else:
            current_section.append(line)

    # Add the final section.
    if current_section:
        sections.append("\n".join(current_section).strip())

    return sections


def build_vector_store():
    """Create the ChromaDB vector store."""

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    # Recreate collection so rerunning this file gives a clean index.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    policy_docs = load_policy_documents()

    all_chunks = []
    all_sources = []
    all_ids = []

    for document in policy_docs:
        chunks = chunk_text(document["text"])

        for index, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_sources.append(document["source"])
            all_ids.append(
                f"{document['source']}_{index}"
            )

    print(f"Loaded {len(policy_docs)} policy documents.")
    print(f"Created {len(all_chunks)} text chunks.")

    # Create local embeddings using all-MiniLM-L6-v2.
    embeddings = model.encode(
        all_chunks,
        normalize_embeddings=True,
    ).tolist()

    collection.add(
        ids=all_ids,
        documents=all_chunks,
        embeddings=embeddings,
        metadatas=[
            {"source": source}
            for source in all_sources
        ],
    )

    print("ChromaDB vector store created successfully.")

    return collection


def get_collection():
    """Open the existing ChromaDB collection."""

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    return client.get_collection(
        COLLECTION_NAME
    )


def retrieve_policy(query, top_k=3):
    """Retrieve the top-k policy chunks using cosine similarity."""

    collection = get_collection()

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
    ).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
    )

    retrieved = []

    for document, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        retrieved.append(
            {
                "text": document,
                "source": metadata["source"],
                "distance": distance,
            }
        )

    return retrieved


if __name__ == "__main__":
    build_vector_store()

    print("\nTesting retrieval...\n")

    results = retrieve_policy(
        "What should I do if my order is delayed?"
    )

    for result in results:
        print(f"Source: {result['source']}")
        print(f"Distance: {result['distance']:.4f}")
        print(result["text"])
        print("-" * 60)