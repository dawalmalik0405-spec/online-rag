import hashlib
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import torch
import chromadb
from text_extractor import extract_text_from_file


# ---------------- CHUNKING ----------------
def chunking(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_text(text)


# ---------------- EMBEDDINGS ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": device}
)

def embed_chunks(chunks):
    return embeddings.embed_documents(chunks)


# ---------------- CHROMA CLIENT & COLLECTION HELPERS ----------------
client = chromadb.PersistentClient(path="./chromadb_db")


def get_document_collection_name(filename: str) -> str:
    """Generates a valid, deterministic ChromaDB collection name from filename."""
    hash_str = hashlib.md5(filename.encode("utf-8")).hexdigest()
    return f"doc_{hash_str}"


def get_document_collection(filename: str):
    """Retrieves or creates the isolated collection for a specific document."""
    col_name = get_document_collection_name(filename)
    return client.get_or_create_collection(
        name=col_name,
        metadata={"hnsw:space": "cosine", "filename": filename}
    )


def delete_document_collection(filename: str):
    """Deletes the Chroma collection associated with a document."""
    col_name = get_document_collection_name(filename)
    try:
        client.delete_collection(name=col_name)
        return True
    except Exception:
        return False


# ---------------- STORE IN VECTOR DB ----------------
def store_vdb(chunks, vectors, filename):
    col_name = get_document_collection_name(filename)

    # Clean existing collection on re-upload to avoid duplicate chunks
    try:
        client.delete_collection(name=col_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=col_name,
        metadata={"hnsw:space": "cosine", "filename": filename}
    )

    # Deterministic chunk IDs: filename_chunk_0, filename_chunk_1, ...
    ids = [f"{filename}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"file": filename, "chunk_index": i} for i in range(len(chunks))]

    collection.add(
        documents=chunks,
        embeddings=vectors,
        ids=ids,
        metadatas=metadatas
    )

    return {
        "stored_chunks": len(chunks),
        "first_id": ids[0],
        "last_id": ids[-1]
    }


# ---------------- MAIN INGEST FUNCTION ----------------
def ingest_file(file_path, filename):
    text = extract_text_from_file(file_path)
    if not text or not text.strip():
        raise ValueError(f"No extractable text found in file: {filename}")

    chunks = chunking(text)
    if not chunks:
        raise ValueError(f"No chunks created for file: {filename}")

    vectors = embed_chunks(chunks)
    if not vectors:
        raise ValueError(f"Embedding failed for file: {filename}")

    result = store_vdb(chunks, vectors, filename)
    return result


# text  = extract_text_from_file("123.pdf")
# print(text)


