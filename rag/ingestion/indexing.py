import os
from pathlib import Path
import uuid
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from tqdm import tqdm

# Locate base directory and load .env
CURRENT_DIR = Path(__file__).resolve().parent
RAG_DIR = CURRENT_DIR.parent
BASE_DIR = RAG_DIR.parent

# Load environment from root .env or local .env
env_path = BASE_DIR / ".env"
if not env_path.exists():
    env_path = RAG_DIR / ".env"
load_dotenv(dotenv_path=env_path)

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY is not set in environment or .env file")

PDF_FOLDER = RAG_DIR / "data"
QDRANT_PATH = str(RAG_DIR / "qdrant_db")
COLLECTION_NAME = "medical_documents"

def index_documents(force_recreate: bool = False):
    print(f"=== Starting Medical RAG Ingestion ===")
    print(f"Document Folder : {PDF_FOLDER}")
    print(f"Qdrant DB Path  : {QDRANT_PATH}")
    print(f"Collection Name : {COLLECTION_NAME}")

    pdf_files = sorted(PDF_FOLDER.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {PDF_FOLDER}")

    print(f"Found {len(pdf_files)} PDF files to process.")

    # 1. Load PDFs
    all_pages = []
    for pdf_file in pdf_files:
        print(f"-> Loading: {pdf_file.name}")
        try:
            loader = PyPDFLoader(str(pdf_file))
            pages = loader.load()
            valid_pages = 0
            for page in pages:
                # Add source metadata
                page.metadata["source_file"] = pdf_file.name
                page.metadata["document_type"] = "clinical_guideline"
                if page.page_content.strip():
                    valid_pages += 1
            all_pages.extend(pages)
            print(f"   Loaded {len(pages)} pages ({valid_pages} non-empty text pages)")
        except Exception as e:
            print(f"   [Warning] Failed loading {pdf_file.name}: {e}")

    print(f"\nTotal pages loaded: {len(all_pages)}")

    # 2. Text Splitting
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.split_documents(all_pages)
    # Filter empty or very short whitespace chunks
    filtered_chunks = [c for c in chunks if len(c.page_content.strip()) > 30]
    print(f"Total chunks created: {len(chunks)} (Filtered meaningful chunks: {len(filtered_chunks)})")

    # 3. Setup Embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    # 4. Initialize Local Qdrant
    client = QdrantClient(path=QDRANT_PATH)

    if force_recreate and client.collection_exists(COLLECTION_NAME):
        print(f"Deleting existing collection {COLLECTION_NAME}...")
        client.delete_collection(COLLECTION_NAME)

    if not client.collection_exists(COLLECTION_NAME):
        print(f"Creating new Qdrant collection '{COLLECTION_NAME}' with 1536-dim Cosine distance...")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=1536,
                distance=Distance.COSINE
            )
        )
    else:
        print(f"Collection '{COLLECTION_NAME}' already exists.")

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings
    )

    # 5. Insert Chunks with Deterministic IDs
    batch_size = 200
    print(f"Indexing {len(filtered_chunks)} chunks in batches of {batch_size}...")

    for i in range(0, len(filtered_chunks), batch_size):
        batch = filtered_chunks[i:i + batch_size]
        batch_ids = []
        for chunk in batch:
            identity = (
                f"{chunk.metadata.get('source_file', '')}|"
                f"{chunk.metadata.get('page', '')}|"
                f"{chunk.metadata.get('start_index', '')}|"
                f"{chunk.page_content[:100]}"
            )
            batch_ids.append(str(uuid.uuid5(uuid.NAMESPACE_URL, identity)))

        vector_store.add_documents(documents=batch, ids=batch_ids)
        print(f"  Indexed batch {i // batch_size + 1}/{(len(filtered_chunks) + batch_size - 1) // batch_size} ({len(batch)} chunks)")

    total_points = client.count(collection_name=COLLECTION_NAME, exact=True).count
    print(f"\n Successfully indexed {total_points} total points in Qdrant collection '{COLLECTION_NAME}'.")
    return total_points

if __name__ == "__main__":
    index_documents()
