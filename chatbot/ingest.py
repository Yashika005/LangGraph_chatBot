import os
import hashlib
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from chatbot.vectordb import get_vectordb

PDF_DIR = "pdfs"
FOLDER_HASH_FILE = "folder_hash.txt"


def get_pdf_hash(file_path):
    """Generate a unique hash for a PDF file"""
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def get_folder_hash(pdf_hashes: dict):
    """Generate a single hash representing all PDFs in the folder"""
    combined = "".join(pdf_hashes[file] for file in sorted(pdf_hashes))
    return hashlib.md5(combined.encode()).hexdigest()


def load_folder_hash():
    if not os.path.exists(FOLDER_HASH_FILE):
        return None
    with open(FOLDER_HASH_FILE, "r") as f:
        return f.read().strip()


def save_folder_hash(folder_hash):
    with open(FOLDER_HASH_FILE, "w") as f:
        f.write(folder_hash)


def ingest_pdfs_if_needed():
    if not os.path.exists(PDF_DIR):
        print(f" {PDF_DIR} directory not found. Creating it...")
        os.makedirs(PDF_DIR)
        return

    vectordb = get_vectordb()

    # Get current PDF files and their hashes
    current_pdfs = {
        f: get_pdf_hash(os.path.join(PDF_DIR, f))
        for f in os.listdir(PDF_DIR)
        if f.lower().endswith(".pdf")
    }

    if not current_pdfs:
        print(f" No PDF files found in {PDF_DIR}")
        existing = vectordb.get()
        if existing and existing.get("ids"):
            print(" Clearing vector DB as no PDFs exist")
            vectordb.clear()
        return

    # -------- Folder Hash Check (FAST EXIT) --------
    current_folder_hash = get_folder_hash(current_pdfs)
    saved_folder_hash = load_folder_hash()

    if saved_folder_hash == current_folder_hash:
        print(" Folder unchanged. Skipping ingestion.")
        return
    # ----------------------------------------------

    # Get existing metadata from vector DB
    existing_data = vectordb.get()
    existing_metadata = {}

    if existing_data and existing_data.get("metadatas"):
        for i, metadata in enumerate(existing_data["metadatas"]):
            if metadata and "source_file" in metadata:
                source_file = metadata["source_file"]
                file_hash = metadata.get("file_hash", "")
                if source_file not in existing_metadata:
                    existing_metadata[source_file] = {
                        "hash": file_hash,
                        "ids": []
                    }
                existing_metadata[source_file]["ids"].append(existing_data["ids"][i])

    # Find PDFs to add (new or modified)
    pdfs_to_add = []
    for pdf_file, pdf_hash in current_pdfs.items():
        if (
            pdf_file not in existing_metadata
            or existing_metadata[pdf_file]["hash"] != pdf_hash
        ):
            pdfs_to_add.append(pdf_file)
            if pdf_file in existing_metadata:
                print(f" PDF modified, removing old version: {pdf_file}")
                vectordb.delete_by_ids(existing_metadata[pdf_file]["ids"])

    # Find PDFs to remove (deleted from folder)
    pdfs_to_remove = [pdf for pdf in existing_metadata if pdf not in current_pdfs]

    for pdf_file in pdfs_to_remove:
        print(f" Removing deleted PDF: {pdf_file}")
        vectordb.delete_by_ids(existing_metadata[pdf_file]["ids"])

    if not pdfs_to_add and not pdfs_to_remove:
        print(f" Vector DB is up to date with {len(current_pdfs)} PDF(s). No changes needed.")
        save_folder_hash(current_folder_hash)
        return

    # Ingest new / modified PDFs
    if pdfs_to_add:
        print(f" Processing {len(pdfs_to_add)} new/modified PDF(s)...")

        for pdf_file in pdfs_to_add:
            file_path = os.path.join(PDF_DIR, pdf_file)
            print(f"  Loading: {pdf_file}")

            loader = PyPDFLoader(file_path)
            documents = loader.load()

            if not documents:
                print(f" No content loaded from {pdf_file}")
                continue

            print(f" Splitting {len(documents)} pages from {pdf_file}...")

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = splitter.split_documents(documents)

            pdf_hash = current_pdfs[pdf_file]
            for chunk in chunks:
                chunk.metadata["source_file"] = pdf_file
                chunk.metadata["file_hash"] = pdf_hash

            print(f" Adding {len(chunks)} chunks from {pdf_file}...")
            vectordb.add_documents(chunks)

        print(" Ingestion complete!")

    # Save folder hash AFTER successful ingestion
    save_folder_hash(current_folder_hash)
