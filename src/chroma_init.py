import os
import hashlib
import shutil
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
import json
import os.path
from src.models import EMBEDDING_MODEL

from dotenv import load_dotenv

load_dotenv()

def calculate_file_hash(file_path):
    """Calculate MD5 hash of file content."""
    with open(file_path, 'rb') as f:
        file_data = f.read()
        return hashlib.md5(file_data).hexdigest()

def get_hash_file_path(persist_directory):
    """Get path to the hash file."""
    hash_dir = os.path.join(persist_directory, "hashes")
    os.makedirs(hash_dir, exist_ok=True)
    return os.path.join(hash_dir, "files.hash")

def save_hashes(files_hashes, persist_directory):
    """Save hashes to a file."""
    hash_file = get_hash_file_path(persist_directory)
    with open(hash_file, 'w') as f:
        json.dump(files_hashes, f)

def get_saved_hashes(persist_directory):
    """Get saved hashes if they exist."""
    hash_file = get_hash_file_path(persist_directory)
    if os.path.exists(hash_file):
        with open(hash_file, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}

def get_current_files_hashes(data_dir):
    """Get hashes of all current text files."""
    current_hashes = {}
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                current_hashes[file_path] = calculate_file_hash(file_path)
    return current_hashes

def has_files_changed(current_hashes, saved_hashes):
    """Check if any files have been added, modified, or deleted."""
    # Check if number of files changed
    if len(current_hashes) != len(saved_hashes):
        return True
    
    # Check if any file has been modified
    for file_path, file_hash in current_hashes.items():
        if file_path not in saved_hashes or saved_hashes[file_path] != file_hash:
            return True
    
    # Check if any file has been removed
    for file_path in saved_hashes:
        if file_path not in current_hashes:
            return True
    
    return False

def init_chroma() -> None:
    """
    Finds all .txt files in the data directory, splits them into chunks,
    and saves them in a single persistent ChromaDB collection.
    """
    # Set directory where to store ChromaDB main collection
    persist_directory = "src/chroma_db"
    
    # Set chunk storage characteristics
    chunk_size = 500
    chunk_overlap = 100
    
    # Set data directory where to find text files
    data_dir = 'data'

    # Set collection name where to store chunks of data with embeddings
    collection_name = "text_documents"
    
    # Find all .txt files in the data directory
    txt_files = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".txt"):
                txt_files.append(os.path.join(root, file))
    
    if not txt_files:
        print(f"No .txt files found in '{data_dir}'")
        return
    
    print(f"Found {len(txt_files)} .txt files")
    
    # Get current file hashes
    current_hashes = get_current_files_hashes(data_dir)
    
    # Get saved hashes
    saved_hashes = get_saved_hashes(persist_directory)
    
    # Check if files have changed
    files_changed = has_files_changed(current_hashes, saved_hashes)
    
    # If no changes, we can skip processing
    if not files_changed and saved_hashes:
        print(f"No changes detected in text files. Using existing collection '{collection_name}'.")
        return
    
    print("Changes detected in text files. Reinitializing collection.")
    
    # Create text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    
    # Clean up existing ChromaDB directory while preserving the hashes
    if os.path.exists(persist_directory):
        print("Cleaning up old collections...")
        # Save hashes directory
        hash_dir = os.path.join(persist_directory, "hashes")
        hash_backup = None
        if os.path.exists(hash_dir):
            hash_backup = os.path.join(os.path.dirname(persist_directory), "hash_backup_temp")
            if os.path.exists(hash_backup):
                shutil.rmtree(hash_backup)
            shutil.copytree(hash_dir, hash_backup)
        
        # Delete everything in the ChromaDB directory except hashes
        for item in os.listdir(persist_directory):
            item_path = os.path.join(persist_directory, item)
            if item != "hashes":
                if os.path.isfile(item_path):
                    os.unlink(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
        
        # Restore hashes if needed
        if hash_backup:
            if os.path.exists(hash_dir):
                shutil.rmtree(hash_dir)
            shutil.copytree(hash_backup, hash_dir)
            shutil.rmtree(hash_backup)
    
    # Process all files and gather chunks
    all_chunks = []
    
    for file_path in txt_files:
        file_name = os.path.basename(file_path)
        try:
            # Load document
            loader = TextLoader(file_path)
            documents = loader.load()
            
            # Add source information to metadata
            for doc in documents:
                doc.metadata["source"] = file_path
                doc.metadata["filename"] = file_name
            
            # Split text into chunks
            chunks = text_splitter.split_documents(documents)
            all_chunks.extend(chunks)
            
            print(f"Processed '{file_name}' into {len(chunks)} chunks")
                     
        except Exception as e:
            print(f"Error processing file '{file_name}': {str(e)}")
    
    if all_chunks:
        # Create a new Chroma collection with all chunks
        db = Chroma.from_documents(
            documents=all_chunks,
            embedding=EMBEDDING_MODEL,
            collection_name=collection_name,
            persist_directory=persist_directory
        )
        
        # Save the hashes of all processed files
        save_hashes(current_hashes, persist_directory)
        
        print(f"Created collection '{collection_name}' with {len(all_chunks)} total chunks from {len(txt_files)} files")
    else:
        print("No documents were processed successfully.")
    
    print("Processing complete!")
