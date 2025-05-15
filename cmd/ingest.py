import os
import subprocess

from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

REPO_URL = "https://github.com/openshift-pipelines/pipelines-as-code.git"
REPO_DIR = "repo2"


def clone_repo():
    if not os.path.exists(REPO_DIR):
        subprocess.run(["git", "clone", REPO_URL, REPO_DIR])
    else:
        print("Repo already cloned.")


def ingest_doc(index_path: str, splitter, embeddings):
    loader = DirectoryLoader(REPO_DIR, glob="**/*.md")
    document = loader.load()
    text = splitter.split_documents(document)
    print("Embedding...", text.metadata)
    # vectorstore = FAISS.from_documents(text, embeddings)
    # vectorstore.save_local(index_path)
    # print("Vectorstore saved.")


def ingest_folder(folder_path: str, index_path: str, embeddings):
    """Processes files, creates embeddings, saves FAISS index."""
    print(f"Ingesting {folder_path}...")
    documents = load_docs_and_code()

    print("Splitting text...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = splitter.split_documents(documents)
    print("Embedding...")
    vectorstore = FAISS.from_documents(texts, embeddings)
    vectorstore.save_local(index_path)
    print("Vectorstore saved.")


def ingest(embedding_model_name, folder, faiss_index_path):
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    ingest_folder(folder, faiss_index_path, embeddings)
    return None


def load_docs_and_code():
    loaders = []

    # Load markdown docs (README + /docs)
    loaders.append(DirectoryLoader(REPO_DIR, glob="**/*.md"))

    # Load code files (Python, Go, YAML, etc.)
    for ext in ["*.py", "*.go", "*.yaml", "*.yml", "*.sh"]:
        loaders.append(DirectoryLoader(REPO_DIR, glob=f"**/{ext}"))

    documents = []
    for loader in loaders:
        print(f"Loading {loader.glob}...")
        documents.extend(loader.load())

    return documents


if __name__ == "__main__":
    ingest_doc()
    # clone_repo()
    # ingest("sentence-transformers/all-mpnet-base-v2", REPO_DIR, "faiss_index_2")
    # create_vector_store()
