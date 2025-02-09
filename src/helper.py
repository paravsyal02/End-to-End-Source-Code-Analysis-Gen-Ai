import os
from git import Repo
from langchain.text_splitter import Language
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers.language.language_parser import LanguageParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from dotenv import load_dotenv
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

#clone any github repositories
def repo_ingestion(repo_url):
    os.makedirs("repo", exist_ok=True)
    repo_path = "repo/"
    Repo.clone_from(repo_url, to_path=repo_path)

#Loading Repositories as documents
def load_repo(repo_path):
    loader = GenericLoader.from_filesystem(repo_path,
                                        glob = "**/*",
                                        suffixes=[".py"],
                                        parser = LanguageParser(language=Language.PYTHON, parser_threshold=500)
    )

    documents = loader.load()
    return documents

# Creating Text Chunks
def text_splitter(documents):
    documents_splitter = RecursiveCharacterTextSplitter.from_language(language = Language.PYTHON,
                                                                      chunk_size = 2000,
                                                                      chunk_overlap = 200)

    text_chunks = documents_splitter.split_documents(documents)

    return text_chunks

#loading embedding model
def load_embedding():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")   
    return embeddings                                          