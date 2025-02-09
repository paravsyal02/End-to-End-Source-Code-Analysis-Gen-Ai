import os
import shutil
import stat
from dotenv import load_dotenv
from git import Repo
from langchain.text_splitter import Language
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import streamlit as st

# Load environment variables
load_dotenv()

# Define the path for the repository and the database
repo_path = "test_repo/"
db_path = 'db'

# Function to forcefully delete files
def force_delete(action, name, exc):
    os.chmod(name, stat.S_IWRITE)
    os.remove(name)

def delete_repo_and_db():
    try:
        if os.path.exists(repo_path):
            shutil.rmtree(repo_path, onerror=force_delete)
        if os.path.exists(db_path):
            shutil.rmtree(db_path, onerror=force_delete)
        st.success("Repository and database deleted successfully!")
    except Exception as e:
        st.error(f"Error deleting files: {str(e)}")

# Streamlit interface for adding a repository URL
st.title("Chatbot - Ask Questions from the Codebase")

# Button to delete repository and database
if st.button("Delete Repository & DB First before giving any Repo"):
    delete_repo_and_db()

# Input for the repository URL
repo_url = st.text_input("Enter the GitHub repository URL:")

# If a URL is provided
if repo_url:
    # Clone the repository if it doesn't exist
    if not os.path.exists(repo_path):
        try:
            repo = Repo.clone_from(repo_url, to_path=repo_path)
            st.success(f"Repository cloned from {repo_url}")
        except Exception as e:
            st.error(f"Failed to clone the repository: {str(e)}")
    else:
        repo = Repo(repo_path)
        st.success(f"Repository already exists at {repo_path}")

    # Load the documents from the repo
    loader = GenericLoader.from_filesystem(repo_path,
                                           glob="**/*",
                                           suffixes=[".py"],
                                           parser=LanguageParser(language=Language.PYTHON, parser_threshold=500))
    documents = loader.load()

    # Split the documents into chunks
    documents_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.PYTHON,
                                                                      chunk_size=500,
                                                                      chunk_overlap=20)
    texts = documents_splitter.split_documents(documents)

    # Set up embeddings and vector store
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectordb = Chroma.from_documents(texts, embedding=embeddings, persist_directory=db_path)

    # Set up the LLM and memory for conversation
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa = ConversationalRetrievalChain.from_llm(llm, retriever=vectordb.as_retriever(search_type="mmr", search_kwags={"k": 8}, memory=memory))

    # User input for the question
    question = st.text_input("Ask a question about the codebase:")

    if question:
        # Fetch the answer from the QA system
        result = qa({"question": question, "chat_history": []})
        
        # Display the result
        st.write(f"Answer: {result['answer']}")
