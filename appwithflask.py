from langchain_community.vectorstores import Chroma
from src.helper import load_embedding, repo_ingestion
from dotenv import load_dotenv
import os
from flask import Flask, render_template, jsonify, request
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain

app = Flask(__name__)

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

embeddings = load_embedding()
persist_directory = "db"
# Load the persisted database from disk
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)
memory = ConversationSummaryMemory(llm=llm, memory_key="chat_history", return_messages=True)
qa = ConversationalRetrievalChain.from_llm(llm, retriever=vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 8}), memory=memory)

@app.route('/', methods=["GET", "POST"])
def index():
    return render_template('index.html')

@app.route('/chatbot', methods=["POST"])
def gitRepo():
    user_input = request.form['question']
    repo_ingestion(user_input)
    os.system("python store_index.py")
    return jsonify({"response": str(user_input)})

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]
    if msg.lower() == "clear":
        os.system("rm -rf repo")
        return "Chat history cleared."
    
    result = qa({"question": msg, "chat_history": []})
    return str(result["answer"])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
