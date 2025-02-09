# End-to-End-Source-Code-Analysis-Gen-Ai
# Chatbot - Flask and Streamlit Implementation

This project implements a medical chatbot using Flask with modular coding and Streamlit without modular coding. It utilizes LangChain, Google Generative AI, and Chroma for conversational retrieval.

## Features
- Flask-based chatbot with modular structure
- Streamlit-based chatbot without modular coding
- Uses LangChain for conversational retrieval
- Embeddings stored in Chroma DB
- Google Generative AI integration
- Supports repository ingestion for codebase-related queries

## Setup Instructions

### 1. Create and Activate a Virtual Environment

#### Windows (cmd or PowerShell)
```sh
python -m venv venv
venv\Scripts\activate
```

#### macOS/Linux
```sh
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
```sh
pip install -r requirements.txt
```

### 3. Set Up Environment Variables
Create a `.env` file and add your Google API Key:
```sh
GOOGLE_API_KEY=your_google_api_key_here
```

### 4. Run Flask Application
```sh
python app.py
```
The Flask app will run on `http://0.0.0.0:8080`.

### 5. Run Streamlit Application (For Non-Modular Approach)
```sh
streamlit run streamlit_app.py
```

### 6. Clear Repository & DB
To delete the `test_repo` and `db` directories forcefully:
- In Flask, send a request with `clear` as input.
- In Streamlit, press the delete button.

## Dependencies
- Flask
- Streamlit
- LangChain
- ChromaDB
- Google Generative AI SDK
- Python-dotenv

## Notes
- Ensure you have the correct API key before running the application.
- The Streamlit app does not follow a modular approach, whereas the Flask app is structured with separate helper functions.
- Use the appropriate command to run the desired application.

---
Developed forchatbot interactions using AI-powered responses. 🚀

