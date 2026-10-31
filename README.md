# MiniGPT – AI-Powered Document Chatbot

MiniGPT is an AI-powered conversational chatbot built with Python and Streamlit. It allows users to interact with AI, upload PDF and TXT documents, and ask questions based on their uploaded content using Retrieval-Augmented Generation (RAG).

## 🚀 Live Demo

**[Try MiniGPT](https://minigpt-vjwyb73qodubuddqfn53hv.streamlit.app/)**

## ✨ Features

- AI-powered conversational chatbot.
- Streaming AI responses.
- Multiple conversation management.
- Persistent chat history using SQLite.
- Customizable AI settings and model selection.
- PDF and TXT document uploads.
- OCR support for scanned PDF documents.
- Document-based question answering.
- Semantic search using sentence embeddings.
- Hybrid document retrieval.
- Source-based answers and document citations.
- Conversation memory.
- Automatic conversation summarization.
- Export conversations.
- Dark-themed user interface.

## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Frontend | Streamlit |
| LLM Integration | Groq API, OpenAI-compatible SDK |
| Database | SQLite |
| Embeddings | Sentence Transformers |
| Semantic Search | NumPy |
| PDF Processing | PyMuPDF, PyPDF |
| OCR | Tesseract, Pytesseract |
| Image Processing | Pillow |
| Version Control | Git, GitHub |

## 🧠 How It Works

1. Users enter a question in the chatbot.
2. MiniGPT sends the conversation to the configured language model.
3. Users can upload PDF or TXT documents.
4. The application extracts document text and performs OCR when needed.
5. Text is divided into smaller chunks.
6. Sentence Transformers generates embeddings for the chunks.
7. Relevant document passages are retrieved through semantic and hybrid search.
8. Retrieved context is passed to the language model.
9. The chatbot generates an answer using the relevant document content.
10. Conversations and document information are stored in SQLite.

## 📂 Project Structure

```text
MiniGPT/
│
├── app.py
├── requirements.txt
├── packages.txt
├── .gitignore
├── minigpt.db
│
└── .streamlit/
    └── secrets.toml
```

**Note:** Do not commit `minigpt.db`, `secrets.toml`, or your virtual environment if they contain private or local data.

## ⚙️ Run Locally

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd MiniGPT
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure API credentials

Create `.streamlit/secrets.toml`:

```toml
GROQ_API_KEY = "YOUR_GROQ_API_KEY"
```

Never commit your API key to GitHub.

### 6. Run the application

```bash
python -m streamlit run app.py
```

## 🌐 Deployment

MiniGPT is deployed using Streamlit Community Cloud.

**Live Application:**  
https://minigpt-vjwyb73qodubuddqfn53hv.streamlit.app/

## 🔮 Future Improvements

- PostgreSQL integration for production-ready persistent storage.
- User authentication and individual chat workspaces.
- Improved document retrieval and reranking.
- Support for additional document formats.
- More advanced RAG evaluation and response-quality metrics.

## 👨‍💻 Author

**Naveen Bontha**

Integrated M.Tech – Software Engineering

- GitHub: https://github.com/Naveenbontha
- LinkedIn: https://www.linkedin.com/in/naveen-bontha/

---

⭐ If you find this project useful, consider giving the repository a star.