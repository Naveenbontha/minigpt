
import streamlit as st
import sqlite3
import hashlib
import re
import math
import numpy as np

from pathlib import Path
from datetime import datetime
from io import BytesIO
from collections import Counter

from openai import OpenAI
from pypdf import PdfReader
import fitz
import pytesseract
from PIL import Image
from sentence_transformers import SentenceTransformer


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="MiniGPT",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

DB_PATH = Path(__file__).parent / "minigpt.db"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

MAX_MEMORY_MESSAGES = 30

SUMMARY_TRIGGER_MESSAGES = 30
RECENT_MESSAGES_AFTER_SUMMARY = 10
SUMMARY_MAX_TOKENS = 1000

# Stage 12: Advanced RAG
RAG_TOP_K = 8
RAG_SEMANTIC_WEIGHT = 0.70
RAG_KEYWORD_WEIGHT = 0.30
RAG_MIN_SCORE = 0.12
RAG_MAX_CONTEXT_CHARS = 14000
RAG_MAX_CHUNKS_PER_DOCUMENT = 4

OCR_DPI = 200
OCR_LANGUAGE = "eng"

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

DEFAULT_SETTINGS = {
    "system_prompt": """You are MiniGPT, a helpful AI assistant.

Your responsibilities:
1. Give clear and accurate answers.
2. Explain programming concepts in simple English.
3. Provide practical coding examples.
4. Help users learn Python, JavaScript and other technologies.
5. Use proper formatting for code and explanations.
6. Remember the context of the current conversation.""",

    "model": "openai/gpt-oss-20b",
    "temperature": 0.7,
    "max_tokens": 2048
}

MODEL_LABELS = {
    "openai/gpt-oss-20b": "GPT OSS 20B",
    "openai/gpt-oss-120b": "GPT OSS 120B",
    "llama-3.3-70b-versatile": "Llama 3.3 70B",
    "llama-3.1-8b-instant": "Llama 3.1 8B Instant"
}


# ============================================================
# 3. CUSTOM CSS
# ============================================================

st.markdown("""
<style>
.stApp {
    background: #0e1117 !important;
    color: #ffffff !important;
}

.block-container {
    max-width: 1050px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

[data-testid="stSidebar"] {
    background: #161b27 !important;
    border-right: 1px solid #303642;
}

.stApp p,
.stApp span,
.stApp label,
.stApp h1,
.stApp h2,
.stApp h3,
.stApp h4,
.stApp li,
.stApp strong,
.stMarkdown,
[data-testid="stMarkdownContainer"] {
    color: #ffffff !important;
}

.mini-brand {
    font-size: 2rem;
    font-weight: 750;
    letter-spacing: -1px;
    color: #ffffff !important;
}

.mini-subtitle {
    color: #c0c5d0 !important;
    font-size: 0.95rem;
}

.section-label {
    color: #c0c5d0 !important;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.welcome-card {
    background: #1c2332 !important;
    border: 1px solid #343d50;
    border-radius: 18px;
    padding: 28px;
    margin-top: 28px;
    text-align: center;
}

.welcome-card h3,
.welcome-card p {
    color: #ffffff !important;
}

[data-testid="stChatMessage"] {
    background: #161b27 !important;
    border-radius: 12px;
    color: #ffffff !important;
}

[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] h1,
[data-testid="stChatMessage"] h2,
[data-testid="stChatMessage"] h3,
[data-testid="stChatMessage"] strong {
    color: #ffffff !important;
}

.stMarkdown pre {
    background: #161b27 !important;
    border: 1px solid #343d50;
}

.stMarkdown code {
    color: #ffffff !important;
}

.stTextInput input,
.stTextArea textarea,
[data-testid="stChatInput"] textarea {
    background: #1c2332 !important;
    color: #ffffff !important;
    border: 1px solid #414a5b !important;
}

[data-baseweb="select"] {
    background: #1c2332 !important;
    color: #ffffff !important;
}

div.stButton > button {
    background: #252d3d;
    color: #ffffff !important;
    border: 1px solid #414a5b;
    border-radius: 10px;
    font-weight: 600;
}

div.stButton > button:hover {
    background: #35415a;
    border-color: #7184ad;
}

.mini-status {
    background: #173c2c;
    color: #8cf0b7 !important;
    padding: 5px 11px;
    border-radius: 20px;
}

.message-time {
    color: #a0a9ba !important;
    font-size: 0.72rem;
}

.mini-footer {
    text-align: center;
    color: #a0a9ba !important;
    font-size: 0.78rem;
    padding-top: 18px;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# 4. DATABASE
# ============================================================

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def initialize_database():

    with get_connection() as conn:

        conn.execute("""
            CREATE TABLE IF NOT EXISTS chats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(chat_id)
                    REFERENCES chats(id)
                    ON DELETE CASCADE
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS app_settings (
                setting_key TEXT PRIMARY KEY,
                setting_value TEXT NOT NULL
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_summaries (
                chat_id INTEGER PRIMARY KEY,
                summary TEXT NOT NULL,
                summarized_through_id INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(chat_id)
                    REFERENCES chats(id)
                    ON DELETE CASCADE
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_hash TEXT NOT NULL UNIQUE,
                filename TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS document_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                page_number INTEGER,
                embedding BLOB NOT NULL,
                FOREIGN KEY(document_id)
                    REFERENCES documents(id)
                    ON DELETE CASCADE,
                UNIQUE(document_id, chunk_index)
            )
        """)


initialize_database()


# ============================================================
# 5. CHAT DATABASE FUNCTIONS
# ============================================================

def create_chat():

    now = datetime.now().isoformat(timespec="seconds")

    with get_connection() as conn:

        cursor = conn.execute("""
            INSERT INTO chats(title, created_at, updated_at)
            VALUES (?, ?, ?)
        """, ("New Chat", now, now))

        return cursor.lastrowid


def get_all_chats():

    with get_connection() as conn:

        rows = conn.execute("""
            SELECT *
            FROM chats
            ORDER BY updated_at DESC
        """).fetchall()

        return [dict(row) for row in rows]


def get_chat_messages(chat_id):

    with get_connection() as conn:

        rows = conn.execute("""
            SELECT id, role, content, created_at
            FROM messages
            WHERE chat_id = ?
            ORDER BY id ASC
        """, (chat_id,)).fetchall()

        return [dict(row) for row in rows]


def save_message(chat_id, role, content):

    now = datetime.now().isoformat(timespec="seconds")

    with get_connection() as conn:

        conn.execute("""
            INSERT INTO messages(
                chat_id, role, content, created_at
            )
            VALUES (?, ?, ?, ?)
        """, (chat_id, role, content, now))

        conn.execute("""
            UPDATE chats
            SET updated_at = ?
            WHERE id = ?
        """, (now, chat_id))


def rename_chat(chat_id, title):

    now = datetime.now().isoformat(timespec="seconds")

    with get_connection() as conn:

        conn.execute("""
            UPDATE chats
            SET title = ?, updated_at = ?
            WHERE id = ?
        """, (title, now, chat_id))


def delete_chat(chat_id):

    with get_connection() as conn:

        conn.execute(
            "DELETE FROM messages WHERE chat_id = ?",
            (chat_id,)
        )

        conn.execute(
            "DELETE FROM chat_summaries WHERE chat_id = ?",
            (chat_id,)
        )

        conn.execute(
            "DELETE FROM chats WHERE id = ?",
            (chat_id,)
        )


def clear_chat(chat_id):

    now = datetime.now().isoformat(timespec="seconds")

    with get_connection() as conn:

        conn.execute(
            "DELETE FROM messages WHERE chat_id = ?",
            (chat_id,)
        )

        conn.execute(
            "DELETE FROM chat_summaries WHERE chat_id = ?",
            (chat_id,)
        )

        conn.execute("""
            UPDATE chats
            SET title = 'New Chat', updated_at = ?
            WHERE id = ?
        """, (now, chat_id))


def update_chat_title(chat_id, prompt):

    with get_connection() as conn:

        row = conn.execute(
            "SELECT title FROM chats WHERE id = ?",
            (chat_id,)
        ).fetchone()

    if row and row["title"] == "New Chat":

        title = prompt.strip().replace("\n", " ")

        if len(title) > 35:
            title = title[:35] + "..."

        rename_chat(chat_id, title)


# ============================================================
# 6. SETTINGS
# ============================================================

def load_app_settings():

    settings = DEFAULT_SETTINGS.copy()

    with get_connection() as conn:

        rows = conn.execute("""
            SELECT setting_key, setting_value
            FROM app_settings
        """).fetchall()

    for row in rows:

        key = row["setting_key"]
        value = row["setting_value"]

        if key not in settings:
            continue

        try:

            if key == "temperature":
                settings[key] = float(value)

            elif key == "max_tokens":
                settings[key] = int(value)

            else:
                settings[key] = value

        except (ValueError, TypeError):
            pass

    if settings["model"] not in MODEL_LABELS:
        settings["model"] = DEFAULT_SETTINGS["model"]

    return settings


def save_app_settings(settings):

    with get_connection() as conn:

        for key, value in settings.items():

            conn.execute("""
                INSERT OR REPLACE INTO app_settings
                (setting_key, setting_value)
                VALUES (?, ?)
            """, (key, str(value)))


for key, value in load_app_settings().items():

    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# 7. GROQ CLIENT
# ============================================================

api_key = st.secrets.get("GROQ_API_KEY")

if not api_key:
    st.error("Groq API key is missing. Check .streamlit/secrets.toml.")
    st.stop()


client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1"
)


# ============================================================
# 8. SESSION STATE
# ============================================================

if "current_chat_id" not in st.session_state:

    chats = get_all_chats()

    st.session_state.current_chat_id = (
        chats[0]["id"] if chats else create_chat()
    )

if "document_chunks" not in st.session_state:
    st.session_state.document_chunks = []

if "document_signature" not in st.session_state:
    st.session_state.document_signature = None

if "document_names" not in st.session_state:
    st.session_state.document_names = []

if "last_retrieved_sources" not in st.session_state:
    st.session_state.last_retrieved_sources = []

if "settings_saved" not in st.session_state:
    st.session_state.settings_saved = False


# ============================================================
# 9. DOCUMENT TEXT EXTRACTION
# ============================================================

def split_into_chunks(text, filename, page_number=None):

    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):

        end = start + CHUNK_SIZE
        content = text[start:end].strip()

        if content:

            chunks.append({
                "text": content,
                "filename": filename,
                "page": page_number
            })

        if end >= len(text):
            break

        start = end - CHUNK_OVERLAP

    return chunks


def extract_scanned_page(pdf_document, page_number):

    page = pdf_document.load_page(page_number)

    zoom = OCR_DPI / 72
    matrix = fitz.Matrix(zoom, zoom)

    pix = page.get_pixmap(
        matrix=matrix,
        alpha=False
    )

    image = Image.open(
        BytesIO(pix.tobytes("png"))
    ).convert("RGB")

    return pytesseract.image_to_string(
        image,
        lang=OCR_LANGUAGE
    )


def extract_document(uploaded_file):

    filename = uploaded_file.name
    file_bytes = uploaded_file.getvalue()

    extracted_chunks = []

    if filename.lower().endswith(".pdf"):

        try:

            pdf = PdfReader(
                BytesIO(file_bytes),
                strict=False
            )

            if pdf.is_encrypted:

                if pdf.decrypt("") == 0:
                    raise ValueError("This PDF requires a password.")

            with fitz.open(
                stream=file_bytes,
                filetype="pdf"
            ) as ocr_pdf:

                for page_number, page in enumerate(
                    pdf.pages,
                    start=1
                ):

                    page_text = ""

                    try:
                        page_text = (
                            page.extract_text(
                                extraction_mode="layout"
                            ) or ""
                        )

                        if not page_text.strip():
                            page_text = page.extract_text() or ""

                    except Exception:
                        page_text = ""

                    if not page_text.strip():

                        page_text = extract_scanned_page(
                            ocr_pdf,
                            page_number - 1
                        )

                    if page_text.strip():

                        extracted_chunks.extend(
                            split_into_chunks(
                                page_text,
                                filename,
                                page_number
                            )
                        )

        except Exception as error:

            raise ValueError(
                f"Could not read PDF '{filename}': {error}"
            )

    elif filename.lower().endswith(".txt"):

        try:
            text = file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            text = file_bytes.decode("latin-1")

        extracted_chunks.extend(
            split_into_chunks(text, filename)
        )

    return extracted_chunks


# ============================================================
# 10. EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )


def generate_embeddings(texts):

    if not texts:
        return np.empty((0, 384), dtype=np.float32)

    model = load_embedding_model()

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    return np.asarray(
        embeddings,
        dtype=np.float32
    )


# ============================================================
# 11. DOCUMENT STORAGE
# ============================================================

def get_document_id(file_hash):

    with get_connection() as conn:

        row = conn.execute("""
            SELECT id
            FROM documents
            WHERE file_hash = ?
        """, (file_hash,)).fetchone()

    return row["id"] if row else None


def save_document_with_embeddings(
    file_hash,
    filename,
    chunks
):

    if not chunks:
        return

    if get_document_id(file_hash) is not None:
        return

    texts = [chunk["text"] for chunk in chunks]

    embeddings = generate_embeddings(texts)

    now = datetime.now().isoformat(timespec="seconds")

    with get_connection() as conn:

        cursor = conn.execute("""
            INSERT INTO documents(
                file_hash, filename, created_at
            )
            VALUES (?, ?, ?)
        """, (file_hash, filename, now))

        document_id = cursor.lastrowid

        rows = []

        for index, (chunk, embedding) in enumerate(
            zip(chunks, embeddings)
        ):

            rows.append((
                document_id,
                index,
                chunk["text"],
                chunk.get("page"),
                sqlite3.Binary(embedding.tobytes())
            ))

        conn.executemany("""
            INSERT INTO document_chunks(
                document_id,
                chunk_index,
                content,
                page_number,
                embedding
            )
            VALUES (?, ?, ?, ?, ?)
        """, rows)


def get_missing_document_hashes(file_hashes):

    if not file_hashes:
        return []

    placeholders = ",".join("?" for _ in file_hashes)

    with get_connection() as conn:

        rows = conn.execute(f"""
            SELECT file_hash
            FROM documents
            WHERE file_hash IN ({placeholders})
        """, tuple(file_hashes)).fetchall()

    saved_hashes = {
        row["file_hash"] for row in rows
    }

    return [
        file_hash
        for file_hash in file_hashes
        if file_hash not in saved_hashes
    ]


def get_document_chunks(file_hashes):

    if not file_hashes:
        return []

    placeholders = ",".join("?" for _ in file_hashes)

    with get_connection() as conn:

        rows = conn.execute(f"""
            SELECT
                d.file_hash,
                d.filename,
                dc.content,
                dc.page_number,
                dc.embedding
            FROM documents d
            JOIN document_chunks dc
                ON d.id = dc.document_id
            WHERE d.file_hash IN ({placeholders})
            ORDER BY d.id, dc.chunk_index
        """, tuple(file_hashes)).fetchall()

    chunks = []

    for row in rows:

        embedding = np.frombuffer(
            row["embedding"],
            dtype=np.float32
        ).copy()

        chunks.append({
            "text": row["content"],
            "filename": row["filename"],
            "page": row["page_number"],
            "file_hash": row["file_hash"],
            "embedding": embedding
        })

    return chunks


# ============================================================
# 12. STAGE 12 - HYBRID SEARCH UTILITIES
# ============================================================

def tokenize_text(text):

    """
    Convert text into searchable terms.

    This is used for the keyword component of hybrid search.
    """

    return re.findall(
        r"[a-zA-Z0-9_]+",
        text.lower()
    )


def calculate_keyword_score(question, content):

    """
    Calculate a simple normalized keyword overlap score.

    This complements semantic similarity by rewarding
    chunks containing the actual words in the question.
    """

    question_terms = set(tokenize_text(question))
    content_terms = set(tokenize_text(content))

    if not question_terms or not content_terms:
        return 0.0

    overlap = question_terms.intersection(content_terms)

    return len(overlap) / len(question_terms)


def calculate_semantic_score(question_embedding, chunk_embedding):

    """
    Both embeddings are normalized.
    Their dot product is cosine similarity.
    """

    return float(
        np.dot(question_embedding, chunk_embedding)
    )


# ============================================================
# 13. STAGE 12 - HYBRID RETRIEVAL
# ============================================================

def retrieve_relevant_chunks(
    question,
    chunks,
    max_chunks=RAG_TOP_K
):

    """
    Advanced RAG retrieval.

    Combines:
      1. Semantic similarity
      2. Keyword overlap

    Then ranks chunks using a weighted score.
    """

    if not question.strip() or not chunks:
        return []

    question_embedding = generate_embeddings([
        question
    ])[0]

    scored_chunks = []

    for chunk in chunks:

        chunk_embedding = chunk.get("embedding")

        if chunk_embedding is None:
            continue

        semantic_score = calculate_semantic_score(
            question_embedding,
            chunk_embedding
        )

        keyword_score = calculate_keyword_score(
            question,
            chunk["text"]
        )

        hybrid_score = (
            RAG_SEMANTIC_WEIGHT * semantic_score
            + RAG_KEYWORD_WEIGHT * keyword_score
        )

        scored_chunks.append({
            **chunk,
            "semantic_score": semantic_score,
            "keyword_score": keyword_score,
            "hybrid_score": hybrid_score
        })

    scored_chunks.sort(
        key=lambda item: item["hybrid_score"],
        reverse=True
    )

    # Keep a balanced number of chunks from each document.
    selected = []
    document_counts = Counter()

    for chunk in scored_chunks:

        if chunk["hybrid_score"] < RAG_MIN_SCORE:
            continue

        filename = chunk["filename"]

        if document_counts[filename] >= RAG_MAX_CHUNKS_PER_DOCUMENT:
            continue

        selected.append(chunk)
        document_counts[filename] += 1

        if len(selected) >= max_chunks:
            break

    # If threshold filtering removed everything, use the best
    # available results rather than returning no context.
    if not selected and scored_chunks:

        selected = scored_chunks[:max_chunks]

    return selected


# ============================================================
# 14. STAGE 12 - CONTEXT BUILDING AND SOURCE REFERENCES
# ============================================================

def build_document_context(question):

    chunks = st.session_state.document_chunks

    if not chunks:
        st.session_state.last_retrieved_sources = []
        return None

    relevant_chunks = retrieve_relevant_chunks(
        question,
        chunks
    )

    if not relevant_chunks:
        st.session_state.last_retrieved_sources = []
        return None

    context_parts = []
    source_details = []
    used_chars = 0

    for index, chunk in enumerate(
        relevant_chunks,
        start=1
    ):

        filename = chunk["filename"]
        page = chunk.get("page")

        source_name = filename

        if page is not None:
            source_name += f", page {page}"

        source_label = f"[Source {index}]"

        content = chunk["text"]

        remaining_chars = (
            RAG_MAX_CONTEXT_CHARS - used_chars
        )

        if remaining_chars <= 0:
            break

        content = content[:remaining_chars]

        context_parts.append(
            f"""
{source_label}
Document: {source_name}
Semantic score: {chunk['semantic_score']:.4f}
Keyword score: {chunk['keyword_score']:.4f}
Hybrid score: {chunk['hybrid_score']:.4f}

Content:
{content}
"""
        )

        source_details.append({
            "label": source_label,
            "filename": filename,
            "page": page,
            "semantic_score": chunk["semantic_score"],
            "keyword_score": chunk["keyword_score"],
            "hybrid_score": chunk["hybrid_score"],
            "content": content
        })

        used_chars += len(content)

    st.session_state.last_retrieved_sources = source_details

    if not context_parts:
        return None

    context = "\n\n".join(context_parts)

    return f"""
The user has uploaded one or more documents.

The following sections were retrieved from those documents
using hybrid search (semantic similarity and keyword matching).

RETRIEVED DOCUMENT CONTEXT:

{context}

IMPORTANT ANSWERING RULES:

1. Use retrieved document content as the primary source
   when the user asks about the uploaded documents.

2. Cite document-based claims using the exact source labels
   provided above, for example [Source 1].

3. Do not invent facts, quotations, page numbers, or sources.

4. If the retrieved content does not contain the answer,
   clearly say that the uploaded documents do not provide
   enough information.

5. If different documents provide different information,
   explain the difference and cite both sources.

6. You may use general knowledge to explain a concept,
   but distinguish it from information in the documents.

7. Do not cite a source unless its content supports the claim.

8. Use simple English and organize longer answers clearly.
"""


# ============================================================
# 15. CONVERSATION SUMMARY FUNCTIONS
# ============================================================

def get_chat_summary(chat_id):

    with get_connection() as conn:

        row = conn.execute("""
            SELECT summary, summarized_through_id, updated_at
            FROM chat_summaries
            WHERE chat_id = ?
        """, (chat_id,)).fetchone()

    if not row:

        return {
            "summary": "",
            "summarized_through_id": 0,
            "updated_at": None
        }

    return dict(row)


def save_chat_summary(
    chat_id,
    summary,
    summarized_through_id
):

    now = datetime.now().isoformat(timespec="seconds")

    with get_connection() as conn:

        conn.execute("""
            INSERT INTO chat_summaries(
                chat_id,
                summary,
                summarized_through_id,
                updated_at
            )
            VALUES (?, ?, ?, ?)

            ON CONFLICT(chat_id)
            DO UPDATE SET
                summary = excluded.summary,
                summarized_through_id =
                    excluded.summarized_through_id,
                updated_at = excluded.updated_at
        """, (
            chat_id,
            summary,
            summarized_through_id,
            now
        ))


def summarize_conversation(
    chat_id,
    conversation_history,
    existing_summary,
    summarized_through_id
):

    unsummarized = [
        message
        for message in conversation_history
        if message["id"] > summarized_through_id
    ]

    if len(unsummarized) <= SUMMARY_TRIGGER_MESSAGES:
        return existing_summary

    messages_to_summarize = unsummarized[
        :-RECENT_MESSAGES_AFTER_SUMMARY
    ]

    if not messages_to_summarize:
        return existing_summary

    conversation_text = "\n\n".join(
        f"{message['role'].upper()}: {message['content']}"
        for message in messages_to_summarize
    )

    summary_prompt = f"""
Update the existing conversation summary using the new messages.

Preserve:
- Important user requirements and preferences.
- Relevant user-provided information.
- Programming languages and technologies discussed.
- Decisions already made.
- Project status and completed work.
- Unfinished tasks and next steps.
- Important technical details, errors and solutions.

Rules:
- Do not invent facts.
- Do not confuse assistant suggestions with user decisions.
- Do not present unaccepted suggestions as decisions.
- Remove repeated information.
- Preserve important technical names and values.
- Return only the updated summary.

EXISTING SUMMARY:
{existing_summary or "No previous summary exists."}

NEW MESSAGES:
{conversation_text}
"""

    response = client.chat.completions.create(
        model=st.session_state.model,
        messages=[
            {
                "role": "system",
                "content": "Accurately summarize conversations without inventing facts."
            },
            {
                "role": "user",
                "content": summary_prompt
            }
        ],
        temperature=0.2,
        max_completion_tokens=SUMMARY_MAX_TOKENS,
        stream=False
    )

    new_summary = response.choices[0].message.content

    if not new_summary or not new_summary.strip():
        raise ValueError("The model returned an empty summary.")

    new_summary = new_summary.strip()

    last_id = messages_to_summarize[-1]["id"]

    save_chat_summary(
        chat_id,
        new_summary,
        last_id
    )

    return new_summary


# ============================================================
# 16. BUILD MODEL MESSAGES
# ============================================================

def build_model_messages(
    conversation_history,
    system_prompt,
    document_context=None,
    conversation_summary=None,
    summarized_through_id=0
):

    model_messages = [
        {
            "role": "system",
            "content": system_prompt
        }
    ]

    if conversation_summary:

        model_messages.append({
            "role": "system",
            "content": f"""
SUMMARY OF EARLIER CONVERSATION:

{conversation_summary}

Use this summary to understand earlier messages.
Do not treat it as a direct quotation from the user.
"""
        })

    if document_context:

        model_messages.append({
            "role": "system",
            "content": document_context
        })

    recent_messages = [
        message
        for message in conversation_history
        if message["id"] > summarized_through_id
    ][-MAX_MEMORY_MESSAGES:]

    for message in recent_messages:

        role = message["role"]
        content = message["content"]

        if role not in ("user", "assistant"):
            continue

        if not isinstance(content, str) or not content.strip():
            continue

        model_messages.append({
            "role": role,
            "content": content
        })

    return model_messages


# ============================================================
# 17. STREAMING RESPONSE
# ============================================================

def generate_response(model_messages):

    stream = client.chat.completions.create(
        model=st.session_state.model,
        messages=model_messages,
        temperature=st.session_state.temperature,
        max_completion_tokens=st.session_state.max_tokens,
        stream=True
    )

    for chunk in stream:

        if not chunk.choices:
            continue

        content = chunk.choices[0].delta.content

        if content:
            yield content


# ============================================================
# 18. SIDEBAR - BRAND AND NEW CHAT
# ============================================================

with st.sidebar:

    st.markdown(
        '<div class="mini-brand">🤖 MiniGPT</div>',
        unsafe_allow_html=True
    )

    st.caption("Your personal AI workspace")

    st.markdown(
        '<span class="mini-status">● Groq connected</span>',
        unsafe_allow_html=True
    )

    st.divider()

    if st.button(
        "＋ New chat",
        use_container_width=True,
        type="primary"
    ):

        st.session_state.current_chat_id = create_chat()
        st.rerun()


# ============================================================
# 19. SIDEBAR - CHAT HISTORY
# ============================================================

chats = get_all_chats()

chat_ids = [chat["id"] for chat in chats]

chat_titles = {
    chat["id"]: chat["title"]
    for chat in chats
}

if st.session_state.current_chat_id not in chat_ids:

    st.session_state.current_chat_id = (
        chats[0]["id"] if chats else create_chat()
    )

    st.rerun()


with st.sidebar:

    st.markdown(
        '<div class="section-label">Your conversations</div>',
        unsafe_allow_html=True
    )

    selected_chat_id = st.selectbox(
        "Choose a conversation",
        options=chat_ids,
        index=chat_ids.index(
            st.session_state.current_chat_id
        ),
        format_func=lambda chat_id: chat_titles[chat_id],
        label_visibility="collapsed",
        key="chat_selector"
    )

    st.session_state.current_chat_id = selected_chat_id

    with st.expander("✏️ Rename chat"):

        rename_value = st.text_input(
            "New conversation title",
            key="rename_title"
        )

        if st.button("Save title", use_container_width=True):

            if rename_value.strip():

                rename_chat(
                    selected_chat_id,
                    rename_value.strip()
                )

                st.rerun()

            else:
                st.warning("Enter a title first.")

    with st.expander("🗑️ Delete chat"):

        st.warning(
            "This permanently deletes this conversation and its messages."
        )

        confirm_delete = st.checkbox(
            "I understand this cannot be undone",
            key="confirm_delete_chat"
        )

        if st.button(
            "Delete this chat",
            disabled=not confirm_delete,
            use_container_width=True
        ):

            delete_chat(selected_chat_id)

            remaining = get_all_chats()

            st.session_state.current_chat_id = (
                remaining[0]["id"] if remaining else create_chat()
            )

            st.rerun()


# ============================================================
# 20. SIDEBAR - AI SETTINGS
# ============================================================

with st.sidebar:

    with st.expander("⚙️ AI settings"):

        if st.button(
            "Reset to defaults",
            key="reset_settings",
            use_container_width=True
        ):

            for key, value in DEFAULT_SETTINGS.items():
                st.session_state[key] = value

            save_app_settings(DEFAULT_SETTINGS)
            st.session_state.settings_saved = True

            st.rerun()

        st.selectbox(
            "AI model",
            options=list(MODEL_LABELS.keys()),
            format_func=lambda model_id: MODEL_LABELS[model_id],
            key="model"
        )

        st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            step=0.1,
            key="temperature"
        )

        st.slider(
            "Maximum response tokens",
            min_value=256,
            max_value=8192,
            step=256,
            key="max_tokens"
        )

        st.text_area(
            "System prompt",
            key="system_prompt",
            height=200
        )

        if st.button(
            "Save settings",
            type="primary",
            use_container_width=True
        ):

            current_settings = {
                "system_prompt": st.session_state.system_prompt,
                "model": st.session_state.model,
                "temperature": st.session_state.temperature,
                "max_tokens": st.session_state.max_tokens
            }

            save_app_settings(current_settings)

            st.session_state.settings_saved = True
            st.rerun()

        if st.session_state.settings_saved:
            st.success("Settings saved.")

    st.divider()

    st.caption(
        f"Model: {MODEL_LABELS[st.session_state.model]}"
    )

    st.caption(
        f"Total conversations: {len(get_all_chats())}"
    )


# ============================================================
# 21. SIDEBAR - MULTI-DOCUMENT UPLOAD
# ============================================================

with st.sidebar:

    st.divider()

    st.markdown("### 📄 Chat with documents")

    st.caption(
        "Upload multiple PDF and TXT files. "
        "MiniGPT will search across all selected documents."
    )

    uploaded_files = st.file_uploader(
        "Upload your documents",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        key="document_uploader"
    )

    if uploaded_files:

        file_hashes = {}

        for uploaded_file in uploaded_files:

            file_hash = hashlib.sha256(
                uploaded_file.getvalue()
            ).hexdigest()

            file_hashes[file_hash] = uploaded_file

        current_signature = hashlib.sha256(
            "|".join(sorted(file_hashes.keys())).encode("utf-8")
        ).hexdigest()

        if current_signature != st.session_state.document_signature:

            missing_hashes = get_missing_document_hashes(
                list(file_hashes.keys())
            )

            if missing_hashes:

                st.info(
                    "Processing new documents and generating embeddings..."
                )

                progress = st.progress(0)

                for index, file_hash in enumerate(missing_hashes):

                    uploaded_file = file_hashes[file_hash]

                    try:

                        file_chunks = extract_document(uploaded_file)

                        if file_chunks:

                            save_document_with_embeddings(
                                file_hash,
                                uploaded_file.name,
                                file_chunks
                            )

                        else:

                            st.warning(
                                f"No text extracted from {uploaded_file.name}."
                            )

                    except Exception as error:

                        st.error(
                            f"Could not process {uploaded_file.name}: {error}"
                        )

                    progress.progress(
                        (index + 1) / len(missing_hashes)
                    )

                progress.empty()

            st.session_state.document_chunks = get_document_chunks(
                list(file_hashes.keys())
            )

            st.session_state.document_names = list(
                dict.fromkeys(
                    chunk["filename"]
                    for chunk in st.session_state.document_chunks
                )
            )

            st.session_state.document_signature = current_signature

            st.rerun()

    else:

        # Remove documents from the current session when uploader is empty.
        st.session_state.document_chunks = []
        st.session_state.document_names = []
        st.session_state.document_signature = None

    if st.session_state.document_chunks:

        st.success(
            f"{len(st.session_state.document_chunks)} document sections loaded."
        )

        for filename in st.session_state.document_names:
            st.write(f"📄 {filename}")

        if st.button(
            "Clear current documents",
            use_container_width=True
        ):

            st.session_state.document_chunks = []
            st.session_state.document_names = []

            # Persistent database records are not deleted.
            st.session_state.document_signature = (
                st.session_state.document_signature
            )

            st.rerun()

    else:

        st.info("No documents uploaded yet.")


# ============================================================
# 22. MAIN HEADER
# ============================================================

current_chat_id = st.session_state.current_chat_id

current_chat_title = next(
    (
        chat["title"]
        for chat in get_all_chats()
        if chat["id"] == current_chat_id
    ),
    "New Chat"
)

st.markdown(
    '<div class="mini-brand">MiniGPT</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="mini-subtitle">'
    'A simple AI workspace powered by Groq'
    '</div>',
    unsafe_allow_html=True
)

header_left, header_right = st.columns([7, 2])

with header_left:
    st.caption(f"Current conversation: **{current_chat_title}**")

with header_right:
    st.caption(f"Model: {MODEL_LABELS[st.session_state.model]}")


# ============================================================
# 23. DISPLAY CHAT HISTORY
# ============================================================

messages = get_chat_messages(current_chat_id)

if not messages:

    st.markdown("""
    <div class="welcome-card">
        <h3>👋 Welcome to MiniGPT</h3>
        <p>Ask a question, explore an idea, or get help with coding.</p>
    </div>
    """, unsafe_allow_html=True)

    st.write("**Try asking:**")

    columns = st.columns(3)

    suggestions = [
        "Explain Python functions",
        "What is REST API?",
        "Give me a SQL practice question"
    ]

    for column, suggestion in zip(columns, suggestions):

        with column:

            if st.button(
                suggestion,
                use_container_width=True
            ):

                st.session_state.pending_prompt = suggestion
                st.rerun()

else:

    for message in messages:

        avatar = (
            "🧑‍💻"
            if message["role"] == "user"
            else "🤖"
        )

        with st.chat_message(
            message["role"],
            avatar=avatar
        ):

            st.markdown(message["content"])

            if message.get("created_at"):

                try:

                    timestamp = datetime.fromisoformat(
                        message["created_at"]
                    ).strftime("%d %b %Y, %I:%M %p")

                    st.markdown(
                        f'<div class="message-time">{timestamp}</div>',
                        unsafe_allow_html=True
                    )

                except (ValueError, TypeError):
                    pass


# ============================================================
# 24. EXPORT AND CLEAR CHAT
# ============================================================

if messages:

    st.write("")

    action_col1, action_col2, action_col3 = st.columns([1, 1, 4])

    export_lines = [
        f"# {current_chat_title}",
        "",
        f"Exported: {datetime.now().strftime('%d %b %Y, %I:%M %p')}",
        ""
    ]

    for message in messages:

        role_name = (
            "You" if message["role"] == "user" else "MiniGPT"
        )

        export_lines.extend([
            f"## {role_name}",
            "",
            message["content"],
            "",
            "---",
            ""
        ])

    with action_col1:

        st.download_button(
            "⬇️ Export Markdown",
            data="\n".join(export_lines),
            file_name="minigpt_conversation.md",
            mime="text/markdown",
            use_container_width=True
        )

    with action_col2:

        plain_text = "\n\n".join(
            (
                "You" if message["role"] == "user" else "MiniGPT"
            )
            + ":\n"
            + message["content"]
            for message in messages
        )

        st.download_button(
            "⬇️ Export TXT",
            data=plain_text,
            file_name="minigpt_conversation.txt",
            mime="text/plain",
            use_container_width=True
        )

    with action_col3:

        with st.popover("🧹 Clear conversation"):

            st.warning(
                "This removes all messages from this conversation."
            )

            confirm_clear = st.checkbox(
                "I want to clear these messages",
                key=f"confirm_clear_{current_chat_id}"
            )

            if st.button(
                "Confirm clear",
                disabled=not confirm_clear,
                type="primary",
                use_container_width=True
            ):

                clear_chat(current_chat_id)
                st.rerun()


# ============================================================
# 25. CHAT INPUT AND ADVANCED RAG
# ============================================================

typed_prompt = st.chat_input("Message MiniGPT...")

pending_prompt = st.session_state.pop(
    "pending_prompt",
    None
)

prompt = typed_prompt or pending_prompt

if prompt:

    # Save the user message.
    save_message(
        current_chat_id,
        "user",
        prompt
    )

    update_chat_title(
        current_chat_id,
        prompt
    )

    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    conversation_history = get_chat_messages(
        current_chat_id
    )

    # Load conversation summary.
    summary_data = get_chat_summary(current_chat_id)

    conversation_summary = summary_data["summary"]

    summarized_through_id = summary_data[
        "summarized_through_id"
    ]

    # Update summary if necessary.
    unsummarized_messages = [
        message
        for message in conversation_history
        if message["id"] > summarized_through_id
    ]

    if len(unsummarized_messages) > SUMMARY_TRIGGER_MESSAGES:

        try:

            with st.spinner("Updating conversation memory..."):

                conversation_summary = summarize_conversation(
                    current_chat_id,
                    conversation_history,
                    conversation_summary,
                    summarized_through_id
                )

                summary_data = get_chat_summary(
                    current_chat_id
                )

                summarized_through_id = summary_data[
                    "summarized_through_id"
                ]

                st.toast("Conversation memory updated.", icon="🧠")

        except Exception as error:

            st.warning(
                f"Conversation summarization failed: {error}"
            )

    # Stage 12: Hybrid document retrieval.
    document_context = None

    st.session_state.last_retrieved_sources = []

    if st.session_state.document_chunks:

        try:

            with st.spinner(
                "Searching documents using hybrid RAG..."
            ):

                document_context = build_document_context(
                    prompt
                )

        except Exception as error:

            st.warning(
                f"Document retrieval failed: {error}"
            )

    # Build model messages.
    model_messages = build_model_messages(
        conversation_history=conversation_history,
        system_prompt=st.session_state.system_prompt,
        document_context=document_context,
        conversation_summary=conversation_summary,
        summarized_through_id=summarized_through_id
    )

    # Generate the response.
    with st.chat_message("assistant", avatar="🤖"):

        try:

            full_response = st.write_stream(
                generate_response(model_messages),
                cursor="▌"
            )

            if full_response and full_response.strip():

                save_message(
                    current_chat_id,
                    "assistant",
                    full_response
                )

        except Exception as error:

            st.error(f"Something went wrong: {error}")


# ============================================================
# 26. DISPLAY RETRIEVED SOURCES
# ============================================================

sources = st.session_state.last_retrieved_sources

if sources:

    with st.expander(
        f"📚 Retrieved sources ({len(sources)})",
        expanded=False
    ):

        st.caption(
            "These are the document sections retrieved for the latest question."
        )

        for source in sources:

            source_title = source["label"]

            if source["page"] is not None:
                source_title += (
                    f" — {source['filename']}, "
                    f"page {source['page']}"
                )
            else:
                source_title += f" — {source['filename']}"

            st.markdown(f"**{source_title}**")

            st.caption(
                f"Hybrid: {source['hybrid_score']:.4f} | "
                f"Semantic: {source['semantic_score']:.4f} | "
                f"Keyword: {source['keyword_score']:.4f}"
            )

            st.write(source["content"])

            st.divider()


# ============================================================
# 27. CONVERSATION MEMORY STATUS
# ============================================================

summary_data = get_chat_summary(current_chat_id)

if summary_data["summary"]:

    with st.expander("🧠 Conversation memory"):

        st.caption(
            "This summary helps MiniGPT remember earlier messages."
        )

        st.markdown(summary_data["summary"])

        st.caption(
            f"Last updated: {summary_data['updated_at']}"
        )

else:

    st.caption(
        "🧠 Conversation summary will be created "
        "after enough messages accumulate."
    )


# ============================================================
# 28. STAGE 12 - RAG STATUS
# ============================================================

if st.session_state.document_chunks:

    with st.expander("🔎 Advanced RAG information"):

        st.write(
            "**Embedding model:**",
            EMBEDDING_MODEL_NAME
        )

        st.write(
            "**Loaded document chunks:**",
            len(st.session_state.document_chunks)
        )

        st.write(
            "**Retrieval method:**",
            "Hybrid semantic + keyword search"
        )

        st.write(
            "**Semantic weight:**",
            RAG_SEMANTIC_WEIGHT
        )

        st.write(
            "**Keyword weight:**",
            RAG_KEYWORD_WEIGHT
        )

        st.write(
            "**Maximum retrieved chunks:**",
            RAG_TOP_K
        )

        st.write(
            "**Maximum context characters:**",
            RAG_MAX_CONTEXT_CHARS
        )

        st.caption(
            "MiniGPT combines semantic similarity and keyword "
            "overlap to select relevant sections from uploaded documents."
        )


# ============================================================
# 29. FOOTER
# ============================================================

st.markdown("""
<div class="mini-footer">
    MiniGPT · Streamlit · Groq · SQLite · Tesseract OCR
    · Sentence Transformers · Hybrid RAG · Stage 12
</div>
""", unsafe_allow_html=True)
