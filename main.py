# main.py
import os
import requests
import PyPDF2
import io
from google import genai
from google.genai import types
from typing import List, Optional, Tuple
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from dotenv import load_dotenv
from docx import Document as DocxDocument

load_dotenv()

_fallback_vocab: Optional[dict] = None
_fallback_tf_matrix: Optional[np.ndarray] = None

def create_embeddings(text: str) -> tuple:
    global _fallback_vocab, _fallback_tf_matrix  # must be first line inside function
    
    chunks = _chunk_text(text)
    if not chunks:
        raise HTTPException(status_code=400, detail="No readable text found in the document")

    try:
        embeddings = embed_texts(chunks, task_type="RETRIEVAL_DOCUMENT", output_dim=768)
        _fallback_vocab, _fallback_tf_matrix = None, None
        return embeddings.astype('float32'), None, chunks
    except Exception:
        vocab, tf = _build_fallback_tf(chunks)
        _fallback_vocab, _fallback_tf_matrix = vocab, tf
        return tf.astype('float32'), None, chunks


app = FastAPI(
    title="HackRx 6.0 - LLM Query Retrieval System",
    description="Intelligent document processing and query system for insurance, legal, HR, and compliance domains",
    version="1.0.0"
)

# Global Gemini client (created lazily)
client: Optional[genai.Client] = None

# CORS (allow all by default; tighten for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()

def get_gemini_client() -> genai.Client:
    """Return a configured Gemini client (lazy init)."""
    global client
    if client is not None:
        return client

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not found in environment variables")

    # Try to configure the genai library; fallback to passing api_key to Client constructor
    try:
        genai.configure(api_key=api_key)
        client = genai.Client()
    except Exception:
        # Some versions allow constructing with api_key directly
        client = genai.Client(api_key=api_key)
    return client

def embed_texts(texts: List[str], task_type: str = "RETRIEVAL_DOCUMENT", output_dim: int = 768) -> np.ndarray:
    """
    Embed a list of texts using Gemini embedding model (gemini-embedding-001).
    Returns a numpy array of shape (len(texts), dim) with L2-normalized rows.
    """
    if not texts:
        return np.zeros((0, output_dim), dtype=np.float32)

    gclient = get_gemini_client()
    try:
        result = gclient.models.embed_content(
            model="gemini-embedding-001",
            contents=texts,
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=output_dim
            )
        )
    except Exception as e:
        raise RuntimeError(f"Embedding generation failed (API): {e}")

    # Robust extraction of embedding vectors
    embeddings = []
    try:
        for emb_obj in result.embeddings:
            # prefer .values attribute
            if hasattr(emb_obj, "values"):
                vals = emb_obj.values
            elif isinstance(emb_obj, dict) and "values" in emb_obj:
                vals = emb_obj["values"]
            elif isinstance(emb_obj, dict) and "embedding" in emb_obj:
                vals = emb_obj["embedding"]
            else:
                # last resort - try to stringify
                raise RuntimeError("Unexpected embedding object format")
            embeddings.append(np.array(vals, dtype=np.float32))
    except Exception as e:
        raise RuntimeError(f"Failed to parse embeddings from response: {e}")

    arr = np.vstack(embeddings).astype(np.float32)

    # Normalize rows for cosine similarity (Gemini returns normalized for default - but safe to normalize)
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-8
    arr = arr / norms
    return arr

def _tokenize(text: str) -> List[str]:
    word = []
    tokens: List[str] = []
    for ch in text.lower():
        if ch.isalnum():
            word.append(ch)
        else:
            if word:
                tok = ''.join(word)
                if len(tok) > 2:
                    tokens.append(tok)
                word = []
    if word:
        tok = ''.join(word)
        if len(tok) > 2:
            tokens.append(tok)
    return tokens

def _build_fallback_tf(chunks: List[str]) -> Tuple[dict, np.ndarray]:
    vocab: dict = {}
    for chunk in chunks:
        for tok in _tokenize(chunk):
            if tok not in vocab:
                vocab[tok] = len(vocab)
    if not vocab:
        vocab = {"document": 0}
    tf = np.zeros((len(chunks), len(vocab)), dtype=np.float32)
    for i, chunk in enumerate(chunks):
        counts: dict = {}
        for tok in _tokenize(chunk):
            counts[tok] = counts.get(tok, 0) + 1
        if counts:
            maxc = max(counts.values())
            for tok, c in counts.items():
                j = vocab.get(tok)
                if j is not None:
                    tf[i, j] = c / maxc
    norms = np.linalg.norm(tf, axis=1, keepdims=True) + 1e-8
    tf = tf / norms
    return vocab, tf

# Global state for last processed document
document_text: str = ""
document_chunks: List[str] = []
document_embeddings: Optional[np.ndarray] = None

# Fallback state (declare globals at module level)
_fallback_vocab: Optional[dict] = None
_fallback_tf_matrix: Optional[np.ndarray] = None

class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    api_key = os.getenv("API_KEY")
    if credentials.credentials != api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    return credentials.credentials

def _extract_text_from_pdf_bytes(data: bytes) -> str:
    pdf_file = io.BytesIO(data)
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        try:
            page_text = page.extract_text() or ""
        except Exception:
            page_text = ""
        if page_text:
            text += page_text + "\n"
    return text

def _extract_text_from_docx_bytes(data: bytes) -> str:
    buf = io.BytesIO(data)
    doc = DocxDocument(buf)
    return "\n".join(p.text for p in doc.paragraphs)

def download_and_extract_text(url: str) -> str:
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        content_type = response.headers.get('content-type', '').lower()

        if url.lower().endswith('.pdf') or 'application/pdf' in content_type:
            return _extract_text_from_pdf_bytes(response.content)

        if url.lower().endswith('.docx') or 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' in content_type:
            return _extract_text_from_docx_bytes(response.content)

        return response.text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error downloading document: {str(e)}")

def _chunk_text(text: str, max_chars: int = 1000, overlap: int = 150) -> List[str]:
    paragraphs = [p.strip() for p in text.split('\n')]
    current = []
    chunks: List[str] = []
    current_len = 0
    for p in paragraphs:
        if not p:
            continue
        if current_len + len(p) + 1 <= max_chars:
            current.append(p)
            current_len += len(p) + 1
        else:
            if current:
                chunk = "\n".join(current)
                chunks.append(chunk)
                if overlap > 0 and len(chunk) > overlap:
                    tail = chunk[-overlap:]
                    current = [tail]
                    current_len = len(tail)
                else:
                    current = []
                    current_len = 0
            if len(p) > max_chars:
                start = 0
                while start < len(p):
                    end = min(start + max_chars, len(p))
                    chunks.append(p[start:end])
                    start = end - overlap if overlap > 0 else end
            else:
                current = [p]
                current_len = len(p)
    if current:
        chunks.append("\n".join(current))
    cleaned = [c.strip() for c in chunks if c and c.strip()]
    return [c for c in cleaned if len(c) > 30]

def create_embeddings(text: str) -> Tuple[np.ndarray, Optional[None], List[str]]:
    """
    Create embeddings for document chunks. This function declares fallback globals
    at the start of the function to avoid Python 'global' placement issues.
    """
    global _fallback_vocab, _fallback_tf_matrix

    chunks = _chunk_text(text)
    if not chunks:
        raise HTTPException(status_code=400, detail="No readable text found in the document")

    # Try Gemini embeddings first
    try:
        embeddings = embed_texts(chunks, task_type="RETRIEVAL_DOCUMENT", output_dim=768)
        _fallback_vocab, _fallback_tf_matrix = None, None
        return embeddings.astype('float32'), None, chunks
    except Exception as e:
        # On any failure, build TF fallback
        vocab, tf = _build_fallback_tf(chunks)
        _fallback_vocab, _fallback_tf_matrix = vocab, tf
        return tf.astype('float32'), None, chunks

def retrieve_relevant_context(question: str, top_k: int = 3) -> str:
    global document_embeddings, document_chunks, _fallback_vocab, _fallback_tf_matrix

    if document_embeddings is None or not document_chunks:
        return ""

    # If we have gemini embeddings, embed the query and compute cosine sims
    if _fallback_vocab is None:
        try:
            q_emb = embed_texts([question], task_type="RETRIEVAL_QUERY", output_dim=document_embeddings.shape[1])
            sims = np.dot(document_embeddings, q_emb[0])
        except Exception:
            # Build fallback and retry
            _fallback_vocab, _fallback_tf_matrix = _build_fallback_tf(document_chunks)
            return retrieve_relevant_context(question, top_k)
    else:
        # fallback TF-based cosine similarity
        vec = np.zeros((len(_fallback_vocab),), dtype=np.float32)
        counts: dict = {}
        for tok in _tokenize(question):
            counts[tok] = counts.get(tok, 0) + 1
        if counts:
            maxc = max(counts.values())
            for tok, c in counts.items():
                j = _fallback_vocab.get(tok)
                if j is not None:
                    vec[j] = c / maxc
        vec = vec / (np.linalg.norm(vec) + 1e-8)
        sims = _fallback_tf_matrix @ vec

    top_k = min(top_k, len(sims))
    top_indices = np.argsort(-sims)[:top_k]
    selected = [document_chunks[i] for i in top_indices]
    return "\n\n".join(selected)

def generate_answer(question: str, context: str) -> str:
    """
    Try to generate an answer using Gemini text model. If generation fails, fall back
    to extracting a short snippet from context.
    """
    try:
        prompt = f"""
You are an expert in analyzing insurance, legal, HR, and compliance documents.
Answer the question accurately and concisely based only on the provided context.

Context:
{context}

Question: {question}

Instructions:
- If the answer is not in the context, reply exactly: "The information is not available in the provided document."
- Keep answers concise (1-3 sentences), cite terms from the context when possible.

Answer:
"""
        gclient = get_gemini_client()
        # Use model generate_content. This is defensive: we try to extract text in a few common places.
        response = gclient.models.generate_content(model="gemini-1.5-flash", contents=prompt)
        text = ""
        if hasattr(response, "text"):
            text = (response.text or "").strip()
        elif isinstance(response, dict):
            # Try to parse dictionary-shaped responses conservatively
            output = response.get("output") or response.get("outputs") or []
            if output:
                first = output[0]
                # nested content array in some libs
                if isinstance(first, dict):
                    content = first.get("content") or first.get("text") or []
                    if isinstance(content, list) and content:
                        text = content[0].get("text", "") if isinstance(content[0], dict) else str(content[0])
                    else:
                        text = first.get("text", "") or ""
        if text:
            return text
        raise RuntimeError("Empty response from generation API")
    except Exception:
        # Fallback: extract up to 3 sentence-like spans from context
        if not context:
            return "The information is not available in the provided document."
        spans = []
        cur = []
        for ch in context:
            cur.append(ch)
            if ch in '.!?\n' and len(''.join(cur).strip()) > 40:
                spans.append(''.join(cur).strip())
                cur = []
            if len(spans) >= 3:
                break
        snippet = ' '.join(spans) if spans else context[:350]
        return snippet.strip()

@app.post("/hackrx/run", response_model=QueryResponse)
async def process_query(request: QueryRequest, token: str = Depends(verify_token)):
    global document_text, document_embeddings, document_chunks
    try:
        # Download and process the document
        print(f"[INFO] Downloading document: {request.documents}")
        document_text = download_and_extract_text(request.documents)

        # Build embeddings (or fallback TF)
        print("[INFO] Creating embeddings / fallback TF")
        document_embeddings, _, document_chunks = create_embeddings(document_text)

        # Process questions
        answers = []
        for i, question in enumerate(request.questions):
            print(f"[INFO] Question {i+1}: {question}")
            context = retrieve_relevant_context(question)
            answer = generate_answer(question, context)
            answers.append(answer)

        return QueryResponse(answers=answers)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/")
async def root():
    return {
        "message": "HackRx 6.0 - LLM Query Retrieval System",
        "status": "running",
        "endpoint": "/hackrx/run"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
