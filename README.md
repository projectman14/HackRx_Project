# HackRx 6.0 - LLM-Powered Intelligent Query-Retrieval System

A production-ready FastAPI application that processes large documents and answers contextual questions using Google Gemini AI and semantic search with FAISS embeddings.

## 🚀 Features

- **Document Processing**: Handles PDFs, DOCX, and text documents
- **Semantic Search**: Uses FAISS embeddings for intelligent document retrieval
- **AI-Powered Answers**: Leverages Google Gemini for accurate responses
- **Domain Expertise**: Specialized for insurance, legal, HR, and compliance domains
- **RESTful API**: Clean, documented API endpoints
- **Authentication**: Secure Bearer token authentication
- **Production Ready**: Deployable on Heroku, Railway, Render, and Netlify

## 🛠️ Tech Stack

- **Backend**: FastAPI
- **LLM**: Google Gemini Pro
- **Embeddings**: Sentence Transformers + FAISS
- **Document Processing**: PyPDF2, python-docx
- **Deployment**: Heroku/Railway/Render/Netlify

## 📋 Requirements

- Python 3.11+
- Google Gemini API Key
- Custom API Key for authentication

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone <your-repo-url>
cd hackrx-project
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Environment Setup

Create a `.env` file based on `env_example.txt`:

```bash
cp env_example.txt .env
```

Edit `.env` with your API keys:

```env
GOOGLE_API_KEY=your_google_gemini_api_key_here
API_KEY=your_custom_api_key_here
LOG_LEVEL=INFO
```

### 4. Run Locally

```bash
python main.py
```

The API will be available at `http://localhost:8000`

## 📚 API Documentation

### Main Endpoint: `/hackrx/run`

**Method**: `POST`

**Authentication**: Bearer Token

**Request Format**:
```json
{
    "documents": "https://example.com/policy.pdf",
    "questions": [
        "What is the grace period for premium payment?",
        "What is the waiting period for pre-existing diseases?",
        "Does this policy cover maternity expenses?"
    ]
}
```

**Response Format**:
```json
{
    "answers": [
        "A grace period of thirty days is provided for premium payment...",
        "There is a waiting period of thirty-six months for pre-existing diseases...",
        "Maternity expenses are covered under specific conditions..."
    ]
}
```

### Health Check: `/health`

**Method**: `GET`

Returns system health status.

## 🌐 Deployment

### Heroku Deployment

1. **Install Heroku CLI**
2. **Login to Heroku**:
   ```bash
   heroku login
   ```

3. **Create Heroku App**:
   ```bash
   heroku create your-app-name
   ```

4. **Set Environment Variables**:
   ```bash
   heroku config:set GOOGLE_API_KEY=your_google_api_key
   heroku config:set API_KEY=your_custom_api_key
   ```

5. **Deploy**:
   ```bash
   git add .
   git commit -m "Initial deployment"
   git push heroku main
   ```

### Railway Deployment

1. **Connect to Railway**:
   - Go to [Railway.app](https://railway.app)
   - Connect your GitHub repository
   - Add environment variables in Railway dashboard

2. **Deploy**:
   - Railway will automatically deploy on push to main branch

### Render Deployment

1. **Create Render Account**
2. **New Web Service**
3. **Connect Repository**
4. **Set Environment Variables**
5. **Deploy**

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GOOGLE_API_KEY` | Google Gemini API Key | Yes |
| `API_KEY` | Custom API Key for authentication | Yes |
| `LOG_LEVEL` | Logging level (INFO/DEBUG) | No |

### API Authentication

The system uses Bearer token authentication. Include your API key in the Authorization header:

```
Authorization: Bearer your_api_key_here
```

## 📊 System Architecture

```
Input Document (PDF/DOCX) 
    ↓
Document Processing (PyPDF2)
    ↓
Text Extraction
    ↓
Embedding Generation (Sentence Transformers)
    ↓
FAISS Index Creation
    ↓
Query Processing
    ↓
Semantic Search
    ↓
Context Retrieval
    ↓
Google Gemini Answer Generation
    ↓
Structured JSON Response
```

## 🧪 Testing

### Test the API

```bash
curl -X POST "https://your-app.herokuapp.com/hackrx/run" \
  -H "Authorization: Bearer your_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf",
    "questions": [
      "What is the grace period for premium payment?",
      "What is the waiting period for pre-existing diseases?"
    ]
  }'
```

## 🔍 Features Explained

### 1. Document Processing
- Downloads documents from URLs
- Extracts text from PDFs and other formats
- Handles large documents efficiently

### 2. Semantic Search
- Uses FAISS for fast similarity search
- Sentence transformers for embedding generation
- Retrieves most relevant document chunks

### 3. AI Answer Generation
- Google Gemini Pro for accurate responses
- Context-aware answer generation
- Domain-specific prompting

### 4. Security
- Bearer token authentication
- Input validation
- Error handling

## 📈 Performance

- **Response Time**: < 30 seconds
- **Document Size**: Handles large documents
- **Concurrent Requests**: FastAPI async support
- **Memory Efficient**: Streaming document processing

## 🐛 Troubleshooting

### Common Issues

1. **API Key Issues**:
   - Ensure Google Gemini API key is valid
   - Check Bearer token authentication

2. **Document Download Issues**:
   - Verify document URL is accessible
   - Check document format support

3. **Memory Issues**:
   - Large documents may require more memory
   - Consider chunking for very large documents

## 📝 License

This project is created for HackRx 6.0 hackathon.

## 🤝 Contributing

This is a hackathon project. Feel free to fork and improve!

---

**Ready for HackRx 6.0 Submission! 🚀**
