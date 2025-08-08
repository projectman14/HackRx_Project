# 🚀 HackRx 6.0 - Quick Start Guide

## ✅ Project Status: READY TO DEPLOY

Your HackRx 6.0 project is fully functional and ready for deployment!

## 📋 What You Have

✅ **Complete API System**: FastAPI with `/hackrx/run` endpoint  
✅ **Document Processing**: PDF, DOCX, and text file support  
✅ **AI Integration**: Google Gemini for embeddings and text generation  
✅ **Authentication**: Bearer token security  
✅ **Deployment Ready**: Heroku, Railway, Render support  
✅ **Testing Suite**: Comprehensive test scripts  

## 🚀 Quick Deployment Steps

### Option 1: Heroku (Recommended)

1. **Install Heroku CLI**:
   ```bash
   # Windows
   winget install --id=Heroku.HerokuCLI
   ```

2. **Login to Heroku**:
   ```bash
   heroku login
   ```

3. **Create App**:
   ```bash
   heroku create your-hackrx-app-name
   ```

4. **Set Environment Variables**:
   ```bash
   heroku config:set GOOGLE_API_KEY=your_google_gemini_api_key
   heroku config:set API_KEY=hackrx_2024_secure_key_123
   ```

5. **Deploy**:
   ```bash
   git add .
   git commit -m "HackRx 6.0 deployment"
   git push heroku main
   ```

### Option 2: Railway (Easiest)

1. Go to [Railway.app](https://railway.app)
2. Connect your GitHub repository
3. Add environment variables in Railway dashboard:
   - `GOOGLE_API_KEY`: Your Google Gemini API key
   - `API_KEY`: `hackrx_2024_secure_key_123`
4. Deploy automatically

## 🧪 Local Testing

### 1. Get API Keys

**Google Gemini API Key**:
- Visit: https://makersuite.google.com/app/apikey
- Create a new API key
- Copy the key

### 2. Configure Environment

Edit `.env` file:
```env
GOOGLE_API_KEY=your_actual_gemini_api_key_here
API_KEY=hackrx_2024_secure_key_123
LOG_LEVEL=INFO
```

### 3. Run Locally

```bash
# Start the server
python main.py

# In another terminal, test the API
python test_api.py
```

### 4. Test Endpoints

**Health Check**:
```bash
curl http://localhost:8000/health
```

**Main API**:
```bash
curl -X POST "http://localhost:8000/hackrx/run" \
  -H "Authorization: Bearer hackrx_2024_secure_key_123" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf",
    "questions": [
      "What is the grace period for premium payment?",
      "What is the waiting period for pre-existing diseases?"
    ]
  }'
```

## 📊 API Documentation

### Endpoint: `POST /hackrx/run`

**Authentication**: Bearer Token  
**Content-Type**: application/json  

**Request**:
```json
{
    "documents": "https://example.com/policy.pdf",
    "questions": [
        "What is the grace period for premium payment?",
        "What is the waiting period for pre-existing diseases?"
    ]
}
```

**Response**:
```json
{
    "answers": [
        "A grace period of thirty days is provided for premium payment...",
        "There is a waiting period of thirty-six months for pre-existing diseases..."
    ]
}
```

## 🔧 Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GOOGLE_API_KEY` | Google Gemini API Key | Yes |
| `API_KEY` | Custom API Key for authentication | Yes |
| `LOG_LEVEL` | Logging level (INFO/DEBUG) | No |

## 🌐 Deployment URLs

After deployment, your API will be available at:
- **Heroku**: `https://your-app-name.herokuapp.com`
- **Railway**: `https://your-app-name.railway.app`
- **Render**: `https://your-app-name.onrender.com`

## 🧪 Testing Your Deployment

1. **Health Check**:
   ```bash
   curl https://your-app.herokuapp.com/health
   ```

2. **Test Main Endpoint**:
   ```bash
   curl -X POST "https://your-app.herokuapp.com/hackrx/run" \
     -H "Authorization: Bearer hackrx_2024_secure_key_123" \
     -H "Content-Type: application/json" \
     -d '{
       "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf",
       "questions": [
         "What is the grace period for premium payment?"
       ]
     }'
   ```

## 📝 Hackathon Submission

Your project is ready for HackRx 6.0 submission! Include:

1. **Deployed URL**: Your live API endpoint
2. **GitHub Repository**: Source code
3. **Documentation**: README.md and PROJECT_SUMMARY.md
4. **Demo**: Working API with test cases

## 🎯 Key Features

- ✅ **Document Processing**: PDF, DOCX, text files
- ✅ **Semantic Search**: Google Gemini embeddings
- ✅ **AI Answers**: Context-aware responses
- ✅ **Production Ready**: Error handling, logging, CORS
- ✅ **Multi-Platform**: Heroku, Railway, Render, Netlify
- ✅ **Security**: Bearer token authentication
- ✅ **Testing**: Comprehensive test suite

## 🚀 Ready to Deploy!

Your HackRx 6.0 project is **100% ready** for deployment and submission. Follow the steps above to get it live!

---

**🎉 Good luck with your HackRx 6.0 submission!**
