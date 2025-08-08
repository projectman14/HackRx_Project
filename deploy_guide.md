# 🚀 HackRx 6.0 Deployment Guide

This guide will help you deploy your HackRx 6.0 project to various platforms.

## 📋 Prerequisites

1. **Google Gemini API Key**: Get from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. **Git Repository**: Your project should be in a Git repository
3. **Platform Account**: Account on your chosen deployment platform

## 🌐 Deployment Options

### Option 1: Heroku (Recommended)

#### Step 1: Install Heroku CLI
```bash
# Windows
winget install --id=Heroku.HerokuCLI

# macOS
brew tap heroku/brew && brew install heroku

# Linux
curl https://cli-assets.heroku.com/install.sh | sh
```

#### Step 2: Login to Heroku
```bash
heroku login
```

#### Step 3: Create Heroku App
```bash
heroku create your-hackrx-app-name
```

#### Step 4: Set Environment Variables
```bash
heroku config:set GOOGLE_API_KEY=your_google_gemini_api_key
heroku config:set API_KEY=your_custom_api_key_here
```

#### Step 5: Deploy
```bash
git add .
git commit -m "Initial HackRx 6.0 deployment"
git push heroku main
```

#### Step 6: Verify Deployment
```bash
heroku open
```

### Option 2: Railway

#### Step 1: Connect to Railway
1. Go to [Railway.app](https://railway.app)
2. Sign up/Login with GitHub
3. Click "New Project"
4. Select "Deploy from GitHub repo"

#### Step 2: Configure Environment Variables
In Railway dashboard:
- `GOOGLE_API_KEY`: Your Google Gemini API key
- `API_KEY`: Your custom API key

#### Step 3: Deploy
Railway will automatically deploy when you push to your main branch.

### Option 3: Render

#### Step 1: Create Render Account
1. Go to [Render.com](https://render.com)
2. Sign up with GitHub

#### Step 2: Create New Web Service
1. Click "New +"
2. Select "Web Service"
3. Connect your GitHub repository

#### Step 3: Configure Service
- **Name**: `hackrx-api`
- **Environment**: `Python 3`
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `uvicorn main:app --host=0.0.0.0 --port=$PORT`

#### Step 4: Set Environment Variables
Add these in Render dashboard:
- `GOOGLE_API_KEY`
- `API_KEY`

#### Step 5: Deploy
Click "Create Web Service"

### Option 4: Netlify Functions

#### Step 1: Create netlify.toml
```toml
[build]
  functions = "functions"
  publish = "public"

[functions]
  directory = "functions"
```

#### Step 2: Create Function
Create `functions/hackrx-run.js`:
```javascript
const { spawn } = require('child_process');

exports.handler = async (event, context) => {
  // Implementation for Netlify Functions
  // This would require adapting the Python code to JavaScript
};
```

## 🔧 Environment Variables

### Required Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `GOOGLE_API_KEY` | Google Gemini API Key | `AIzaSyC...` |
| `API_KEY` | Custom API Key for authentication | `hackrx_2024_secure_key` |

### Optional Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LOG_LEVEL` | Logging level | `INFO` |
| `PORT` | Server port | `8000` |

## 🧪 Testing Your Deployment

### 1. Health Check
```bash
curl https://your-app.herokuapp.com/health
```

### 2. Test Main Endpoint
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

### 3. Run Test Script
```bash
python test_api.py
```

## 📊 Monitoring

### Heroku Logs
```bash
heroku logs --tail
```

### Railway Logs
Available in Railway dashboard

### Render Logs
Available in Render dashboard

## 🔍 Troubleshooting

### Common Issues

1. **Build Failures**
   - Check Python version compatibility
   - Verify all dependencies in requirements.txt
   - Check for missing environment variables

2. **Runtime Errors**
   - Check logs for specific error messages
   - Verify API keys are correctly set
   - Ensure all required packages are installed

3. **Timeout Issues**
   - Increase timeout limits if needed
   - Optimize document processing for large files

4. **Memory Issues**
   - Upgrade to higher tier if needed
   - Optimize embedding generation
   - Consider chunking large documents

### Performance Optimization

1. **Caching**
   - Implement document caching
   - Cache embeddings for repeated documents

2. **Async Processing**
   - Use background tasks for large documents
   - Implement queue system for heavy processing

3. **Resource Management**
   - Monitor memory usage
   - Implement proper cleanup

## 🚀 Production Checklist

- [ ] Environment variables configured
- [ ] API keys secured
- [ ] HTTPS enabled
- [ ] Health check endpoint working
- [ ] Main endpoint tested
- [ ] Authentication working
- [ ] Error handling implemented
- [ ] Logs configured
- [ ] Monitoring set up

## 📞 Support

If you encounter issues:

1. Check the platform's documentation
2. Review application logs
3. Test locally first
4. Verify all environment variables
5. Check API key permissions

---

**Your HackRx 6.0 API is now ready for submission! 🎉**
