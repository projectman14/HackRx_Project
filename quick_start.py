#!/usr/bin/env python3
"""
Quick start script for HackRx 6.0
"""

import os
import sys
import subprocess
from pathlib import Path

def check_env_file():
    """Check if .env file exists and has proper keys"""
    env_file = Path(".env")
    if not env_file.exists():
        print("❌ .env file not found. Run setup.py first.")
        return False
    
    # Read .env file
    with open(env_file, "r") as f:
        content = f.read()
    
    # Check if API keys are set
    if "your_google_gemini_api_key_here" in content or "your_custom_api_key_here" in content:
        print("⚠️  Please update your .env file with actual API keys")
        print("   - GOOGLE_API_KEY: Get from https://makersuite.google.com/app/apikey")
        print("   - API_KEY: Set a custom API key for authentication")
        return False
    
    return True

def start_server():
    """Start the FastAPI server"""
    try:
        print("🚀 Starting HackRx 6.0 server...")
        print("📡 API will be available at: http://localhost:8000")
        print("📚 API docs at: http://localhost:8000/docs")
        print("🔍 Health check at: http://localhost:8000/health")
        print("\nPress Ctrl+C to stop the server")
        print("-" * 50)
        
        subprocess.run([sys.executable, "main.py"])
        
    except KeyboardInterrupt:
        print("\n👋 Server stopped")
    except Exception as e:
        print(f"❌ Failed to start server: {e}")

def main():
    """Main quick start function"""
    print("🚀 HackRx 6.0 Quick Start")
    print("=" * 30)
    
    # Check if setup is complete
    if not check_env_file():
        print("\nTo set up the project:")
        print("1. Run: python setup.py")
        print("2. Edit .env file with your API keys")
        print("3. Run this script again")
        return
    
    # Start server
    start_server()

if __name__ == "__main__":
    main()
