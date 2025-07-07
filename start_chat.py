#!/usr/bin/env python3
"""
Start script for SQL Agent Chat Interface
Launches both backend and frontend
"""

import subprocess
import time
import sys
import os
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import fastapi
        import streamlit
        import requests
        print("✅ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install dependencies:")
        print("pip install -r requirements.txt")
        return False

def check_env_file():
    """Check if .env file exists"""
    if not Path(".env").exists():
        print("❌ .env file not found")
        print("Please create a .env file with your OpenAI API key:")
        print("OPENAI_API_KEY=your_api_key_here")
        return False
    print("✅ .env file found")
    return True

def check_parquet_file():
    """Check if cortex.parquet exists"""
    if not Path("cortex.parquet").exists():
        print("❌ cortex.parquet file not found")
        print("Please ensure cortex.parquet is in the root directory")
        return False
    print("✅ cortex.parquet file found")
    return True

def start_backend():
    """Start the FastAPI backend"""
    print("🚀 Starting backend server...")
    backend_dir = Path("backend")
    if not backend_dir.exists():
        print("❌ Backend directory not found")
        return None
    
    try:
        # Start backend in background
        process = subprocess.Popen(
            [sys.executable, "main.py"],
            cwd=backend_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait a bit for the server to start
        time.sleep(3)
        
        # Check if process is still running
        if process.poll() is None:
            print("✅ Backend started successfully")
            return process
        else:
            stdout, stderr = process.communicate()
            print(f"❌ Backend failed to start: {stderr.decode()}")
            return None
            
    except Exception as e:
        print(f"❌ Error starting backend: {e}")
        return None

def start_frontend():
    """Start the Streamlit frontend"""
    print("🚀 Starting frontend server...")
    frontend_dir = Path("frontend")
    if not frontend_dir.exists():
        print("❌ Frontend directory not found")
        return None
    
    try:
        # Start frontend in background with better error handling
        process = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", "streamlit_app.py", "--server.port", "8501", "--server.headless", "true"],
            cwd=frontend_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait a bit for the server to start
        time.sleep(8)
        
        # Check if process is still running
        if process.poll() is None:
            # Try to check if the server is responding
            try:
                import requests
                response = requests.get("http://localhost:8501", timeout=5)
                if response.status_code == 200:
                    print("✅ Frontend started successfully")
                    return process
                else:
                    print(f"⚠️ Frontend started but not responding properly (status: {response.status_code})")
                    return process
            except requests.exceptions.RequestException:
                print("⚠️ Frontend started but not responding yet")
                return process
        else:
            stdout, stderr = process.communicate()
            print(f"❌ Frontend failed to start: {stderr}")
            return None
            
    except Exception as e:
        print(f"❌ Error starting frontend: {e}")
        return None

def main():
    """Main function to start the Real Estate Asset Management Assistant"""
    print("🏢 Real Estate Asset Management Assistant")
    print("=" * 50)
    
    # Check prerequisites
    if not check_dependencies():
        return
    
    if not check_env_file():
        return
    
    if not check_parquet_file():
        return
    
    print("\n📋 Starting services...")
    
    # Start backend
    backend_process = start_backend()
    if not backend_process:
        print("❌ Failed to start backend. Exiting.")
        return
    
    # Start frontend
    frontend_process = start_frontend()
    if not frontend_process:
        print("❌ Failed to start frontend. Stopping backend.")
        backend_process.terminate()
        return
    
    print("\n🎉 Both services started successfully!")
    print("=" * 50)
    print("📱 Frontend: http://localhost:8501")
    print("🔧 Backend API: http://localhost:8000")
    print("📚 API Docs: http://localhost:8000/docs")
    print("💡 Ask questions about your property portfolio!")
    print("\n💡 Press Ctrl+C to stop all services")
    print("=" * 50)
    
    try:
        # Keep the script running
        while True:
            time.sleep(1)
            
            # Check if processes are still running
            if backend_process.poll() is not None:
                print("❌ Backend process stopped unexpectedly")
                break
                
            if frontend_process.poll() is not None:
                print("❌ Frontend process stopped unexpectedly")
                break
                
    except KeyboardInterrupt:
        print("\n🛑 Stopping services...")
        
        # Terminate processes
        if backend_process:
            backend_process.terminate()
            print("✅ Backend stopped")
            
        if frontend_process:
            frontend_process.terminate()
            print("✅ Frontend stopped")
            
        print("👋 Goodbye!")

if __name__ == "__main__":
    main() 