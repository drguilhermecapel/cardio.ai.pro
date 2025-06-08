#!/usr/bin/env python3
"""
Simple HTTP server for serving the CardioAI Pro frontend.
This script serves the built React application.
"""

import os
import sys
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
import webbrowser
import threading
import time

class CustomHTTPRequestHandler(SimpleHTTPRequestHandler):
    """Custom handler to serve React app with proper routing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(Path(__file__).parent / "frontend_build"), **kwargs)
    
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()
    
    def do_GET(self):
        if self.path.startswith('/api/'):
            self.send_error(404, "API endpoint - should be handled by backend")
            return
        
        if '.' not in self.path.split('/')[-1] and self.path != '/':
            self.path = '/index.html'
        
        return super().do_GET()

def open_browser_delayed():
    """Open browser after a short delay."""
    time.sleep(2)
    webbrowser.open('http://localhost:3000')

def main():
    """Start the frontend server."""
    port = 3000
    
    frontend_build = Path(__file__).parent / "frontend_build"
    if not frontend_build.exists():
        print("❌ Frontend build not found!")
        print("Please run build_frontend.py first.")
        sys.exit(1)
    
    print(f"Starting CardioAI Pro frontend server on port {port}...")
    print(f"Serving files from: {frontend_build}")
    print(f"Open your browser to: http://localhost:{port}")
    print("Press Ctrl+C to stop the server")
    
    browser_thread = threading.Thread(target=open_browser_delayed)
    browser_thread.daemon = True
    browser_thread.start()
    
    try:
        server = HTTPServer(('localhost', port), CustomHTTPRequestHandler)
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down frontend server...")
        server.shutdown()

if __name__ == "__main__":
    main()
