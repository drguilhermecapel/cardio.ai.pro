#!/usr/bin/env python3
"""
Build script for creating standalone Windows frontend of CardioAI Pro.
Builds the React application and prepares it for bundling.
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

def check_node_npm():
    """Check if Node.js and npm are available."""
    print("Checking Node.js and npm availability...")
    
    try:
        result = subprocess.run(["node", "--version"], capture_output=True, text=True, check=True)
        node_version = result.stdout.strip()
        print(f"Node.js version: {node_version}")
        
        result = subprocess.run(["npm", "--version"], capture_output=True, text=True, check=True)
        npm_version = result.stdout.strip()
        print(f"npm version: {npm_version}")
        
        return True
        
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Node.js or npm not found!")
        print("Please install Node.js from https://nodejs.org/")
        return False

def install_dependencies():
    """Install frontend dependencies."""
    print("Installing frontend dependencies...")
    
    frontend_dir = Path(__file__).parent.parent / "frontend"
    os.chdir(frontend_dir)
    
    if (frontend_dir / "package-lock.json").exists():
        subprocess.run(["npm", "install"], check=True)
    elif (frontend_dir / "yarn.lock").exists():
        try:
            subprocess.run(["yarn", "install"], check=True)
        except FileNotFoundError:
            print("Yarn not found, falling back to npm...")
            subprocess.run(["npm", "install"], check=True)
    elif (frontend_dir / "pnpm-lock.yaml").exists():
        try:
            subprocess.run(["pnpm", "install"], check=True)
        except FileNotFoundError:
            print("pnpm not found, falling back to npm...")
            subprocess.run(["npm", "install"], check=True)
    else:
        subprocess.run(["npm", "install"], check=True)

def create_production_env():
    """Create production environment configuration."""
    print("Creating production environment configuration...")
    
    frontend_dir = Path(__file__).parent.parent / "frontend"
    env_file = frontend_dir / ".env.production"
    
    env_content = """# Production environment for standalone Windows build
VITE_API_URL=http://localhost:8000
VITE_APP_TITLE=CardioAI Pro
VITE_APP_VERSION=1.0.0
VITE_ENVIRONMENT=production
"""
    
    with open(env_file, "w") as f:
        f.write(env_content)
    
    print(f"Created production environment file: {env_file}")

def build_frontend():
    """Build the React frontend for production."""
    print("Building React frontend for production...")
    
    frontend_dir = Path(__file__).parent.parent / "frontend"
    os.chdir(frontend_dir)
    
    if (frontend_dir / "package-lock.json").exists():
        subprocess.run(["npm", "run", "build"], check=True)
    elif (frontend_dir / "yarn.lock").exists():
        try:
            subprocess.run(["yarn", "build"], check=True)
        except FileNotFoundError:
            subprocess.run(["npm", "run", "build"], check=True)
    elif (frontend_dir / "pnpm-lock.yaml").exists():
        try:
            subprocess.run(["pnpm", "run", "build"], check=True)
        except FileNotFoundError:
            subprocess.run(["npm", "run", "build"], check=True)
    else:
        subprocess.run(["npm", "run", "build"], check=True)

def copy_build_files():
    """Copy built frontend files to installer directory."""
    print("Copying built frontend files...")
    
    frontend_dir = Path(__file__).parent.parent / "frontend"
    installer_dir = Path(__file__).parent
    
    dist_dir = frontend_dir / "dist"
    frontend_build_dir = installer_dir / "frontend_build"
    
    if not dist_dir.exists():
        raise FileNotFoundError(f"Frontend build directory not found: {dist_dir}")
    
    if frontend_build_dir.exists():
        shutil.rmtree(frontend_build_dir)
    
    shutil.copytree(dist_dir, frontend_build_dir)
    print(f"Frontend build copied to: {frontend_build_dir}")

def create_frontend_server_script():
    """Create a simple HTTP server script for serving the frontend."""
    print("Creating frontend server script...")
    
    installer_dir = Path(__file__).parent
    server_script = installer_dir / "serve_frontend.py"
    
    script_content = '''#!/usr/bin/env python3
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
        print("\\nShutting down frontend server...")
        server.shutdown()

if __name__ == "__main__":
    main()
'''
    
    with open(server_script, "w") as f:
        f.write(script_content)
    
    print(f"Frontend server script created: {server_script}")

def main():
    """Main build process."""
    try:
        print("Starting CardioAI Pro frontend build process...")
        
        if not check_node_npm():
            sys.exit(1)
        
        create_production_env()
        install_dependencies()
        build_frontend()
        copy_build_files()
        create_frontend_server_script()
        
        print("\n✅ Frontend build completed successfully!")
        print("Files created:")
        installer_dir = Path(__file__).parent
        print(f"  - {installer_dir / 'frontend_build'} (directory)")
        print(f"  - {installer_dir / 'serve_frontend.py'}")
        
    except Exception as e:
        print(f"\n❌ Build failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
