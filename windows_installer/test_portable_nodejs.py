#!/usr/bin/env python3
"""
Test script to simulate Windows installer portable Node.js functionality on Linux.
This tests the portable Node.js download and setup logic.
"""

import os
import sys
import shutil
import subprocess
import tempfile
from pathlib import Path
import urllib.request
import zipfile

def test_portable_nodejs_download():
    """Test downloading and setting up portable Node.js."""
    print("Testing portable Node.js download functionality...")
    
    installer_dir = Path(__file__).parent
    portable_node_dir = installer_dir / "portable_node"
    
    if portable_node_dir.exists():
        shutil.rmtree(portable_node_dir)
    
    portable_node_dir.mkdir(exist_ok=True)
    
    node_version = "v18.19.0"
    node_url = f"https://nodejs.org/dist/{node_version}/node-{node_version}-linux-x64.tar.xz"
    
    print(f"Downloading Node.js {node_version} for testing...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        node_archive = temp_path / "node-portable.tar.xz"
        
        try:
            urllib.request.urlretrieve(node_url, node_archive)
            print("✅ Download completed")
            
            print("Extracting Node.js...")
            subprocess.run([
                "tar", "-xf", str(node_archive), "-C", str(temp_path)
            ], check=True)
            
            extracted_dir = temp_path / f"node-{node_version}-linux-x64"
            if not extracted_dir.exists():
                raise FileNotFoundError(f"Extracted directory not found: {extracted_dir}")
            
            print("Setting up portable Node.js...")
            for item in extracted_dir.iterdir():
                if item.is_dir():
                    shutil.copytree(item, portable_node_dir / item.name, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, portable_node_dir)
            
            print("✅ Portable Node.js setup completed")
            
            node_exe = portable_node_dir / "bin" / "node"
            npm_exe = portable_node_dir / "bin" / "npm"
            
            if node_exe.exists():
                result = subprocess.run([str(node_exe), "--version"], capture_output=True, text=True)
                print(f"✅ Portable Node.js version: {result.stdout.strip()}")
                
                if npm_exe.exists():
                    result = subprocess.run([str(npm_exe), "--version"], capture_output=True, text=True)
                    print(f"✅ Portable npm version: {result.stdout.strip()}")
                else:
                    print("⚠️ npm not found in portable installation")
            else:
                print("❌ Node.js executable not found")
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to download/setup portable Node.js: {e}")
            return False

def test_build_frontend_with_portable():
    """Test building frontend with portable Node.js."""
    print("\nTesting frontend build with portable Node.js...")
    
    installer_dir = Path(__file__).parent
    portable_node_dir = installer_dir / "portable_node"
    
    if not portable_node_dir.exists():
        print("❌ Portable Node.js not found, skipping frontend build test")
        return False
    
    try:
        result = subprocess.run([
            sys.executable, "build_frontend.py"
        ], cwd=installer_dir, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Frontend build with portable Node.js succeeded")
            return True
        else:
            print(f"❌ Frontend build failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing frontend build: {e}")
        return False

def main():
    """Main test function."""
    print("=" * 60)
    print("CardioAI Pro Portable Node.js Test")
    print("=" * 60)
    
    success = True
    
    if not test_portable_nodejs_download():
        success = False
    
    if not test_build_frontend_with_portable():
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("✅ All tests passed! Portable Node.js functionality works correctly.")
        print("\nFiles created:")
        installer_dir = Path(__file__).parent
        portable_node_dir = installer_dir / "portable_node"
        if portable_node_dir.exists():
            print(f"  - {portable_node_dir} (portable Node.js)")
            print(f"  - {portable_node_dir / 'bin' / 'node'} (Node.js executable)")
            print(f"  - {portable_node_dir / 'bin' / 'npm'} (npm executable)")
    else:
        print("❌ Some tests failed. Check the output above for details.")
    
    print("=" * 60)
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
