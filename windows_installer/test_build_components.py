#!/usr/bin/env python3
"""
Simple test script to verify build components work without full PyInstaller build.
This tests the core functionality needed for the Windows installer.
"""

import os
import sys
from pathlib import Path

def test_backend_imports():
    """Test that backend can be imported and basic functionality works."""
    print("Testing backend imports...")
    
    backend_dir = Path(__file__).parent.parent / "backend"
    sys.path.insert(0, str(backend_dir))
    
    try:
        from app.core.config import settings
        from app.main import app
        print("✅ Backend imports successful")
        
        os.environ['STANDALONE_MODE'] = 'true'
        print(f"✅ Standalone mode: {settings.STANDALONE_MODE}")
        print(f"✅ Database URL: {settings.DATABASE_URL}")
        
        return True
    except Exception as e:
        print(f"❌ Backend import failed: {e}")
        return False

def test_frontend_structure():
    """Test that frontend structure is correct for building."""
    print("Testing frontend structure...")
    
    frontend_dir = Path(__file__).parent.parent / "frontend"
    
    required_files = [
        "package.json",
        "src/main.tsx",
        "src/App.tsx",
    ]
    
    missing_files = []
    for file_path in required_files:
        full_path = frontend_dir / file_path
        if not full_path.exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing frontend files: {missing_files}")
        return False
    
    print("✅ Frontend structure is valid")
    return True

def test_models_setup():
    """Test that models directory can be created and populated."""
    print("Testing models setup...")
    
    backend_dir = Path(__file__).parent.parent / "backend"
    models_dir = backend_dir / "models"
    
    models_dir.mkdir(exist_ok=True)
    
    test_models = [
        "ecg_classifier.onnx",
        "rhythm_detector.onnx",
        "quality_assessor.onnx",
    ]
    
    for model_name in test_models:
        model_path = models_dir / model_name
        if not model_path.exists():
            with open(model_path, "wb") as f:
                f.write(b"PLACEHOLDER_ONNX_MODEL")
    
    print(f"✅ Models directory created with {len(test_models)} placeholder models")
    return True

def test_installer_scripts():
    """Test that installer scripts are properly formatted."""
    print("Testing installer scripts...")
    
    installer_dir = Path(__file__).parent
    
    required_scripts = [
        "build_backend.py",
        "build_frontend.py", 
        "cardioai_installer.nsi",
        "build_installer.bat",
        "README.md"
    ]
    
    missing_scripts = []
    for script_name in required_scripts:
        script_path = installer_dir / script_name
        if not script_path.exists():
            missing_scripts.append(script_name)
    
    if missing_scripts:
        print(f"❌ Missing installer scripts: {missing_scripts}")
        return False
    
    print("✅ All installer scripts are present")
    return True

def main():
    """Run all component tests."""
    print("=" * 50)
    print("CardioAI Pro Windows Installer Component Tests")
    print("=" * 50)
    print()
    
    tests = [
        ("Backend Imports", test_backend_imports),
        ("Frontend Structure", test_frontend_structure),
        ("Models Setup", test_models_setup),
        ("Installer Scripts", test_installer_scripts),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"Running {test_name}...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
        print()
    
    print("=" * 50)
    print("Test Results Summary")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print()
    print(f"Tests passed: {passed}/{len(results)}")
    
    if passed == len(results):
        print("🎉 All component tests passed! The installer framework is ready.")
        return True
    else:
        print("⚠️  Some tests failed. Review the issues above before building the installer.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
