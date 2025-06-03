#!/usr/bin/env python3
"""
Hybrid ECG Analysis System Installer
Installs dependencies and sets up the hybrid ECG analysis system for cardio.ai.pro
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(command, description):
    """Run a shell command and handle errors"""
    print(f"Running: {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed: {e.stderr}")
        return None


def check_python_version():
    """Check if Python 3.11.9 is available"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 11:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor}.{version.micro} is not compatible. Requires Python 3.11+")
        return False


def install_system_dependencies():
    """Install system-level dependencies"""
    print("\nInstalling system dependencies...")
    
    run_command("sudo apt-get update", "Updating package list")
    
    packages = [
        "build-essential",
        "python3-dev",
        "libffi-dev",
        "libssl-dev",
        "libblas-dev",
        "liblapack-dev",
        "gfortran",
        "pkg-config"
    ]
    
    for package in packages:
        run_command(f"sudo apt-get install -y {package}", f"Installing {package}")


def install_poetry_dependencies():
    """Install Python dependencies using Poetry"""
    print("\nInstalling Python dependencies...")
    
    if run_command("poetry --version", "Checking Poetry installation") is None:
        print("Poetry not found. Installing Poetry...")
        run_command("curl -sSL https://install.python-poetry.org | python3 -", "Installing Poetry")
        run_command("export PATH=\"$HOME/.local/bin:$PATH\"", "Adding Poetry to PATH")
    
    run_command("poetry install", "Installing project dependencies")
    
    ml_packages = [
        "torch>=2.1.0",
        "tensorflow>=2.15.0", 
        "xgboost>=2.0.3",
        "lightgbm>=4.3.0",
        "biosppy>=2.2.0",
        "wfdb>=4.1.2",
        "pyedflib>=0.1.36",
        "tqdm>=4.66.1"
    ]
    
    for package in ml_packages:
        run_command(f"poetry add {package}", f"Installing {package}")


def setup_models_directory():
    """Create models directory structure"""
    print("\nSetting up models directory...")
    
    models_dir = Path("backend/models")
    models_dir.mkdir(exist_ok=True)
    
    subdirs = ["hybrid", "onnx", "tensorflow", "pytorch"]
    for subdir in subdirs:
        (models_dir / subdir).mkdir(exist_ok=True)
    
    print("✓ Models directory structure created")


def run_tests():
    """Run the hybrid ECG analysis tests"""
    print("\nRunning hybrid ECG analysis tests...")
    
    test_result = run_command(
        "poetry run pytest backend/tests/test_hybrid_ecg_service.py -v",
        "Running hybrid ECG service tests"
    )
    
    if test_result is not None:
        print("✓ All hybrid ECG tests passed")
        return True
    else:
        print("✗ Some tests failed")
        return False


def run_linting():
    """Run code linting"""
    print("\nRunning code linting...")
    
    run_command("poetry run ruff check backend/app/services/hybrid_ecg_service.py", "Linting hybrid ECG service")
    run_command("poetry run ruff check backend/app/services/regulatory_validation.py", "Linting regulatory validation")
    run_command("poetry run ruff check backend/app/utils/ecg_hybrid_processor.py", "Linting ECG hybrid processor")
    
    print("✓ Linting completed")


def create_sample_data():
    """Create sample ECG data for testing"""
    print("\nCreating sample ECG data...")
    
    sample_dir = Path("backend/sample_data")
    sample_dir.mkdir(exist_ok=True)
    
    sample_csv = sample_dir / "sample_ecg.csv"
    with open(sample_csv, 'w') as f:
        f.write("I,II,III,aVR,aVL,aVF,V1,V2,V3,V4,V5,V6\n")
        import random
        for i in range(5000):
            values = [str(random.gauss(0, 0.1)) for _ in range(12)]
            f.write(",".join(values) + "\n")
    
    print("✓ Sample ECG data created")


def main():
    """Main installer function"""
    print("=" * 60)
    print("Hybrid ECG Analysis System Installer")
    print("=" * 60)
    
    if not check_python_version():
        sys.exit(1)
    
    install_system_dependencies()
    
    install_poetry_dependencies()
    
    setup_models_directory()
    
    create_sample_data()
    
    run_linting()
    
    if run_tests():
        print("\n" + "=" * 60)
        print("✓ Hybrid ECG Analysis System installed successfully!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Start the backend server: poetry run uvicorn app.main:app --reload")
        print("2. Access the API documentation at: http://localhost:8000/docs")
        print("3. Upload ECG files for analysis using the hybrid AI system")
        print("4. Review regulatory validation reports for compliance")
        print("\nSupported ECG formats: .csv, .txt, .dat, .edf, .jpg, .png")
        print("Regulatory standards: FDA, ANVISA, NMSA (China), EU MDR")
    else:
        print("\n" + "=" * 60)
        print("✗ Installation completed with test failures")
        print("=" * 60)
        print("Please review test output and fix any issues before using the system")


if __name__ == "__main__":
    main()
