#!/usr/bin/env python3
"""Simple coverage maximizer runner to avoid pytest configuration issues."""

import subprocess
import sys
import os

def run_coverage_maximizer():
    """Run the coverage maximizer with minimal configuration."""
    os.chdir('/home/ubuntu/cardio.ai.pro/backend')
    
    os.environ['PYTHONPATH'] = '/home/ubuntu/cardio.ai.pro/backend'
    os.environ['TESTING'] = 'true'
    
    cmd = [
        sys.executable, '-m', 'pytest',
        'tests/test_coverage_maximizer.py',
        '--cov=app',
        '--cov-report=term',
        '--tb=short',
        '-v',
        '--timeout=60'
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    print("STDOUT:")
    print(result.stdout)
    print("\nSTDERR:")
    print(result.stderr)
    print(f"\nReturn code: {result.returncode}")
    
    return result.returncode == 0

if __name__ == "__main__":
    success = run_coverage_maximizer()
    sys.exit(0 if success else 1)
