#!/usr/bin/env python3
"""
Quick coverage test execution for regulatory compliance
Target: 80% coverage for FDA, ANVISA, NMSA, EU standards
"""

import subprocess
import sys

def run_coverage_test():
    """Run ultra-targeted coverage test quickly"""
    try:
        cmd = [
            sys.executable, "-m", "pytest",
            "tests/test_ultra_targeted_80_coverage.py",
            "--cov=app",
            "--cov-report=term-missing",
            "--tb=no",
            "-q",
            "--disable-warnings",
            "--maxfail=1"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        print("COVERAGE TEST RESULTS:")
        print("=" * 50)
        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        lines = result.stdout.split('\n')
        for line in lines:
            if 'TOTAL' in line and '%' in line:
                print(f"\n🎯 COVERAGE RESULT: {line}")
                break
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("⚠️ Test execution timed out after 30 seconds")
        return False
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False

if __name__ == "__main__":
    success = run_coverage_test()
    sys.exit(0 if success else 1)
