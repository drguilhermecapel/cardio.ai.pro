#!/usr/bin/env python3
"""
Syntax validation script for critical test files
"""

import ast
import glob

def validate_critical_files():
    """Validate syntax of critical test files mentioned in CI logs"""
    critical_files = [
        'tests/test_80_percent_simple.py',
        'tests/integration/test_api_integration.py'
    ]
    
    print('🔍 Checking critical files syntax:')
    for file in critical_files:
        try:
            with open(file, 'r') as f:
                content = f.read()
            ast.parse(content)
            print(f'✅ {file}: Valid syntax')
        except SyntaxError as e:
            print(f'❌ {file}: {e}')
        except Exception as e:
            print(f'⚠️  {file}: {e}')

if __name__ == "__main__":
    validate_critical_files()
