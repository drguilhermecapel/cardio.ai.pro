#!/usr/bin/env python3
"""
Validate syntax of recently fixed test files
"""

import ast

def validate_fixed_files():
    """Validate syntax of the two files we just fixed"""
    fixed_files = [
        'tests/test_low_coverage_services_targeted.py',
        'tests/test_services_basic_coverage.py'
    ]
    
    print('🔍 Validating recently fixed files:')
    for file in fixed_files:
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
    validate_fixed_files()
