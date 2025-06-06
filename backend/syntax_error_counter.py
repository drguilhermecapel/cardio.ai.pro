#!/usr/bin/env python3
"""
Count and categorize syntax errors in test files
"""

import ast
import os

def count_syntax_errors():
    """Count syntax errors in all test files"""
    test_dir = 'tests'
    error_count = 0
    error_files = []
    
    for root, dirs, files in os.walk(test_dir):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r') as f:
                        content = f.read()
                    ast.parse(content)
                except SyntaxError as e:
                    error_count += 1
                    error_files.append(f'{filepath}:{e.lineno} - {e.msg}')
                except Exception:
                    pass
    
    print(f'Total syntax errors found: {error_count}')
    print('\nFirst 10 errors:')
    for error in error_files[:10]:
        print(f'  {error}')
    
    return error_count, error_files

if __name__ == "__main__":
    count_syntax_errors()
