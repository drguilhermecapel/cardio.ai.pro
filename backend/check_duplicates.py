#!/usr/bin/env python3
"""Check for duplicate decorators in test files"""

import re
import os

def check_file_for_duplicates(filepath):
    """Check a single file for duplicate decorators"""
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    lines = content.split('\n')
    
    print(f"\n=== Checking {filepath} ===")
    
    fixture_lines = []
    for i, line in enumerate(lines, 1):
        if '@pytest.fixture' in line:
            fixture_lines.append((i, line.strip()))
    
    print(f"Found {len(fixture_lines)} @pytest.fixture decorators:")
    for line_num, line_content in fixture_lines:
        print(f"  Line {line_num}: {line_content}")
    
    for i in range(len(fixture_lines) - 1):
        current_line = fixture_lines[i][0]
        next_line = fixture_lines[i + 1][0]
        if next_line - current_line <= 2:  # Within 2 lines of each other
            print(f"⚠️  POTENTIAL DUPLICATE: Lines {current_line} and {next_line}")
    
    function_pattern = r'def\s+(\w+)\s*\('
    for i, line in enumerate(lines):
        if 'def ' in line:
            match = re.search(function_pattern, line)
            if match:
                func_name = match.group(1)
                decorators = []
                j = i - 1
                while j >= 0 and (lines[j].strip().startswith('@') or lines[j].strip() == ''):
                    if lines[j].strip().startswith('@pytest.fixture'):
                        decorators.append(j + 1)
                    j -= 1
                
                if len(decorators) > 1:
                    print(f"🚨 DUPLICATE DECORATORS on function '{func_name}': lines {decorators}")

if __name__ == '__main__':
    check_file_for_duplicates('tests/test_coverage_maximizer.py')
