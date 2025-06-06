#!/usr/bin/env python3
"""
Comprehensive syntax fix strategy for all remaining test files
Based on CI analysis showing 29 syntax errors
"""

import ast
import os
import re

def fix_all_syntax_errors():
    """Fix all identified syntax errors systematically"""
    
    priority_fixes = {
        'tests/test_simple_80_coverage_final.py': 'unmatched )',
        'tests/test_ecg_hybrid_processor_critical.py': 'unexpected unindent',
        'tests/test_critical_zero_coverage_services.py': 'unexpected unindent',
        'tests/test_hybrid_ecg_service.py': 'unexpected unindent',
        'tests/test_hybrid_ecg_direct_import.py': 'unexpected unindent',
        'tests/test_hybrid_ecg_service_medical_grade.py': 'unexpected unindent',
        'tests/test_ml_model_service_generated.py': 'unexpected unindent',
        'tests/test_ecg_repository_generated.py': 'unexpected unindent',
        'tests/test_validation_service_generated.py': 'unexpected unindent',
        'tests/test_ecg_hybrid_processor_coverage.py': 'unexpected unindent'
    }
    
    print("🔧 Starting comprehensive syntax fixes...")
    
    for filepath, error_type in priority_fixes.items():
        if os.path.exists(filepath):
            print(f"Fixing {filepath} ({error_type})")
            fix_file_syntax(filepath, error_type)
        else:
            print(f"⚠️  File not found: {filepath}")
    
    print("\n🔍 Scanning for additional syntax errors...")
    scan_and_fix_remaining_errors()

def fix_file_syntax(filepath, error_type):
    """Fix specific syntax error in file"""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        if error_type == 'unmatched )':
            content = fix_unmatched_parentheses(content)
        elif error_type == 'unexpected unindent':
            content = fix_indentation_errors(content)
        elif 'await' in error_type and 'outside async' in error_type:
            content = fix_async_await_errors(content)
        
        try:
            ast.parse(content)
            with open(filepath, 'w') as f:
                f.write(content)
            print(f"✅ Fixed {filepath}")
        except SyntaxError as e:
            print(f"❌ Still has syntax error in {filepath}: {e}")
            
    except Exception as e:
        print(f"⚠️  Error processing {filepath}: {e}")

def fix_unmatched_parentheses(content):
    """Fix unmatched parentheses in content"""
    lines = content.split('\n')
    fixed_lines = []
    
    for line in lines:
        if line.strip().endswith('))') and line.count('(') < line.count(')'):
            line = line.rstrip(')') + ')'
        fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)

def fix_indentation_errors(content):
    """Fix indentation errors in content"""
    lines = content.split('\n')
    fixed_lines = []
    in_class = False
    class_indent = 0
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        if stripped.startswith('class '):
            in_class = True
            class_indent = len(line) - len(line.lstrip())
            fixed_lines.append(line)
            continue
        
        if in_class and stripped.startswith(('@pytest.mark.asyncio', 'async def', 'def ')):
            if stripped.startswith('@pytest.mark.asyncio'):
                fixed_line = ' ' * (class_indent + 4) + stripped
                fixed_lines.append(fixed_line)
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if next_line.startswith(('async def', 'def ')):
                        lines[i + 1] = ' ' * (class_indent + 4) + next_line
            elif stripped.startswith(('async def', 'def ')):
                fixed_line = ' ' * (class_indent + 4) + stripped
                fixed_lines.append(fixed_line)
                continue
        
        fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)

def fix_async_await_errors(content):
    """Fix async/await errors in content"""
    lines = content.split('\n')
    fixed_lines = []
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        if 'await ' in stripped and not any('async def' in lines[j] for j in range(max(0, i-10), i)):
            for j in range(i-1, max(0, i-10), -1):
                if 'def ' in lines[j] and 'async def' not in lines[j]:
                    lines[j] = lines[j].replace('def ', 'async def ')
                    break
        
        if 'eawait ' in stripped:
            line = line.replace('eawait ', 'await ')
        
        fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)

def scan_and_fix_remaining_errors():
    """Scan all test files for remaining syntax errors"""
    test_dir = 'tests'
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
                    error_files.append((filepath, e.lineno, e.msg))
                except Exception:
                    pass
    
    print(f"\n📊 Found {len(error_files)} files with syntax errors")
    
    for filepath, lineno, msg in error_files[:15]:  # Fix top 15
        print(f"  {filepath}:{lineno} - {msg}")
        
        try:
            with open(filepath, 'r') as f:
                content = f.read()
            
            content = fix_common_syntax_issues(content)
            
            try:
                ast.parse(content)
                with open(filepath, 'w') as f:
                    f.write(content)
                print(f"    ✅ Fixed {filepath}")
            except SyntaxError:
                print(f"    ❌ Still has errors: {filepath}")
                
        except Exception as e:
            print(f"    ⚠️  Error: {e}")

def fix_common_syntax_issues(content):
    """Apply common syntax fixes"""
    content = re.sub(r'eawait\s+', 'await ', content)  # Fix eawait typo
    content = re.sub(r'\)\)', ')', content)  # Fix double closing parens
    content = re.sub(r'async\s+def\s+async\s+def', 'async def', content)  # Fix duplicate async
    
    return content

if __name__ == "__main__":
    fix_all_syntax_errors()
