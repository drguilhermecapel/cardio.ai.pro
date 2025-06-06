#!/usr/bin/env python3
"""
Fix remaining 18 syntax errors based on categorization analysis
Targets: 16 indentation errors + 2 unmatched parentheses
"""

import os
import re
import ast

def fix_indentation_errors():
    """Fix all 16 indentation errors in test files"""
    
    indentation_files = [
        'tests/test_ecg_service_phase2.py',
        'tests/test_hybrid_ecg_service_95_coverage.py', 
        'tests/test_notification_service_generated.py',
        'tests/test_validation_service_phase2.py',
        'tests/test_hybrid_ecg_service_simple.py',
        'tests/test_critical_low_coverage_80_target.py',
        'tests/test_hybrid_ecg_service_corrected_signatures.py',
        'tests/test_coverage_maximizer.py',
        'tests/test_hybrid_ecg_service_critical_new.py'
    ]
    
    print("🔧 Fixing indentation errors...")
    
    for filepath in indentation_files:
        if os.path.exists(filepath):
            print(f"Processing {filepath}")
            fix_file_indentation(filepath)
        else:
            print(f"⚠️  File not found: {filepath}")

def fix_file_indentation(filepath):
    """Fix indentation in a specific file"""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        original_content = content
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
            
            if in_class and stripped.startswith(('@pytest.fixture', '@pytest.mark.asyncio', 'def ', 'async def')):
                if stripped.startswith('@pytest.fixture'):
                    fixed_line = ' ' * (class_indent + 4) + stripped
                    fixed_lines.append(fixed_line)
                    if i + 1 < len(lines):
                        next_line = lines[i + 1].strip()
                        if next_line.startswith(('def ', 'async def')):
                            lines[i + 1] = ' ' * (class_indent + 4) + next_line
                elif stripped.startswith('@pytest.mark.asyncio'):
                    fixed_line = ' ' * (class_indent + 4) + stripped
                    fixed_lines.append(fixed_line)
                    if i + 1 < len(lines):
                        next_line = lines[i + 1].strip()
                        if next_line.startswith('async def'):
                            lines[i + 1] = ' ' * (class_indent + 4) + next_line
                elif stripped.startswith(('def ', 'async def')):
                    fixed_line = ' ' * (class_indent + 4) + stripped
                    fixed_lines.append(fixed_line)
                    continue
            
            if stripped.startswith('class ') and in_class:
                in_class = True
                class_indent = len(line) - len(line.lstrip())
            elif stripped and not line.startswith(' ') and not stripped.startswith('#') and in_class:
                in_class = False
            
            fixed_lines.append(line)
        
        fixed_content = '\n'.join(fixed_lines)
        
        try:
            ast.parse(fixed_content)
            with open(filepath, 'w') as f:
                f.write(fixed_content)
            print(f"✅ Fixed indentation in {filepath}")
        except SyntaxError as e:
            print(f"❌ Still has syntax error in {filepath}: {e}")
            
    except Exception as e:
        print(f"⚠️  Error processing {filepath}: {e}")

def fix_unmatched_parentheses():
    """Fix unmatched parentheses errors"""
    
    parentheses_files = [
        'tests/test_hybrid_ecg_zero_coverage.py'
    ]
    
    print("\n🔧 Fixing unmatched parentheses...")
    
    for filepath in parentheses_files:
        if os.path.exists(filepath):
            print(f"Processing {filepath}")
            fix_file_parentheses(filepath)
        else:
            print(f"⚠️  File not found: {filepath}")

def fix_file_parentheses(filepath):
    """Fix unmatched parentheses in a specific file"""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        lines = content.split('\n')
        fixed_lines = []
        
        for line in lines:
            if ')' in line and line.count(')') > line.count('('):
                if re.search(r'\w+\([^)]*\)\)', line):
                    line = re.sub(r'\)\)', ')', line)
                elif 'validate_signal' in line and line.endswith('))'):
                    line = line.rstrip(')') + ')'
            
            fixed_lines.append(line)
        
        fixed_content = '\n'.join(fixed_lines)
        
        try:
            ast.parse(fixed_content)
            with open(filepath, 'w') as f:
                f.write(fixed_content)
            print(f"✅ Fixed parentheses in {filepath}")
        except SyntaxError as e:
            print(f"❌ Still has syntax error in {filepath}: {e}")
            
    except Exception as e:
        print(f"⚠️  Error processing {filepath}: {e}")

def validate_all_fixes():
    """Validate that all fixes were successful"""
    print("\n🔍 Validating all fixes...")
    
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
    
    print(f'\n📊 Final syntax error count: {error_count}')
    if error_count > 0:
        print('\nRemaining errors:')
        for error in error_files[:10]:
            print(f'  {error}')
    else:
        print('🎉 All syntax errors fixed!')
    
    return error_count

if __name__ == "__main__":
    print("🚀 Starting comprehensive syntax error fixes...")
    
    fix_indentation_errors()
    
    fix_unmatched_parentheses()
    
    final_error_count = validate_all_fixes()
    
    if final_error_count == 0:
        print("\n✅ SUCCESS: All 18 syntax errors have been resolved!")
        print("🎯 Ready to proceed to test collection and coverage measurement")
    else:
        print(f"\n⚠️  {final_error_count} errors remain - manual intervention required")
