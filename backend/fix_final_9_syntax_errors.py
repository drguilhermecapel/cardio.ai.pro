#!/usr/bin/env python3
"""
Fix the final 9 remaining syntax errors manually
Based on the latest error report
"""

import os
import ast

def fix_final_syntax_errors():
    """Fix the remaining 9 syntax errors"""
    
    files_to_fix = {
        'tests/test_hybrid_ecg_service_critical_new.py': 'unexpected unindent',
        'tests/test_final_80_coverage_focused.py': 'unmatched )',
        'tests/test_corrected_critical_services.py': 'unexpected unindent',
        'tests/test_major_services_coverage.py': 'unexpected unindent',
        'tests/test_80_coverage_final_strategic.py': 'unexpected unindent',
        'tests/test_targeted_high_coverage.py': 'unexpected unindent',
        'tests/test_ml_model_service_phase2.py': 'unexpected unindent',
        'tests/test_hybrid_ecg_additional_coverage.py': 'unexpected unindent',
        'tests/test_hybrid_ecg_service_clean.py': 'unexpected unindent'
    }
    
    print("🔧 Fixing final 9 syntax errors...")
    
    for filepath, error_type in files_to_fix.items():
        if os.path.exists(filepath):
            print(f"Processing {filepath} ({error_type})")
            fix_file_syntax(filepath, error_type)
        else:
            print(f"⚠️  File not found: {filepath}")

def fix_file_syntax(filepath, error_type):
    """Fix specific syntax error in file"""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        if error_type == 'unmatched )':
            content = fix_unmatched_parentheses(content)
        elif error_type == 'unexpected unindent':
            content = fix_indentation_errors(content)
        
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
        if ')' in line and line.count(')') > line.count('('):
            if line.strip().endswith('))'):
                line = line.rstrip(')') + ')'
            elif ')' in line:
                open_count = line.count('(')
                close_count = line.count(')')
                if close_count > open_count:
                    extra_closes = close_count - open_count
                    for _ in range(extra_closes):
                        line = line[::-1].replace(')', '', 1)[::-1]
        
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
        
        if stripped and not line.startswith(' ') and not stripped.startswith('#') and not stripped.startswith('class '):
            if not stripped.startswith(('import ', 'from ', '"""', "'''", '@')):
                in_class = False
        
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
        
        fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)

def validate_all_fixes():
    """Validate that all fixes were successful"""
    print("\n🔍 Final validation...")
    
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
    print("🚀 Starting final syntax error fixes...")
    
    fix_final_syntax_errors()
    
    final_error_count = validate_all_fixes()
    
    if final_error_count == 0:
        print("\n✅ SUCCESS: All syntax errors have been resolved!")
        print("🎯 Ready to commit and trigger CI re-run")
    else:
        print(f"\n⚠️  {final_error_count} errors remain - additional intervention required")
