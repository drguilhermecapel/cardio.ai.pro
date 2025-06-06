#!/usr/bin/env python3
"""
Fix duplicate @pytest.fixture decorators and remaining syntax errors
Based on CI collection errors analysis
"""

import os
import re
import ast
from pathlib import Path

def fix_duplicate_fixtures_and_syntax():
    """Fix duplicate fixtures and syntax errors in test files"""
    
    duplicate_fixture_files = [
        'tests/test_corrected_critical_services.py',
        'tests/test_coverage_maximizer.py', 
        'tests/test_ecg_service_phase2.py',
        'tests/test_hybrid_ecg_service_simple.py',
        'tests/test_major_services_coverage.py',
        'tests/test_ml_model_service_phase2.py',
        'tests/test_notification_service_generated.py',
        'tests/test_validation_service_phase2.py'
    ]
    
    syntax_error_files = [
        'tests/test_final_80_coverage_focused.py',
        'tests/test_fix_ecg_simple.py',
        'tests/test_fix_notification_simple.py',
        'tests/test_hybrid_ecg_additional_coverage.py',
        'tests/test_hybrid_ecg_direct_import.py',
        'tests/test_hybrid_ecg_service.py',
        'tests/test_hybrid_ecg_service_clean.py',
        'tests/test_hybrid_ecg_service_corrected_signatures.py',
        'tests/test_hybrid_ecg_service_medical_grade.py'
    ]
    
    print("🔧 Fixing duplicate fixtures and syntax errors...")
    
    for filepath in duplicate_fixture_files:
        if os.path.exists(filepath):
            print(f"Processing duplicate fixtures in {filepath}")
            fix_duplicate_fixtures(filepath)
        else:
            print(f"⚠️  File not found: {filepath}")
    
    for filepath in syntax_error_files:
        if os.path.exists(filepath):
            print(f"Processing syntax errors in {filepath}")
            fix_syntax_errors(filepath)
        else:
            print(f"⚠️  File not found: {filepath}")

def fix_duplicate_fixtures(filepath):
    """Remove duplicate @pytest.fixture decorators"""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        original_content = content
        
        lines = content.split('\n')
        fixed_lines = []
        i = 0
        
        while i < len(lines):
            line = lines[i]
            
            if '@pytest.fixture' in line:
                j = i + 1
                while j < len(lines) and (lines[j].strip() == '' or lines[j].strip().startswith('@')):
                    if '@pytest.fixture' in lines[j]:
                        print(f"  Removing duplicate @pytest.fixture at line {j+1}")
                        j += 1
                        break
                    j += 1
                
                fixed_lines.append(line)
                i = j
            else:
                fixed_lines.append(line)
                i += 1
        
        content = '\n'.join(fixed_lines)
        
        try:
            ast.parse(content)
            if content != original_content:
                with open(filepath, 'w') as f:
                    f.write(content)
                print(f"✅ Fixed duplicate fixtures in {filepath}")
            else:
                print(f"ℹ️  No duplicate fixtures found in {filepath}")
        except SyntaxError as e:
            print(f"❌ Syntax error after fixing {filepath}: {e}")
            
    except Exception as e:
        print(f"⚠️  Error processing {filepath}: {e}")

def fix_syntax_errors(filepath):
    """Fix syntax errors like 'await' outside async function"""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        original_content = content
        
        lines = content.split('\n')
        fixed_lines = []
        
        for i, line in enumerate(lines):
            if 'await ' in line and not line.strip().startswith('#'):
                func_line_idx = None
                for j in range(i-1, max(0, i-20), -1):
                    if 'def ' in lines[j]:
                        func_line_idx = j
                        break
                
                if func_line_idx is not None:
                    func_line = lines[func_line_idx]
                    if 'async def' not in func_line and 'def ' in func_line:
                        lines[func_line_idx] = func_line.replace('def ', 'async def ')
                        print(f"  Made function async at line {func_line_idx+1}")
                        
                        decorator_line_idx = func_line_idx - 1
                        while decorator_line_idx >= 0 and lines[decorator_line_idx].strip().startswith('@'):
                            decorator_line_idx -= 1
                        
                        decorator_line_idx += 1
                        if decorator_line_idx < len(lines) and '@pytest.mark.asyncio' not in lines[decorator_line_idx]:
                            indent = len(func_line) - len(func_line.lstrip())
                            lines.insert(decorator_line_idx, ' ' * indent + '@pytest.mark.asyncio')
                            print(f"  Added @pytest.mark.asyncio decorator at line {decorator_line_idx+1}")
            
            fixed_lines.append(line)
        
        content = '\n'.join(lines)
        
        content = re.sub(r'(\s+)v\s*\n\s*await', r'\1await', content, flags=re.MULTILINE)
        
        try:
            ast.parse(content)
            if content != original_content:
                with open(filepath, 'w') as f:
                    f.write(content)
                print(f"✅ Fixed syntax errors in {filepath}")
            else:
                print(f"ℹ️  No syntax errors found in {filepath}")
        except SyntaxError as e:
            print(f"❌ Still has syntax error in {filepath}: {e}")
            
    except Exception as e:
        print(f"⚠️  Error processing {filepath}: {e}")

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
    print("🚀 Starting duplicate fixture and syntax error fixes...")
    
    fix_duplicate_fixtures_and_syntax()
    
    final_error_count = validate_all_fixes()
    
    if final_error_count == 0:
        print("\n✅ SUCCESS: All collection errors have been resolved!")
        print("🎯 Ready to commit and trigger CI re-run")
    else:
        print(f"\n⚠️  {final_error_count} errors remain - additional intervention required")
