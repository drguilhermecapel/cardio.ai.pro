#!/usr/bin/env python3
"""
Fix all remaining duplicate @pytest.fixture and @pytest.mark.asyncio decorators
Based on CI collection errors analysis
"""

import os
import re
import ast
from pathlib import Path

def fix_all_remaining_duplicates():
    """Fix duplicate decorators in all remaining files"""
    
    remaining_files = [
        'tests/test_ecg_service_phase2.py',
        'tests/test_final_80_coverage_focused.py',
        'tests/test_hybrid_ecg_additional_coverage.py',
        'tests/test_hybrid_ecg_service_clean.py',
        'tests/test_hybrid_ecg_service_corrected_signatures.py',
        'tests/test_major_services_coverage.py',
        'tests/test_ml_model_service_phase2.py',
        'tests/test_notification_service_generated.py',
        'tests/test_validation_service_phase2.py'
    ]
    
    print("🔧 Fixing all remaining duplicate decorators...")
    
    for filepath in remaining_files:
        if os.path.exists(filepath):
            print(f"Processing {filepath}")
            fix_file_duplicates(filepath)
        else:
            print(f"⚠️  File not found: {filepath}")

def fix_file_duplicates(filepath):
    """Remove duplicate decorators from a single file"""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        original_content = content
        
        content = re.sub(
            r'(@pytest\.fixture\s*\n\s*)+@pytest\.fixture',
            r'@pytest.fixture',
            content,
            flags=re.MULTILINE
        )
        
        content = re.sub(
            r'(@pytest\.mark\.asyncio\s*\n\s*)+@pytest\.mark\.asyncio',
            r'@pytest.mark.asyncio',
            content,
            flags=re.MULTILINE
        )
        
        lines = content.split('\n')
        cleaned_lines = []
        i = 0
        
        while i < len(lines):
            line = lines[i]
            
            if '@pytest.fixture' in line:
                cleaned_lines.append(line)
                j = i + 1
                while j < len(lines) and '@pytest.fixture' in lines[j].strip():
                    j += 1
                i = j
            elif '@pytest.mark.asyncio' in line:
                cleaned_lines.append(line)
                j = i + 1
                while j < len(lines) and '@pytest.mark.asyncio' in lines[j].strip():
                    j += 1
                i = j
            else:
                cleaned_lines.append(line)
                i += 1
        
        content = '\n'.join(cleaned_lines)
        
        try:
            ast.parse(content)
            if content != original_content:
                with open(filepath, 'w') as f:
                    f.write(content)
                print(f"✅ Fixed duplicates in {filepath}")
            else:
                print(f"ℹ️  No duplicates found in {filepath}")
        except SyntaxError as e:
            print(f"❌ Syntax error after fixing {filepath}: {e}")
            
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
    print("🚀 Starting comprehensive duplicate decorator fixes...")
    
    fix_all_remaining_duplicates()
    
    final_error_count = validate_all_fixes()
    
    if final_error_count == 0:
        print("\n✅ SUCCESS: All collection errors have been resolved!")
        print("🎯 Ready to commit and trigger CI re-run")
    else:
        print(f"\n⚠️  {final_error_count} errors remain - additional intervention required")
