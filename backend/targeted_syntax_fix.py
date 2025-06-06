#!/usr/bin/env python3
"""
Targeted Syntax Fix Script
Fixes the most common syntax error patterns identified in the comprehensive analysis
"""

import os
import re
import glob
import ast
from pathlib import Path

def fix_unexpected_unindent_errors():
    """Fix unexpected unindent errors - most common pattern (70% of errors)"""
    
    unindent_files = [
        'tests/test_low_coverage_services_targeted.py',
        'tests/test_hybrid_ecg_service_critical.py', 
        'tests/test_ecg_hybrid_processor_critical.py',
        'tests/test_critical_zero_coverage_services.py',
        'tests/test_hybrid_ecg_direct_import.py',
        'tests/test_hybrid_ecg_service_medical_grade.py',
        'tests/test_ml_model_service_generated.py',
        'tests/test_ecg_repository_generated.py',
        'tests/test_validation_service_generated.py',
        'tests/test_ecg_service_phase2.py',
        'tests/test_hybrid_ecg_service_95_coverage.py',
        'tests/test_notification_service_generated.py',
        'tests/test_validation_service_phase2.py',
        'tests/test_hybrid_ecg_service_simple.py',
        'tests/test_hybrid_ecg_service_corrected_signatures.py',
        'tests/test_coverage_maximizer.py',
        'tests/test_hybrid_ecg_service_critical_new.py',
        'tests/test_corrected_critical_services.py',
        'tests/test_major_services_coverage.py',
        'tests/test_ml_model_service_phase2.py',
        'tests/test_hybrid_ecg_additional_coverage.py',
        'tests/test_hybrid_ecg_service_clean.py'
    ]
    
    fixed_count = 0
    
    for file_path in unindent_files:
        if not os.path.exists(file_path):
            continue
            
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            original_content = content
            lines = content.split('\n')
            fixed_lines = []
            
            for i, line in enumerate(lines):
                if line.strip() and not line.startswith(' ') and not line.startswith('\t'):
                    if i > 0:
                        prev_line = lines[i-1].strip()
                        if (prev_line.endswith(':') or 
                            prev_line.startswith('def ') or 
                            prev_line.startswith('class ') or
                            prev_line.startswith('if ') or
                            prev_line.startswith('try:') or
                            prev_line.startswith('except') or
                            prev_line.startswith('with ')):
                            if not line.startswith('#') and not line.startswith('import') and not line.startswith('from'):
                                line = '    ' + line
                
                fixed_lines.append(line)
            
            content = '\n'.join(fixed_lines)
            
            try:
                ast.parse(content)
                if content != original_content:
                    with open(file_path, 'w') as f:
                        f.write(content)
                    print(f"✅ Fixed unexpected unindent in {file_path}")
                    fixed_count += 1
            except SyntaxError:
                pass
                
        except Exception as e:
            print(f"❌ Error fixing {file_path}: {e}")
    
    return fixed_count

def fix_expected_indented_block_errors():
    """Fix expected indented block errors - 20% of errors"""
    
    block_files = [
        'tests/test_services_basic_coverage.py',
        'tests/test_simple_method_coverage.py', 
        'tests/test_hybrid_ecg_service.py',
        'tests/test_ecg_hybrid_processor_coverage.py',
        'tests/test_hybrid_ecg_service_real_methods.py',
        'tests/test_critical_low_coverage_80_target.py',
        'tests/test_80_coverage_final_strategic.py',
        'tests/test_targeted_high_coverage.py'
    ]
    
    fixed_count = 0
    
    for file_path in block_files:
        if not os.path.exists(file_path):
            continue
            
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            original_content = content
            lines = content.split('\n')
            fixed_lines = []
            
            for i, line in enumerate(lines):
                fixed_lines.append(line)
                
                if line.strip().endswith(':'):
                    if i + 1 < len(lines):
                        next_line = lines[i + 1]
                        if next_line.strip() == '' or not next_line.startswith('    '):
                            indent = len(line) - len(line.lstrip()) + 4
                            fixed_lines.append(' ' * indent + 'pass')
            
            content = '\n'.join(fixed_lines)
            
            try:
                ast.parse(content)
                if content != original_content:
                    with open(file_path, 'w') as f:
                        f.write(content)
                    print(f"✅ Fixed expected indented block in {file_path}")
                    fixed_count += 1
            except SyntaxError:
                pass
                
        except Exception as e:
            print(f"❌ Error fixing {file_path}: {e}")
    
    return fixed_count

def fix_parentheses_errors():
    """Fix unmatched parentheses errors - 10% of errors"""
    
    paren_files = [
        'tests/test_simple_80_coverage_final.py',
        'tests/test_hybrid_ecg_zero_coverage.py',
        'tests/test_final_80_coverage_focused.py'
    ]
    
    fixed_count = 0
    
    for file_path in paren_files:
        if not os.path.exists(file_path):
            continue
            
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            original_content = content
            
            open_parens = content.count('(')
            close_parens = content.count(')')
            
            if open_parens > close_parens:
                content += ')' * (open_parens - close_parens)
            elif close_parens > open_parens:
                pass
            
            try:
                ast.parse(content)
                if content != original_content:
                    with open(file_path, 'w') as f:
                        f.write(content)
                    print(f"✅ Fixed parentheses in {file_path}")
                    fixed_count += 1
            except SyntaxError:
                pass
                
        except Exception as e:
            print(f"❌ Error fixing {file_path}: {e}")
    
    return fixed_count

def validate_all_fixes():
    """Validate syntax of all test files after fixes"""
    test_files = glob.glob("tests/**/*.py", recursive=True)
    
    valid_files = []
    invalid_files = []
    
    for test_file in test_files:
        if 'conftest' in test_file or '__pycache__' in test_file:
            continue
            
        try:
            with open(test_file, 'r') as f:
                content = f.read()
            
            ast.parse(content)
            valid_files.append(test_file)
            
        except SyntaxError as e:
            invalid_files.append((test_file, str(e)))
        except Exception as e:
            invalid_files.append((test_file, str(e)))
    
    return valid_files, invalid_files

if __name__ == "__main__":
    print("🔧 Targeted Syntax Fix - Phase 1")
    print("=" * 50)
    
    print("\n1️⃣ Fixing unexpected unindent errors...")
    unindent_fixed = fix_unexpected_unindent_errors()
    print(f"   Fixed {unindent_fixed} files with unindent errors")
    
    print("\n2️⃣ Fixing expected indented block errors...")
    block_fixed = fix_expected_indented_block_errors()
    print(f"   Fixed {block_fixed} files with block errors")
    
    print("\n3️⃣ Fixing parentheses errors...")
    paren_fixed = fix_parentheses_errors()
    print(f"   Fixed {paren_fixed} files with parentheses errors")
    
    print("\n🔍 Validating all test files...")
    valid_files, invalid_files = validate_all_fixes()
    
    print(f"\n📈 Results:")
    print(f"✅ Valid files: {len(valid_files)}")
    print(f"❌ Invalid files: {len(invalid_files)}")
    
    if invalid_files:
        print(f"\n❌ Files still needing attention ({len(invalid_files)}):")
        for file, error in invalid_files[:10]:  # Show first 10
            print(f"  - {file}: {error}")
    
    total_fixed = unindent_fixed + block_fixed + paren_fixed
    print(f"\n🎯 Total files fixed: {total_fixed}")
    
    if len(invalid_files) == 0:
        print("\n🎉 ALL TEST FILES HAVE VALID SYNTAX!")
        print("✅ Ready for CI test collection")
    elif len(invalid_files) < 10:
        print(f"\n⚠️  Only {len(invalid_files)} files still need attention")
        print("📝 Significant progress made!")
    else:
        print(f"\n⚠️  {len(invalid_files)} files still need attention")
        print("📝 Continue with manual fixes for remaining issues")
