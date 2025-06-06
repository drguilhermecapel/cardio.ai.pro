#!/usr/bin/env python3
"""
Comprehensive fix for all timeout syntax errors
Target: Fix ALL remaining 'async @pytest.mark.timeout(30)' patterns
"""

import os
import re
import glob
from pathlib import Path

def fix_comprehensive_timeout_syntax():
    """Fix all remaining timeout syntax errors with comprehensive patterns"""
    test_files = glob.glob("tests/**/*.py", recursive=True)
    
    patterns_to_fix = [
        (r'(\s*)async\s+@pytest\.mark\.timeout\(30\)\s+(def\s+test_[^(]+\([^)]*\):)', 
         r'\1@pytest.mark.timeout(30)\n\1async \2'),
        
        (r'(\s*)async\s+@pytest\.mark\.timeout\(30\)\s*\n(\s*)(def\s+test_[^(]+\([^)]*\):)', 
         r'\1@pytest.mark.timeout(30)\n\1async \3'),
        
        (r'(\s*)async\s+@pytest\.mark\.timeout\(30\)(\s*)', 
         r'\1@pytest.mark.timeout(30)\n\1async\2'),
         
        (r'(\s*)async\s+@pytest\.mark\.timeout\(30\)$', 
         r'\1@pytest.mark.timeout(30)\n\1async'),
    ]
    
    fixed_files = []
    
    for test_file in test_files:
        if 'conftest' in test_file or '__pycache__' in test_file:
            continue
            
        try:
            with open(test_file, 'r') as f:
                content = f.read()
            
            original_content = content
            
            for pattern, replacement in patterns_to_fix:
                content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
            
            content = re.sub(
                r'(\s*)async\s+(@pytest\.mark\.[^)]+\))',
                r'\1\2\n\1async',
                content,
                flags=re.MULTILINE
            )
            
            if content != original_content:
                with open(test_file, 'w') as f:
                    f.write(content)
                fixed_files.append(test_file)
                print(f"✅ Fixed comprehensive timeout syntax in {test_file}")
                
        except Exception as e:
            print(f"Error processing {test_file}: {e}")
    
    return fixed_files

def verify_no_syntax_errors():
    """Verify no async @pytest patterns remain"""
    test_files = glob.glob("tests/**/*.py", recursive=True)
    
    remaining_errors = []
    
    for test_file in test_files:
        if 'conftest' in test_file or '__pycache__' in test_file:
            continue
            
        try:
            with open(test_file, 'r') as f:
                lines = f.readlines()
            
            for i, line in enumerate(lines, 1):
                if re.search(r'async\s+@pytest\.mark', line):
                    remaining_errors.append(f"{test_file}:{i} - {line.strip()}")
                    
        except Exception as e:
            print(f"Error checking {test_file}: {e}")
    
    return remaining_errors

if __name__ == "__main__":
    print("🚀 Starting comprehensive timeout syntax fix...")
    
    fixed_files = fix_comprehensive_timeout_syntax()
    
    print(f"\n✅ Fixed syntax errors in {len(fixed_files)} files")
    
    print("\n🔍 Verifying no syntax errors remain...")
    remaining_errors = verify_no_syntax_errors()
    
    if remaining_errors:
        print(f"\n⚠️  Found {len(remaining_errors)} remaining syntax errors:")
        for error in remaining_errors:
            print(f"  - {error}")
    else:
        print("\n✅ No remaining syntax errors found!")
    
    print("\n📊 Ready for CI re-run with pytest marker configuration")
