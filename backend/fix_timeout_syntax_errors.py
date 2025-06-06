#!/usr/bin/env python3
"""
Fix syntax errors caused by incorrect timeout decorator placement
Target: Fix 'async @pytest.mark.timeout(30)' syntax errors
"""

import os
import re
import glob
from pathlib import Path

def fix_timeout_syntax_errors():
    """Fix syntax errors where timeout decorators are placed after async keyword"""
    test_files = glob.glob("tests/**/*.py", recursive=True)
    
    fixed_files = []
    
    for test_file in test_files:
        if 'conftest' in test_file or '__pycache__' in test_file:
            continue
            
        try:
            with open(test_file, 'r') as f:
                content = f.read()
            
            original_content = content
            
            content = re.sub(
                r'(\s+)async\s+@pytest\.mark\.timeout\(30\)\s*\n(\s+)(def\s+test_[^(]+\([^)]*\):)',
                r'\1@pytest.mark.timeout(30)\n\1async \3',
                content
            )
            
            content = re.sub(
                r'(\s+)async\s+@pytest\.mark\.timeout\(30\)\s+(def\s+test_[^(]+\([^)]*\):)',
                r'\1@pytest.mark.timeout(30)\n\1async \2',
                content
            )
            
            if content != original_content:
                with open(test_file, 'w') as f:
                    f.write(content)
                fixed_files.append(test_file)
                print(f"✅ Fixed timeout syntax errors in {test_file}")
                
        except Exception as e:
            print(f"Error processing {test_file}: {e}")
    
    return fixed_files

if __name__ == "__main__":
    print("🚀 Fixing timeout decorator syntax errors...")
    
    fixed_files = fix_timeout_syntax_errors()
    
    print(f"\n✅ Fixed syntax errors in {len(fixed_files)} files")
    print("📊 Ready for CI re-run without syntax errors")
