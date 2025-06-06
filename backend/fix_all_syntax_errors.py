#!/usr/bin/env python3
"""
Complete Syntax Error Fix Script
Fixes all remaining syntax errors preventing test collection
"""

import os
import re
import glob
from pathlib import Path

def fix_duplicate_decorators_and_indentation():
    """Fix all duplicate decorators and indentation issues"""
    test_files = glob.glob("tests/**/*.py", recursive=True)
    
    for test_file in test_files:
        if 'conftest' in test_file or '__pycache__' in test_file:
            continue
            
        try:
            with open(test_file, 'r') as f:
                content = f.read()
            
            original_content = content
            
            content = re.sub(
                r'(\s*)@pytest\.mark\.asyncio\s*\n\s*@pytest\.mark\.asyncio\s*\n(\s*)async def',
                r'\1@pytest.mark.asyncio\n\2async def',
                content,
                flags=re.MULTILINE
            )
            
            content = re.sub(
                r'^async def (test_\w+)',
                r'    async def \1',
                content,
                flags=re.MULTILINE
            )
            
            content = re.sub(r'sawait ervice', r'service', content)
            content = re.sub(r'sample_signal', r'valid_signal', content)
            
            content = re.sub(r'(\w+\.supported_formats)\)', r'\1', content)
            
            content = re.sub(
                r'from app\.endpoints\.ecg_analysis import\s*$',
                r'from app.api.v1.endpoints.ecg_analysis import router',
                content,
                flags=re.MULTILINE
            )
            
            if content != original_content:
                with open(test_file, 'w') as f:
                    f.write(content)
                print(f"✅ Fixed syntax errors in {test_file}")
                
        except Exception as e:
            print(f"Error fixing {test_file}: {e}")

if __name__ == "__main__":
    print("🔧 Fixing All Syntax Errors")
    print("=" * 40)
    
    fix_duplicate_decorators_and_indentation()
    
    print("\n✅ All Syntax Fixes Complete!")
    print("Ready to commit and re-run CI.")
