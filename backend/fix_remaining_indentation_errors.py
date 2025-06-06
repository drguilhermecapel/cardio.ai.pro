#!/usr/bin/env python3
"""
Fix remaining indentation errors in test files
"""

import os
import re
import glob

def fix_indentation_errors():
    """Fix all remaining indentation errors"""
    test_files = glob.glob("tests/**/*.py", recursive=True)
    
    for test_file in test_files:
        if 'conftest' in test_file or '__pycache__' in test_file:
            continue
            
        try:
            with open(test_file, 'r') as f:
                content = f.read()
            
            original_content = content
            
            content = re.sub(
                r'(@pytest\.mark\.asyncio)\s*\n\s+async def',
                r'\1\nasync def',
                content,
                flags=re.MULTILINE
            )
            
            content = re.sub(
                r'(@pytest\.fixture)\s*\n\s+def',
                r'\1\ndef',
                content,
                flags=re.MULTILINE
            )
            
            if content != original_content:
                with open(test_file, 'w') as f:
                    f.write(content)
                print(f"✅ Fixed indentation in {test_file}")
                
        except Exception as e:
            print(f"Error fixing {test_file}: {e}")

if __name__ == "__main__":
    print("🔧 Fixing Remaining Indentation Errors")
    print("=" * 40)
    
    fix_indentation_errors()
    
    print("\n✅ Indentation Fixes Complete!")
