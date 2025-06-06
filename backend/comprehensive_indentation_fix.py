#!/usr/bin/env python3
"""
Comprehensive Indentation Fix Script
Fixes all indentation errors preventing test collection in CI
"""

import os
import re
import glob
import ast
from pathlib import Path

def fix_comprehensive_indentation():
    """Fix all indentation issues systematically"""
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
                r'(@pytest\.mark\.asyncio)\s*\n\s+async def',
                r'\1\nasync def',
                content,
                flags=re.MULTILINE
            )
            
            content = re.sub(
                r'(@pytest\.fixture[^\n]*)\s*\n\s+def',
                r'\1\ndef',
                content,
                flags=re.MULTILINE
            )
            
            content = re.sub(
                r'(@pytest\.mark\.asyncio)\s*\n\s+async def (test_\w+)',
                r'\1\n    async def \2',
                content,
                flags=re.MULTILINE
            )
            
            lines = content.split('\n')
            fixed_lines = []
            in_class = False
            
            for i, line in enumerate(lines):
                if line.strip().startswith('class ') and line.strip().endswith(':'):
                    in_class = True
                    fixed_lines.append(line)
                    continue
                
                if in_class and line.strip().startswith('@pytest.mark.asyncio'):
                    fixed_lines.append('    ' + line.strip())
                    if i + 1 < len(lines):
                        next_line = lines[i + 1]
                        if next_line.strip().startswith('async def'):
                            fixed_lines.append('    ' + next_line.strip())
                            i += 1  # Skip the next line as we've processed it
                            continue
                elif in_class and line.strip().startswith('async def test_'):
                    fixed_lines.append('    ' + line.strip())
                elif in_class and line.strip().startswith('def test_'):
                    fixed_lines.append('    ' + line.strip())
                else:
                    fixed_lines.append(line)
                
                if line.strip() and not line.startswith(' ') and not line.startswith('\t') and not line.strip().startswith('#'):
                    if not line.strip().startswith('class ') and not line.strip().startswith('@') and not line.strip().startswith('import') and not line.strip().startswith('from'):
                        in_class = False
            
            content = '\n'.join(fixed_lines)
            
            try:
                ast.parse(content)
                syntax_valid = True
            except SyntaxError as e:
                print(f"⚠️  Syntax error in {test_file}: {e}")
                syntax_valid = False
            
            if content != original_content and syntax_valid:
                with open(test_file, 'w') as f:
                    f.write(content)
                fixed_files.append(test_file)
                print(f"✅ Fixed indentation in {test_file}")
                
        except Exception as e:
            print(f"❌ Error fixing {test_file}: {e}")
    
    return fixed_files

def validate_all_test_files():
    """Validate syntax of all test files"""
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
            print(f"❌ Syntax error in {test_file}: {e}")
        except Exception as e:
            invalid_files.append((test_file, str(e)))
            print(f"❌ Error in {test_file}: {e}")
    
    return valid_files, invalid_files

if __name__ == "__main__":
    print("🔧 Comprehensive Indentation Fix")
    print("=" * 50)
    
    fixed_files = fix_comprehensive_indentation()
    
    print(f"\n📊 Fixed {len(fixed_files)} files")
    
    print("\n🔍 Validating all test files...")
    valid_files, invalid_files = validate_all_test_files()
    
    print(f"\n📈 Results:")
    print(f"✅ Valid files: {len(valid_files)}")
    print(f"❌ Invalid files: {len(invalid_files)}")
    
    if invalid_files:
        print("\n❌ Files with remaining issues:")
        for file, error in invalid_files[:10]:  # Show first 10
            print(f"  - {file}: {error}")
    
    if len(invalid_files) == 0:
        print("\n🎉 All test files have valid syntax!")
        print("✅ Ready for CI test collection")
    else:
        print(f"\n⚠️  {len(invalid_files)} files still need attention")
