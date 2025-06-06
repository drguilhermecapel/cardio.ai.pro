#!/usr/bin/env python3
"""
Critical Test Issues Fixer - Based on CI Analysis
Fixes the most common issues causing 129 test failures
"""

import os
import re
import glob

def fix_method_names():
    """Fix incorrect method names in test files"""
    test_files = glob.glob("tests/**/*.py", recursive=True)
    
    fixes = [
        (r'\.un_load_model\(', '.unload_model('),
        (r'service\._analyze_ecg_comprehensive', 'service.analyze_ecg_comprehensive'),
        (r'validate_signal\(sample_signal, 500\)', 'validate_signal(sample_signal)'),
        (r'\.load_model\(', '._load_model('),
        (r'get_analysis\(', 'get_by_id('),
    ]
    
    for test_file in test_files:
        if 'conftest' in test_file:
            continue
            
        try:
            with open(test_file, 'r') as f:
                content = f.read()
            
            original_content = content
            for pattern, replacement in fixes:
                content = re.sub(pattern, replacement, content)
            
            if content != original_content:
                with open(test_file, 'w') as f:
                    f.write(content)
                print(f"Fixed method names in {test_file}")
                
        except Exception as e:
            print(f"Error fixing {test_file}: {e}")

def fix_async_decorators():
    """Add missing @pytest.mark.asyncio decorators"""
    test_files = glob.glob("tests/**/*.py", recursive=True)
    
    for test_file in test_files:
        if 'conftest' in test_file:
            continue
            
        try:
            with open(test_file, 'r') as f:
                lines = f.readlines()
            
            modified = False
            new_lines = []
            
            for i, line in enumerate(lines):
                if line.strip().startswith('async def test_'):
                    prev_line = lines[i-1].strip() if i > 0 else ""
                    if '@pytest.mark.asyncio' not in prev_line:
                        new_lines.append('@pytest.mark.asyncio\n')
                        modified = True
                
                new_lines.append(line)
            
            if modified:
                with open(test_file, 'w') as f:
                    f.writelines(new_lines)
                print(f"Added async decorators to {test_file}")
                
        except Exception as e:
            print(f"Error fixing async decorators in {test_file}: {e}")

if __name__ == "__main__":
    print("🔧 Fixing critical test issues...")
    fix_method_names()
    fix_async_decorators()
    print("✅ Critical fixes applied!")
