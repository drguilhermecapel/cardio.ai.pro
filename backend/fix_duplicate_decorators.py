#!/usr/bin/env python3
"""
Fix duplicate @pytest.fixture decorators in test files
"""
import re
import os

def fix_duplicate_decorators():
    """Remove duplicate @pytest.fixture decorators from test files"""
    
    files_to_fix = [
        'tests/test_corrected_critical_services.py',
        'tests/test_coverage_maximizer.py', 
        'tests/test_hybrid_ecg_service_clean.py',
        'tests/test_hybrid_ecg_service_corrected_signatures.py',
        'tests/test_major_services_coverage.py',
        'tests/test_ml_model_service_phase2.py',
        'tests/test_notification_service_generated.py',
        'tests/test_validation_service_phase2.py'
    ]
    
    fixed_count = 0
    
    for file_path in files_to_fix:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                content = f.read()
            
            original_content = content
            
            pattern = r'@pytest\.fixture\s*\n\s*@pytest\.fixture'
            content = re.sub(pattern, '@pytest.fixture', content)
            
            pattern2 = r'(@pytest\.fixture\s*\n\s*)+@pytest\.fixture'
            content = re.sub(pattern2, '@pytest.fixture', content)
            
            if content != original_content:
                with open(file_path, 'w') as f:
                    f.write(content)
                print(f'✅ Fixed duplicate decorators in {file_path}')
                fixed_count += 1
            else:
                print(f'ℹ️  No duplicate decorators found in {file_path}')
        else:
            print(f'❌ File not found: {file_path}')
    
    print(f'\n🎯 Summary: Fixed {fixed_count} files with duplicate decorators')
    return fixed_count

if __name__ == '__main__':
    fix_duplicate_decorators()
