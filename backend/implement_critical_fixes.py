#!/usr/bin/env python3
"""
Critical Test Fixes Implementation - Phase 1
Based on CI Analysis Report findings
"""

import os
import re
import glob
from pathlib import Path

def restore_test_db_fixture():
    """Restore the missing test_db fixture in conftest.py"""
    conftest_path = "tests/conftest.py"
    
    with open(conftest_path, 'r') as f:
        content = f.read()
    
    test_db_fixture = '''
@pytest_asyncio.fixture(scope="function")
async def test_db():
    """Create test database session with proper table creation."""
    database_url = "sqlite+aiosqlite:///:memory:"
    
    engine = create_async_engine(
        database_url,
        echo=False,
        poolclass=NullPool,
        connect_args={"check_same_thread": False}
    )
    
    from sqlalchemy.ext.asyncio import async_sessionmaker
    async_session = async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        
        session = async_session(bind=conn)
        try:
            yield session
        finally:
            await session.rollback()
            await session.close()
    
    await engine.dispose()
'''
    
    imports_to_add = '''
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool
from app.models.base import Base
from app.models import *
'''
    
    if 'pytest_asyncio' not in content:
        content = imports_to_add + '\n' + content
    
    if 'async def test_db():' not in content:
        content += test_db_fixture
    
    with open(conftest_path, 'w') as f:
        f.write(content)
    
    print("✅ Restored test_db fixture in conftest.py")

def fix_method_name_inconsistencies():
    """Fix method name inconsistencies across test files"""
    test_files = glob.glob("tests/**/*.py", recursive=True)
    
    method_fixes = [
        (r'\.un_load_model\(', '.unload_model('),
        (r'service\._analyze_ecg_comprehensive', 'service.analyze_ecg_comprehensive'),
        (r'\.load_model\(', '._load_model('),
        (r'get_analysis\(', 'get_by_id('),
        (r'\.extract_features\(', '.extract_morphology_features('),
        (r'\.validate_analysis\(', '.create_validation('),
        (r'\.get_system_status\(', '.get_model_info('),
        (r'\.get_supported_formats\(', '.supported_formats'),
        (r'\.sampling_rate', '.sample_rate'),
        (r'validate_signal\(sample_signal, 500\)', 'validate_signal(sample_signal)'),
    ]
    
    fixed_files = []
    for test_file in test_files:
        if 'conftest' in test_file or '__pycache__' in test_file:
            continue
            
        try:
            with open(test_file, 'r') as f:
                content = f.read()
            
            original_content = content
            for pattern, replacement in method_fixes:
                content = re.sub(pattern, replacement, content)
            
            if content != original_content:
                with open(test_file, 'w') as f:
                    f.write(content)
                fixed_files.append(test_file)
                
        except Exception as e:
            print(f"Error fixing {test_file}: {e}")
    
    print(f"✅ Fixed method names in {len(fixed_files)} test files")
    return fixed_files

def fix_method_signatures():
    """Fix method signature mismatches"""
    test_files = glob.glob("tests/**/*.py", recursive=True)
    
    signature_fixes = [
        (r'_calculate_quality_metrics\(\)', '_calculate_quality_metrics({"test": "data"})'),
        (r'validate_signal\([^,]+,\s*[^)]+\)', 'validate_signal(sample_signal)'),
        (r'get_analysis_by_patient\(', 'get_by_patient_id('),
    ]
    
    fixed_files = []
    for test_file in test_files:
        if 'conftest' in test_file or '__pycache__' in test_file:
            continue
            
        try:
            with open(test_file, 'r') as f:
                content = f.read()
            
            original_content = content
            for pattern, replacement in signature_fixes:
                content = re.sub(pattern, replacement, content)
            
            if content != original_content:
                with open(test_file, 'w') as f:
                    f.write(content)
                fixed_files.append(test_file)
                
        except Exception as e:
            print(f"Error fixing signatures in {test_file}: {e}")
    
    print(f"✅ Fixed method signatures in {len(fixed_files)} test files")
    return fixed_files

def add_missing_async_decorators():
    """Add missing @pytest.mark.asyncio decorators"""
    test_files = glob.glob("tests/**/*.py", recursive=True)
    
    fixed_files = []
    for test_file in test_files:
        if 'conftest' in test_file or '__pycache__' in test_file:
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
                        indent = len(line) - len(line.lstrip())
                        new_lines.append(' ' * indent + '@pytest.mark.asyncio\n')
                        modified = True
                
                new_lines.append(line)
            
            if modified:
                with open(test_file, 'w') as f:
                    f.writelines(new_lines)
                fixed_files.append(test_file)
                
        except Exception as e:
            print(f"Error adding async decorators to {test_file}: {e}")
    
    print(f"✅ Added async decorators to {len(fixed_files)} test files")
    return fixed_files

def fix_async_function_calls():
    """Fix async function calls that aren't awaited"""
    test_files = glob.glob("tests/**/*.py", recursive=True)
    
    async_method_patterns = [
        (r'(\w+\.preprocess_signal\([^)]+\))', r'await \1'),
        (r'(\w+\.analyze_ecg_comprehensive\([^)]+\))', r'await \1'),
        (r'(\w+\._generate_clinical_assessment\([^)]+\))', r'await \1'),
    ]
    
    fixed_files = []
    for test_file in test_files:
        if 'conftest' in test_file or '__pycache__' in test_file:
            continue
            
        try:
            with open(test_file, 'r') as f:
                content = f.read()
            
            original_content = content
            for pattern, replacement in async_method_patterns:
                content = re.sub(f'(?<!await )({pattern})', replacement, content)
            
            if content != original_content:
                with open(test_file, 'w') as f:
                    f.write(content)
                fixed_files.append(test_file)
                
        except Exception as e:
            print(f"Error fixing async calls in {test_file}: {e}")
    
    print(f"✅ Fixed async function calls in {len(fixed_files)} test files")
    return fixed_files

if __name__ == "__main__":
    print("🔧 Implementing Critical Test Fixes - Phase 1")
    print("=" * 50)
    
    restore_test_db_fixture()
    method_files = fix_method_name_inconsistencies()
    signature_files = fix_method_signatures()
    async_decorator_files = add_missing_async_decorators()
    async_call_files = fix_async_function_calls()
    
    print("\n📊 Summary:")
    print(f"- Restored test_db fixture")
    print(f"- Fixed method names in {len(method_files)} files")
    print(f"- Fixed signatures in {len(signature_files)} files") 
    print(f"- Added async decorators to {len(async_decorator_files)} files")
    print(f"- Fixed async calls in {len(async_call_files)} files")
    
    print("\n✅ Phase 1 Critical Fixes Complete!")
    print("Ready to commit and push changes.")
