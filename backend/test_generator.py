#!/usr/bin/env python3
"""
Automated test generator for CardioAI Pro
Helps quickly create test templates for uncovered code
"""

import ast
import os
from pathlib import Path
from typing import List, Set

class TestGenerator:
    def __init__(self, source_file: Path):
        self.source_file = source_file
        self.module_name = source_file.stem
        self.class_names: Set[str] = set()
        self.methods: List[tuple] = []
        
    def analyze_source(self):
        """Analyze source file to extract classes and methods"""
        with open(self.source_file, 'r') as f:
            tree = ast.parse(f.read())
            
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                self.class_names.add(node.name)
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        is_async = isinstance(item, ast.AsyncFunctionDef)
                        self.methods.append((node.name, item.name, is_async))
    
    def generate_test_template(self) -> str:
        """Generate test template for the source file"""
        self.analyze_source()
        
        template = f'''"""
Tests for {self.module_name}
Generated test template - implement test logic
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
import numpy as np
from datetime import datetime

from app.{self.source_file.parent.name}.{self.module_name} import {', '.join(self.class_names)}

'''
        
        class_methods = {}
        for class_name, method_name, is_async in self.methods:
            if class_name not in class_methods:
                class_methods[class_name] = []
            class_methods[class_name].append((method_name, is_async))
        
        for class_name, methods in class_methods.items():
            template += f'\n\nclass Test{class_name}:\n'
            template += f'    """Test cases for {class_name}"""\n\n'
            
            template += '    @pytest.fixture\n'
            template += f'    def {class_name.lower()}_instance(self):\n'
            template += f'        """Create {class_name} instance for testing"""\n'
            template += f'        # TODO: Add proper initialization\n'
            template += f'        return {class_name}()\n\n'
            
            for method_name, is_async in methods:
                if method_name.startswith('_') and not method_name.startswith('__'):
                    continue  # Skip private methods unless critical
                    
                if is_async:
                    template += '    @pytest.mark.asyncio\n'
                    template += f'    async def test_{method_name}(self, {class_name.lower()}_instance):\n'
                else:
                    template += f'    def test_{method_name}(self, {class_name.lower()}_instance):\n'
                
                template += f'        """Test {method_name} method"""\n'
                template += '        # Arrange\n'
                template += '        # TODO: Set up test data and mocks\n\n'
                template += '        # Act\n'
                template += f'        # result = {"await " if is_async else ""}{class_name.lower()}_instance.{method_name}()\n\n'
                template += '        # Assert\n'
                template += '        # TODO: Add assertions\n'
                template += '        assert True  # Replace with actual assertion\n\n'
            
            template += f'    def test_{class_name.lower()}_edge_cases(self, {class_name.lower()}_instance):\n'
            template += '        """Test edge cases and error handling"""\n'
            template += '        # TODO: Test boundary conditions\n'
            template += '        pass\n\n'
            
            template += f'    def test_{class_name.lower()}_integration(self):\n'
            template += '        """Test integration with other components"""\n'
            template += '        # TODO: Test realistic scenarios\n'
            template += '        pass\n'
        
        return template

def generate_tests_for_low_coverage_files():
    """Generate test templates for files with low coverage"""
    priority_files = [
        'app/services/ml_model_service.py',
        'app/services/validation_service.py',
        'app/services/notification_service.py',
        'app/repositories/ecg_repository.py',
        'app/api/v1/endpoints/ecg_analysis.py',
    ]
    
    for file_path in priority_files:
        source_file = Path(file_path)
        if source_file.exists():
            generator = TestGenerator(source_file)
            test_content = generator.generate_test_template()
            
            test_file = Path(f'tests/test_{source_file.stem}_generated.py')
            with open(test_file, 'w') as f:
                f.write(test_content)
            
            print(f"Generated test template: {test_file}")

if __name__ == "__main__":
    generate_tests_for_low_coverage_files()
    print("\nTest templates generated! Remember to:")
    print("1. Implement the TODO sections")
    print("2. Add specific test data")
    print("3. Mock external dependencies")
    print("4. Test error conditions")
    print("5. Add regulatory compliance checks")
