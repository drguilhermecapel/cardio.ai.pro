#!/usr/bin/env python3
import os
import shutil

def cleanup_backend():
    """Clean up backend directory for deployment."""
    cleanup_dirs = [
        '.pytest_cache',
        'htmlcov',
        '__pycache__',
        'dist',
        'build',
        '.coverage',
        '*.egg-info'
    ]
    
    for root, dirs, files in os.walk('.'):
        for dir_name in dirs[:]:
            if dir_name in cleanup_dirs or dir_name == '__pycache__':
                dir_path = os.path.join(root, dir_name)
                print(f"Removing directory: {dir_path}")
                shutil.rmtree(dir_path, ignore_errors=True)
                dirs.remove(dir_name)
        
        for file_name in files:
            if file_name.endswith('.pyc') or file_name == '.coverage':
                file_path = os.path.join(root, file_name)
                print(f"Removing file: {file_path}")
                os.remove(file_path)

if __name__ == "__main__":
    cleanup_backend()
    print("Cleanup completed")
