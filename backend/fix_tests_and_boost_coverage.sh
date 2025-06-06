#!/bin/bash

echo "🔧 Fixing Event Loop issues..."

pip install pytest-asyncio==0.23.0 pytest-mock pytest-cov

cat > pytest.ini << EOF
[pytest]
asyncio_mode = auto
asyncio_default_fixture_loop_scope = function
markers =
    asyncio: marks tests as async
    slow: marks tests as slow
EOF

echo "🔧 Fixing method name issues..."

find tests -name "*.py" -exec sed -i 's/load_model/_load_model/g' {} \;

find tests -name "*.py" -exec sed -i 's/validate_signal(sample_signal, 500)/validate_signal(sample_signal)/g' {} \;

python << 'EOF'
import os
import re

def fix_missing_awaits(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    
    patterns = [
        (r'result = (processor\.preprocess_signal\()', r'result = await \1'),
        (r'assert isinstance\((processor\.preprocess_signal\(.*?\)), np\.ndarray\)', 
         r'result = await \1\nassert isinstance(result, np.ndarray)'),
    ]
    
    for pattern, replacement in patterns:
        content = re.sub(pattern, replacement, content)
    
    with open(filepath, 'w') as f:
        f.write(content)

for root, dirs, files in os.walk('tests'):
    for file in files:
        if file.endswith('.py'):
            fix_missing_awaits(os.path.join(root, file))
EOF

echo "✅ Event loop fixes applied!"
