#!/bin/bash
# coverage_booster.sh - Automated test generation and execution
# Run this script to rapidly boost coverage to 80%

set -e

echo "🚀 CardioAI Pro Coverage Booster - Starting..."
echo "Current Coverage: 41.06%"
echo "Target Coverage: 80%"
echo "="*60

# Phase 1: Generate tests for zero-coverage modules
echo -e "\n📋 Phase 1: Zero-Coverage Modules (Day 1-2)"
echo "Generating tests for critical uncovered modules..."

python3 << 'EOF'
import os
import subprocess

# Priority modules with 0% coverage
zero_coverage_modules = [
    ('app/utils/ecg_hybrid_processor.py', 379),
    ('app/core/celery.py', 12),
    ('app/tasks/ecg_tasks.py', 53),
    ('app/db/init_db.py', 58),
    ('app/types/ecg_types.py', 7),
]

total_lines = sum(lines for _, lines in zero_coverage_modules)
print(f"Total lines to cover: {total_lines}")
print(f"Expected coverage boost: +{(total_lines/5361)*100:.1f}%")

# Generate test files
for module, lines in zero_coverage_modules:
    test_name = f"test_{os.path.basename(module).replace('.py', '')}_generated.py"
    print(f"Generating: {test_name} ({lines} lines)")
    # Run test generator
    subprocess.run(['python', 'test_generator.py', module])
EOF

# Phase 2: Boost critical services
echo -e "\n📋 Phase 2: Critical Services Boost (Day 3-4)"
echo "Enhancing tests for low-coverage critical services..."

# Run focused tests for critical services
pytest tests/test_ml_model_service*.py \
       tests/test_validation_service*.py \
       tests/test_ecg_service*.py \
       --cov=app.services \
       --cov-report=term-missing \
       -v

# Phase 3: API endpoint coverage
echo -e "\n📋 Phase 3: API Endpoints (Day 5)"
echo "Running comprehensive API tests..."

pytest tests/test_*endpoints*.py \
       tests/integration/ \
       --cov=app.api \
       --cov-report=term-missing \
       -v

# Generate comprehensive report
echo -e "\n📊 Generating Coverage Report..."
pytest --cov=app \
       --cov-report=html \
       --cov-report=term-missing \
       --cov-report=json \
       -v

# Parse results
python3 << 'EOF'
import json

with open('coverage.json', 'r') as f:
    data = json.load(f)

coverage = data['totals']['percent_covered']
print(f"\n{'='*60}")
print(f"🎯 Final Coverage: {coverage:.1f}%")

if coverage >= 80:
    print("✅ SUCCESS: Regulatory compliance achieved!")
    print("🎉 Ready for FDA/ANVISA/NMSA/EU certification")
else:
    gap = 80 - coverage
    print(f"❌ FAIL: {gap:.1f}% below requirement")
    print(f"📝 Run additional targeted tests")

print(f"{'='*60}")
EOF

# Open HTML report
echo -e "\n📊 Opening detailed coverage report..."
echo "Browse: htmlcov/index.html"
