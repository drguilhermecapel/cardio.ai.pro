#!/usr/bin/env python3
"""
Coverage monitoring and reporting tool for CardioAI Pro
Tracks progress toward 80% regulatory compliance target
"""

import subprocess
import json
from datetime import datetime
from pathlib import Path

class CoverageMonitor:
    def __init__(self, target_coverage=80.0):
        self.target_coverage = target_coverage
        self.critical_modules = [
            'app/services/ecg_service.py',
            'app/services/ml_model_service.py',
            'app/services/validation_service.py',
            'app/services/hybrid_ecg_service.py',
        ]
    
    def run_coverage(self):
        """Run pytest with coverage and return results"""
        cmd = [
            'pytest', 
            '--cov=app', 
            '--cov-report=json',
            '--cov-report=term-missing',
            '-q'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        with open('coverage.json', 'r') as f:
            coverage_data = json.load(f)
        
        return coverage_data
    
    def generate_report(self):
        """Generate comprehensive coverage report"""
        coverage_data = self.run_coverage()
        
        total_coverage = coverage_data['totals']['percent_covered']
        
        print(f"\n{'='*60}")
        print(f"CardioAI Pro Coverage Report - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(f"{'='*60}\n")
        
        status = "✅ PASS" if total_coverage >= self.target_coverage else "❌ FAIL"
        print(f"Overall Coverage: {total_coverage:.1f}% {status}")
        print(f"Target Coverage: {self.target_coverage}%")
        print(f"Gap: {max(0, self.target_coverage - total_coverage):.1f}%\n")
        
        print("Critical Module Coverage:")
        print("-" * 40)
        
        files = coverage_data['files']
        for module in self.critical_modules:
            if module in files:
                module_cov = files[module]['summary']['percent_covered']
                status = "✅" if module_cov >= 90 else "⚠️" if module_cov >= 70 else "❌"
                print(f"{status} {module}: {module_cov:.1f}%")
        
        print("\n\nFiles Needing Attention (< 50% coverage):")
        print("-" * 40)
        
        low_coverage = [
            (f, data['summary']['percent_covered']) 
            for f, data in files.items() 
            if data['summary']['percent_covered'] < 50
        ]
        
        for file, cov in sorted(low_coverage, key=lambda x: x[1])[:10]:
            print(f"❗ {file}: {cov:.1f}%")
        
        self.save_progress(total_coverage)
        self.show_progress_chart()
    
    def save_progress(self, coverage):
        """Save coverage history"""
        history_file = Path('.coverage_history.json')
        
        if history_file.exists():
            with open(history_file, 'r') as f:
                history = json.load(f)
        else:
            history = []
        
        history.append({
            'date': datetime.now().isoformat(),
            'coverage': coverage
        })
        
        history = history[-30:]
        
        with open(history_file, 'w') as f:
            json.dump(history, f)
    
    def show_progress_chart(self):
        """Show coverage progress over time"""
        history_file = Path('.coverage_history.json')
        
        if not history_file.exists():
            return
        
        with open(history_file, 'r') as f:
            history = json.load(f)
        
        if len(history) < 2:
            return
        
        print("\n\nCoverage Trend (last 10 runs):")
        print("-" * 40)
        
        for entry in history[-10:]:
            date = datetime.fromisoformat(entry['date']).strftime('%m/%d %H:%M')
            cov = entry['coverage']
            bar = '█' * int(cov / 2)
            print(f"{date}: {bar} {cov:.1f}%")
        
        if len(history) >= 2:
            improvement = history[-1]['coverage'] - history[-2]['coverage']
            if improvement > 0:
                print(f"\n📈 Coverage improved by {improvement:.1f}%!")
            elif improvement < 0:
                print(f"\n📉 Coverage decreased by {abs(improvement):.1f}%")
            else:
                print("\n➡️ Coverage unchanged")

if __name__ == "__main__":
    monitor = CoverageMonitor()
    monitor.generate_report()
    
    print("\n\n💡 Next Steps:")
    print("1. Run: python generate_test_templates.py")
    print("2. Focus on critical modules first")
    print("3. Use: pytest --cov-report=html for detailed HTML report")
    print("4. Check: htmlcov/index.html in your browser")
