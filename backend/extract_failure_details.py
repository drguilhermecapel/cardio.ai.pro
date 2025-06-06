import re

def extract_test_failure_details():
    """Extract specific failure details from the large CI log file"""
    
    log_file_path = "/home/ubuntu/full_outputs/pytest_tests_test_fi_1749253219.txt"
    
    try:
        with open(log_file_path, 'r') as f:
            content = f.read()
        
        print("🔍 Extracting failure details for test_final_80_coverage_push_comprehensive.py")
        
        failures_match = re.search(r'==+ FAILURES ==+(.*?)(?:==+|$)', content, re.DOTALL)
        if failures_match:
            failures_section = failures_match.group(1)
            
            our_test_pattern = r'_+ (.*test_final_80_coverage_push_comprehensive.*?) _+(.*?)(?=_+ |$)'
            our_failures = re.findall(our_test_pattern, failures_section, re.DOTALL)
            
            if our_failures:
                print(f"✓ Found {len(our_failures)} failure(s) in our test file:")
                for i, (test_name, failure_details) in enumerate(our_failures, 1):
                    print(f"\n--- Failure {i}: {test_name} ---")
                    print(failure_details[:1000])  # First 1000 chars
                    if len(failure_details) > 1000:
                        print("... (truncated)")
            else:
                print("⚠️  No specific failures found for our test file")
                
                our_file_mentions = re.findall(r'.*test_final_80_coverage_push_comprehensive.*', content)
                if our_file_mentions:
                    print(f"✓ Found {len(our_file_mentions)} mentions of our test file:")
                    for mention in our_file_mentions[:5]:  # First 5 mentions
                        print(f"  - {mention.strip()}")
        else:
            print("⚠️  No FAILURES section found in log")
            
        errors_match = re.search(r'==+ ERRORS ==+(.*?)(?:==+|$)', content, re.DOTALL)
        if errors_match:
            errors_section = errors_match.group(1)
            
            our_error_pattern = r'_+ (.*test_final_80_coverage_push_comprehensive.*?) _+(.*?)(?=_+ |$)'
            our_errors = re.findall(our_error_pattern, errors_section, re.DOTALL)
            
            if our_errors:
                print(f"\n🔍 Found {len(our_errors)} error(s) in our test file:")
                for i, (test_name, error_details) in enumerate(our_errors, 1):
                    print(f"\n--- Error {i}: {test_name} ---")
                    print(error_details[:1000])  # First 1000 chars
                    if len(error_details) > 1000:
                        print("... (truncated)")
        
        summary_match = re.search(r'=+ short test summary info =+(.*?)(?:=+|$)', content, re.DOTALL)
        if summary_match:
            summary_section = summary_match.group(1)
            our_summary_lines = [line for line in summary_section.split('\n') 
                               if 'test_final_80_coverage_push_comprehensive' in line]
            if our_summary_lines:
                print(f"\n📋 Short test summary for our file:")
                for line in our_summary_lines:
                    print(f"  {line.strip()}")
                    
    except FileNotFoundError:
        print(f"❌ Log file not found: {log_file_path}")
    except Exception as e:
        print(f"❌ Error reading log file: {e}")

if __name__ == "__main__":
    extract_test_failure_details()
