from app.services.hybrid_ecg_service import HybridECGAnalysisService

def verify_methods():
    service = HybridECGAnalysisService()
    
    required_methods = [
        '_analyze_with_ai',
        '_analyze_emergency_patterns', 
        '_generate_audit_trail',
        '_preprocess_signal',
        '_validate_ecg_signal'
    ]
    
    print("=== METHOD VERIFICATION ===")
    for method in required_methods:
        exists = hasattr(service, method)
        print(f"{method}: {'✓' if exists else '✗'}")
    
    print(f"\nAll methods exist: {all(hasattr(service, m) for m in required_methods)}")

if __name__ == "__main__":
    verify_methods()
