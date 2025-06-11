import sys
import traceback
from unittest.mock import Mock, patch, MagicMock

mock_modules = {
    'pydantic': MagicMock(),
    'torch': MagicMock(),
    'sklearn': MagicMock(),
    'scipy': MagicMock(),
    'celery': MagicMock(),
    'redis': MagicMock(),
    'biosppy': MagicMock(),
    'wfdb': MagicMock(),
    'pyedflib': MagicMock(),
    'pywt': MagicMock(),
    'pandas': MagicMock(),
    'fastapi': MagicMock(),
    'sqlalchemy': MagicMock(),
    'numpy': MagicMock(),
    'neurokit2': MagicMock(),
    'matplotlib': MagicMock(),
    'joblib': MagicMock(),
    'pickle': MagicMock(),
}

for module_name, mock_module in mock_modules.items():
    sys.modules[module_name] = mock_module

def debug_hybrid_ecg_service_test():
    print("🔍 Debugging TestHybridECGServiceComprehensive.test_hybrid_ecg_service_all_methods")
    
    try:
        from tests.test_final_80_coverage_push_comprehensive import TestHybridECGServiceComprehensive
        print("✓ Test class imported successfully")
        
        test_instance = TestHybridECGServiceComprehensive()
        print("✓ Test instance created successfully")
        
        test_instance.test_hybrid_ecg_service_all_methods()
        print("✓ Test method executed successfully")
        
    except Exception as e:
        print(f"✗ Error in test execution: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        traceback.print_exc()
        
        try:
            print("\n🔍 Attempting direct service import...")
            from app.services.hybrid_ecg_service import HybridECGAnalysisService
            print("✓ Service imports successfully")
            
            with patch('app.services.hybrid_ecg_service.MLModelService', Mock()):
                with patch('app.services.hybrid_ecg_service.ECGProcessor', Mock()):
                    with patch('app.services.hybrid_ecg_service.ValidationService', Mock()):
                        service = HybridECGAnalysisService()
                        print("✓ Service instantiates successfully")
                        
        except Exception as import_error:
            print(f"✗ Service import/instantiation error: {str(import_error)}")
            traceback.print_exc()

if __name__ == "__main__":
    debug_hybrid_ecg_service_test()
