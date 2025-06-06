import numpy as np
from app.services.hybrid_ecg_service import HybridECGAnalysisService
from unittest.mock import Mock

print("Debugging signal shapes in preprocessing...")

service = HybridECGAnalysisService(Mock(), Mock())

test_signals = [
    np.array([[0.1, 0.2, 0.3, 0.2, 0.1] * 250]),  # test_preprocessor_functionality
    np.array([[0.1, 0.2, 0.3, 0.2, 0.1] * 250]),  # test_data_integrity_validation  
    np.array([[0.1, 0.2, 0.3] * 500])              # test_preprocessing_performance
]

for i, test_signal in enumerate(test_signals):
    print(f"\nTest signal {i+1}:")
    print(f"  Original shape: {test_signal.shape}")
    print(f"  Original ndim: {test_signal.ndim}")
    
    if test_signal.ndim == 1:
        reshaped = test_signal.reshape(-1, 1)
        print(f"  After reshape: {reshaped.shape}")
    else:
        reshaped = test_signal
        print(f"  No reshape needed: {reshaped.shape}")
    
    for lead in range(reshaped.shape[1]):
        lead_signal = reshaped[:, lead]
        print(f"  Lead {lead} shape: {lead_signal.shape}")
        print(f"  Lead {lead} length: {len(lead_signal)}")
        
        if len(lead_signal) < 20:
            print(f"  ❌ Lead {lead} too short for filter!")
        else:
            print(f"  ✅ Lead {lead} should work with filter")
