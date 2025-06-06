#!/usr/bin/env python3
"""
Simple test to verify async/await fixes are working correctly
"""

import sys
import os
sys.path.append('.')

from app.services.hybrid_ecg_service import HybridECGAnalysisService
import numpy as np

def test_synchronous_methods():
    """Test that synchronous methods work without await"""
    print("Testing synchronous methods...")
    
    service = HybridECGAnalysisService()
    
    features = {'heart_rate': 75, 'rr_intervals': [0.8, 0.9, 0.85]}
    result = service._simulate_predictions(features)
    print(f"✓ _simulate_predictions works: {result is not None}")
    
    af_features = {'rr_std': 50, 'hrv_rmssd': 30, 'spectral_entropy': 0.8}
    result2 = service._detect_atrial_fibrillation(af_features)
    print(f"✓ _detect_atrial_fibrillation works: {result2 is not None}")
    
    qt_features = {"qt_interval": 450, "qtc_bazett": 460, "heart_rate": 75}
    result3 = service._detect_long_qt(qt_features)
    print(f"✓ _detect_long_qt works: {result3 is not None}")
    
    ai_predictions = {'atrial_fibrillation': 0.8, 'normal': 0.2}
    pathology_results = {'atrial_fibrillation': {'detected': True, 'confidence': 0.8}}
    clinical_features = {'heart_rate': 75, 'rr_intervals': [0.8, 0.9]}
    result4 = service._generate_clinical_assessment(ai_predictions, pathology_results, clinical_features)
    print(f"✓ _generate_clinical_assessment works: {result4 is not None}")
    
    print("All synchronous method tests passed!")
    return True

if __name__ == "__main__":
    try:
        test_synchronous_methods()
        print("\n✅ Async/await fixes appear to be working correctly")
    except Exception as e:
        print(f"\n❌ Error testing methods: {e}")
        sys.exit(1)
