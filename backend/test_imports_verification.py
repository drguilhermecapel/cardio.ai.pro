#!/usr/bin/env python3
"""Verify that the target modules can be imported correctly."""

import sys
import os

sys.path.insert(0, '/home/ubuntu/cardio.ai.pro/backend')

def test_imports():
    """Test importing the target modules."""
    try:
        print("Testing imports...")
        
        from app.utils.ecg_hybrid_processor import ECGHybridProcessor
        print("✅ ECGHybridProcessor imported successfully")
        
        processor = ECGHybridProcessor()
        print(f"✅ ECGHybridProcessor instantiated: {type(processor)}")
        
        from app.services.hybrid_ecg_service import (
            ClinicalUrgency, 
            UniversalECGReader, 
            AdvancedPreprocessor,
            FeatureExtractor,
            HybridECGAnalysisService
        )
        print("✅ All hybrid_ecg_service classes imported successfully")
        
        urgency = ClinicalUrgency.LOW
        reader = UniversalECGReader()
        preprocessor = AdvancedPreprocessor()
        extractor = FeatureExtractor()
        service = HybridECGAnalysisService()
        
        print(f"✅ All classes instantiated successfully")
        print(f"   - ClinicalUrgency.LOW: {urgency}")
        print(f"   - UniversalECGReader: {type(reader)}")
        print(f"   - AdvancedPreprocessor: {type(preprocessor)}")
        print(f"   - FeatureExtractor: {type(extractor)}")
        print(f"   - HybridECGAnalysisService: {type(service)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_imports()
    if success:
        print("\n🎉 All imports successful! Tests can proceed.")
    else:
        print("\n💥 Import failures detected. Need to fix imports first.")
    sys.exit(0 if success else 1)
