"""
Verify that circular import fixes work
"""

def test_imports():
    try:
        from app.utils.ecg_hybrid_processor import ECGHybridProcessor
        print("✅ ECGHybridProcessor import successful")
        
        from app.services.hybrid_ecg_service import HybridECGAnalysisService
        print("✅ HybridECGAnalysisService import successful")
        
        from app.services.ml_model_service import MLModelService
        print("✅ MLModelService import successful")
        
        processor = ECGHybridProcessor()
        print(f"✅ ECGHybridProcessor instantiation successful - sampling_rate: {processor.sampling_rate}")
        
        service = HybridECGAnalysisService()
        print(f"✅ HybridECGAnalysisService instantiation successful - fs: {service.fs}")
        
        ml_service = MLModelService()
        print(f"✅ MLModelService instantiation successful - models: {len(ml_service.models)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Import/instantiation failed: {e}")
        return False

if __name__ == "__main__":
    success = test_imports()
    if success:
        print("\n🎉 All circular import fixes successful!")
    else:
        print("\n💥 Circular import issues still exist")
