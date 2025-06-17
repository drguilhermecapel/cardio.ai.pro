"""ML Model Service - Mock implementation for tests."""

from typing import Dict, Any, List
import numpy as np

class MLModelService:
    """Mock ML Model Service for tests."""
    
    def __init__(self):
        self.model = None
        
    async def predict(self, data: np.ndarray) -> Dict[str, Any]:
        """Mock prediction."""
        return {
            "prediction": "normal",
            "confidence": 0.95,
            "probabilities": {"normal": 0.95, "abnormal": 0.05}
        }
    
    async def batch_predict(self, data_list: List[np.ndarray]) -> List[Dict[str, Any]]:
        """Mock batch prediction."""
        return [await self.predict(data) for data in data_list]
    
    def load_model(self, model_path: str) -> None:
        """Mock model loading."""
        self.model = "mock_model"
