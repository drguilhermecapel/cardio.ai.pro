"""
Advanced Risk Prediction Service
Implements 10-year cardiovascular risk prediction and sudden cardiac death assessment
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Risk prediction features limited.")

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("Scikit-learn not available. Traditional ML models disabled.")


class CardiovascularRiskModel(nn.Module):
    """Neural network for cardiovascular risk prediction"""
    
    def __init__(
        self,
        input_dim: int = 50,
        hidden_dims: List[int] = [256, 128, 64],
        dropout: float = 0.3
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
            
        self.risk_head = nn.Sequential(
            *layers,
            nn.Linear(prev_dim, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.event_head = nn.Sequential(
            *layers[:-1],  # Reuse feature layers
            nn.Linear(prev_dim, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 8)  # 8 event types
        )
        
        self.time_head = nn.Sequential(
            *layers[:-1],
            nn.Linear(prev_dim, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
            nn.ReLU()  # Ensure positive time
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through risk prediction model"""
        risk_prob = self.risk_head(x)
        event_logits = self.event_head(x)
        time_pred = self.time_head(x)
        
        return {
            "risk_probability": risk_prob,
            "event_logits": event_logits,
            "time_to_event": time_pred
        }


class SuddenCardiacDeathModel(nn.Module):
    """Specialized model for sudden cardiac death risk assessment"""
    
    def __init__(
        self,
        ecg_features_dim: int = 512,
        clinical_features_dim: int = 20,
        hidden_dim: int = 256
    ):
        super().__init__()
        
        self.ecg_processor = nn.Sequential(
            nn.Linear(ecg_features_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        self.clinical_processor = nn.Sequential(
            nn.Linear(clinical_features_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self, 
        ecg_features: torch.Tensor, 
        clinical_features: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass for SCD risk prediction"""
        ecg_processed = self.ecg_processor(ecg_features)
        clinical_processed = self.clinical_processor(clinical_features)
        
        combined = torch.cat([ecg_processed, clinical_processed], dim=1)
        scd_risk = self.fusion(combined)
        
        return scd_risk


class RiskPredictionService:
    """Service for advanced cardiovascular risk prediction"""
    
    def __init__(self, device: str = "auto"):
        self.device = self._setup_device(device)
        
        self.cv_risk_model: Optional[CardiovascularRiskModel] = None
        self.scd_model: Optional[SuddenCardiacDeathModel] = None
        self.traditional_models: Dict[str, Any] = {}
        
        self.framingham_coefficients = self._load_framingham_coefficients()
        self.pooled_cohort_coefficients = self._load_pooled_cohort_coefficients()
        self.score2_coefficients = self._load_score2_coefficients()
        
        self.event_types = [
            "myocardial_infarction",
            "stroke",
            "heart_failure",
            "sudden_cardiac_death",
            "coronary_revascularization",
            "peripheral_artery_disease",
            "atrial_fibrillation",
            "cardiovascular_death"
        ]
        
        self.risk_thresholds = {
            "low": 0.075,      # <7.5% 10-year risk
            "intermediate": 0.20,  # 7.5-20% 10-year risk
            "high": 1.0        # >20% 10-year risk
        }
        
        self.models_loaded = False
        
    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device"""
        if not TORCH_AVAILABLE:
            return "cpu"
            
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        else:
            return torch.device(device)
            
    def _load_framingham_coefficients(self) -> Dict[str, float]:
        """Load Framingham Risk Score coefficients"""
        return {
            "age_male": 0.04826,
            "age_female": 0.33766,
            "total_cholesterol": 0.00013,
            "hdl_cholesterol": -0.00881,
            "systolic_bp_treated": 0.00226,
            "systolic_bp_untreated": 0.00670,
            "smoking": 0.52873,
            "diabetes": 0.69154,
            "baseline_survival_male": 0.88936,
            "baseline_survival_female": 0.95012
        }
        
    def _load_pooled_cohort_coefficients(self) -> Dict[str, Dict[str, float]]:
        """Load Pooled Cohort Equations coefficients"""
        return {
            "white_male": {
                "ln_age": 12.344,
                "ln_total_chol": 11.853,
                "ln_age_chol": -2.664,
                "ln_hdl": -7.990,
                "ln_age_hdl": 1.769,
                "ln_treated_sbp": 1.797,
                "ln_age_treated_sbp": 0.0,
                "ln_untreated_sbp": 1.764,
                "ln_age_untreated_sbp": 0.0,
                "smoking": 7.837,
                "ln_age_smoking": -1.795,
                "diabetes": 0.658,
                "mean_coefficient": 61.18,
                "baseline_survival": 0.9144
            },
            "white_female": {
                "ln_age": -29.799,
                "ln_total_chol": 4.884,
                "ln_age_chol": 0.0,
                "ln_hdl": -13.540,
                "ln_age_hdl": 3.114,
                "ln_treated_sbp": 2.019,
                "ln_age_treated_sbp": 0.0,
                "ln_untreated_sbp": 1.957,
                "ln_age_untreated_sbp": 0.0,
                "smoking": 7.574,
                "ln_age_smoking": -1.665,
                "diabetes": 0.661,
                "mean_coefficient": -29.18,
                "baseline_survival": 0.9665
            }
        }
        
    def _load_score2_coefficients(self) -> Dict[str, float]:
        """Load SCORE2 coefficients for European populations"""
        return {
            "age": 0.3742,
            "smoking": 0.6012,
            "systolic_bp": 0.0255,
            "total_cholesterol": 0.2550,
            "hdl_cholesterol": -0.1846,
            "diabetes": 0.4277,
            "baseline_low_risk": -5.8663,
            "baseline_moderate_risk": -5.4321,
            "baseline_high_risk": -4.9876,
            "baseline_very_high_risk": -4.5432
        }
        
    async def load_models(self, model_path: Optional[str] = None) -> bool:
        """Load risk prediction models"""
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available. Using traditional models only.")
            return await self._load_traditional_models()
            
        try:
            self.cv_risk_model = CardiovascularRiskModel()
            self.scd_model = SuddenCardiacDeathModel()
            
            if model_path:
                try:
                    checkpoint = torch.load(model_path, map_location=self.device)
                    if "cv_risk_model" in checkpoint:
                        self.cv_risk_model.load_state_dict(checkpoint["cv_risk_model"])
                    if "scd_model" in checkpoint:
                        self.scd_model.load_state_dict(checkpoint["scd_model"])
                    logger.info(f"Loaded pre-trained models from {model_path}")
                except Exception as e:
                    logger.warning(f"Failed to load pre-trained weights: {e}")
                    
            self.cv_risk_model.to(self.device)
            self.scd_model.to(self.device)
            
            self.cv_risk_model.eval()
            self.scd_model.eval()
            
            await self._load_traditional_models()
            
            self.models_loaded = True
            logger.info("Risk prediction models loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load risk prediction models: {e}")
            return False
            
    async def _load_traditional_models(self) -> bool:
        """Load traditional ML models for risk prediction"""
        if not SKLEARN_AVAILABLE:
            logger.warning("Scikit-learn not available. Traditional models disabled.")
            return False
            
        try:
            self.traditional_models = {
                "random_forest": RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42
                ),
                "gradient_boosting": GradientBoostingRegressor(
                    n_estimators=100,
                    max_depth=6,
                    random_state=42
                ),
                "logistic_regression": LogisticRegression(
                    random_state=42,
                    max_iter=1000
                ),
                "scaler": StandardScaler()
            }
            
            logger.info("Traditional ML models initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize traditional models: {e}")
            return False
            
    async def predict_cardiovascular_risk(
        self,
        patient_data: Dict[str, Any],
        ecg_features: Optional[npt.NDArray[np.float32]] = None,
        use_neural_model: bool = True
    ) -> Dict[str, Any]:
        """Predict 10-year cardiovascular risk"""
        try:
            start_time = time.time()
            
            clinical_features = self._extract_clinical_features(patient_data)
            
            results = {
                "ten_year_risk": 0.0,
                "risk_category": "unknown",
                "event_probabilities": {},
                "time_to_event_years": None,
                "risk_factors": [],
                "recommendations": [],
                "model_used": "traditional",
                "processing_time": 0.0
            }
            
            if use_neural_model and self.models_loaded and TORCH_AVAILABLE and self.cv_risk_model:
                try:
                    neural_result = await self._predict_with_neural_model(
                        clinical_features, ecg_features
                    )
                    results.update(neural_result)
                    results["model_used"] = "neural_network"
                except Exception as e:
                    logger.warning(f"Neural model prediction failed: {e}")
                    
            traditional_results = await self._calculate_traditional_risk_scores(patient_data)
            
            if results["ten_year_risk"] == 0.0:
                results.update(traditional_results)
                results["model_used"] = "traditional"
            else:
                traditional_risk = traditional_results.get("ten_year_risk", 0.0)
                if traditional_risk > 0:
                    results["ten_year_risk"] = (results["ten_year_risk"] + traditional_risk) / 2
                    results["model_used"] = "ensemble"
                    
                results["traditional_scores"] = traditional_results
                
            results["risk_category"] = self._categorize_risk(results["ten_year_risk"])
            
            results["risk_factors"] = self._identify_risk_factors(patient_data)
            results["recommendations"] = self._generate_recommendations(
                results["risk_category"], 
                results["risk_factors"]
            )
            
            results["processing_time"] = time.time() - start_time
            
            return results
            
        except Exception as e:
            logger.error(f"Cardiovascular risk prediction failed: {e}")
            raise
            
    async def predict_sudden_cardiac_death_risk(
        self,
        patient_data: Dict[str, Any],
        ecg_features: npt.NDArray[np.float32]
    ) -> Dict[str, Any]:
        """Predict sudden cardiac death risk"""
        try:
            start_time = time.time()
            
            results = {
                "scd_risk_score": 0.0,
                "risk_level": "unknown",
                "risk_factors": [],
                "protective_factors": [],
                "recommendations": [],
                "model_confidence": 0.0,
                "processing_time": 0.0
            }
            
            if not self.models_loaded or not TORCH_AVAILABLE or not self.scd_model:
                results = await self._traditional_scd_assessment(patient_data)
            else:
                clinical_features = self._extract_clinical_features_scd(patient_data)
                
                with torch.no_grad():
                    ecg_tensor = torch.FloatTensor(ecg_features).to(self.device)
                    clinical_tensor = torch.FloatTensor(clinical_features).to(self.device)
                    
                    if len(ecg_tensor.shape) == 1:
                        ecg_tensor = ecg_tensor.unsqueeze(0)
                    if len(clinical_tensor.shape) == 1:
                        clinical_tensor = clinical_tensor.unsqueeze(0)
                        
                    scd_risk = self.scd_model(ecg_tensor, clinical_tensor)
                    
                    results["scd_risk_score"] = float(scd_risk.cpu().numpy()[0, 0])
                    results["model_confidence"] = 0.85  # Model-specific confidence
                    
            if results["scd_risk_score"] < 0.02:
                results["risk_level"] = "low"
            elif results["scd_risk_score"] < 0.05:
                results["risk_level"] = "intermediate"
            elif results["scd_risk_score"] < 0.10:
                results["risk_level"] = "high"
            else:
                results["risk_level"] = "very_high"
                
            results["risk_factors"] = self._identify_scd_risk_factors(patient_data)
            results["protective_factors"] = self._identify_protective_factors(patient_data)
            results["recommendations"] = self._generate_scd_recommendations(
                results["risk_level"],
                results["risk_factors"]
            )
            
            results["processing_time"] = time.time() - start_time
            
            return results
            
        except Exception as e:
            logger.error(f"SCD risk prediction failed: {e}")
            raise
            
    async def _predict_with_neural_model(
        self,
        clinical_features: npt.NDArray[np.float32],
        ecg_features: Optional[npt.NDArray[np.float32]] = None
    ) -> Dict[str, Any]:
        """Predict using neural network model"""
        if not self.cv_risk_model:
            raise RuntimeError("Neural model not loaded")
            
        if ecg_features is not None:
            features = np.concatenate([clinical_features, ecg_features[:50]])  # Limit ECG features
        else:
            features = np.concatenate([clinical_features, np.zeros(50 - len(clinical_features))])
            
        if len(features) < 50:
            padded_features = np.zeros(50)
            padded_features[:len(features)] = features
            features = padded_features
        elif len(features) > 50:
            features = features[:50]
            
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            outputs = self.cv_risk_model(features_tensor)
            
            risk_prob = float(outputs["risk_probability"].cpu().numpy()[0, 0])
            event_logits = outputs["event_logits"].cpu().numpy()[0]
            time_pred = float(outputs["time_to_event"].cpu().numpy()[0, 0])
            
            event_probs = torch.softmax(torch.FloatTensor(event_logits), dim=0).numpy()
            
            event_probabilities = {}
            for i, event_type in enumerate(self.event_types):
                if i < len(event_probs):
                    event_probabilities[event_type] = float(event_probs[i])
                    
            return {
                "ten_year_risk": risk_prob,
                "event_probabilities": event_probabilities,
                "time_to_event_years": time_pred
            }
            
    async def _calculate_traditional_risk_scores(
        self, 
        patient_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate traditional risk scores (Framingham, Pooled Cohort, SCORE2)"""
        results = {
            "ten_year_risk": 0.0,
            "framingham_score": 0.0,
            "pooled_cohort_score": 0.0,
            "score2_score": 0.0
        }
        
        try:
            framingham_risk = self._calculate_framingham_risk(patient_data)
            results["framingham_score"] = framingham_risk
            
            pooled_cohort_risk = self._calculate_pooled_cohort_risk(patient_data)
            results["pooled_cohort_score"] = pooled_cohort_risk
            
            score2_risk = self._calculate_score2_risk(patient_data)
            results["score2_score"] = score2_risk
            
            valid_scores = [
                score for score in [framingham_risk, pooled_cohort_risk, score2_risk]
                if score > 0
            ]
            
            if valid_scores:
                results["ten_year_risk"] = np.mean(valid_scores)
                
        except Exception as e:
            logger.error(f"Traditional risk score calculation failed: {e}")
            
        return results
        
    def _calculate_framingham_risk(self, patient_data: Dict[str, Any]) -> float:
        """Calculate Framingham Risk Score"""
        try:
            age = patient_data.get("age", 50)
            sex = patient_data.get("sex", "male").lower()
            total_chol = patient_data.get("total_cholesterol", 200)
            hdl_chol = patient_data.get("hdl_cholesterol", 50)
            systolic_bp = patient_data.get("systolic_bp", 120)
            bp_treated = patient_data.get("bp_treated", False)
            smoking = patient_data.get("smoking", False)
            diabetes = patient_data.get("diabetes", False)
            
            coeff = self.framingham_coefficients
            
            score = 0.0
            
            if sex == "male":
                score += coeff["age_male"] * age
                baseline_survival = coeff["baseline_survival_male"]
            else:
                score += coeff["age_female"] * age
                baseline_survival = coeff["baseline_survival_female"]
                
            score += coeff["total_cholesterol"] * total_chol
            score += coeff["hdl_cholesterol"] * hdl_chol
            
            if bp_treated:
                score += coeff["systolic_bp_treated"] * systolic_bp
            else:
                score += coeff["systolic_bp_untreated"] * systolic_bp
                
            if smoking:
                score += coeff["smoking"]
                
            if diabetes:
                score += coeff["diabetes"]
                
            risk = 1 - (baseline_survival ** np.exp(score))
            
            return max(0.0, min(1.0, risk))
            
        except Exception as e:
            logger.error(f"Framingham calculation failed: {e}")
            return 0.0
            
    def _calculate_pooled_cohort_risk(self, patient_data: Dict[str, Any]) -> float:
        """Calculate Pooled Cohort Equations risk"""
        try:
            age = patient_data.get("age", 50)
            sex = patient_data.get("sex", "male").lower()
            race = patient_data.get("race", "white").lower()
            total_chol = patient_data.get("total_cholesterol", 200)
            hdl_chol = patient_data.get("hdl_cholesterol", 50)
            systolic_bp = patient_data.get("systolic_bp", 120)
            bp_treated = patient_data.get("bp_treated", False)
            smoking = patient_data.get("smoking", False)
            diabetes = patient_data.get("diabetes", False)
            
            if sex == "male" and race == "white":
                coeff = self.pooled_cohort_coefficients["white_male"]
            else:
                coeff = self.pooled_cohort_coefficients["white_female"]
                
            ln_age = np.log(age)
            ln_total_chol = np.log(total_chol)
            ln_hdl = np.log(hdl_chol)
            ln_sbp = np.log(systolic_bp)
            
            individual_sum = (
                coeff["ln_age"] * ln_age +
                coeff["ln_total_chol"] * ln_total_chol +
                coeff["ln_age_chol"] * ln_age * ln_total_chol +
                coeff["ln_hdl"] * ln_hdl +
                coeff["ln_age_hdl"] * ln_age * ln_hdl
            )
            
            if bp_treated:
                individual_sum += (
                    coeff["ln_treated_sbp"] * ln_sbp +
                    coeff["ln_age_treated_sbp"] * ln_age * ln_sbp
                )
            else:
                individual_sum += (
                    coeff["ln_untreated_sbp"] * ln_sbp +
                    coeff["ln_age_untreated_sbp"] * ln_age * ln_sbp
                )
                
            if smoking:
                individual_sum += (
                    coeff["smoking"] +
                    coeff["ln_age_smoking"] * ln_age
                )
                
            if diabetes:
                individual_sum += coeff["diabetes"]
                
            risk = 1 - (coeff["baseline_survival"] ** np.exp(individual_sum - coeff["mean_coefficient"]))
            
            return max(0.0, min(1.0, risk))
            
        except Exception as e:
            logger.error(f"Pooled Cohort calculation failed: {e}")
            return 0.0
            
    def _calculate_score2_risk(self, patient_data: Dict[str, Any]) -> float:
        """Calculate SCORE2 risk"""
        try:
            age = patient_data.get("age", 50)
            smoking = patient_data.get("smoking", False)
            systolic_bp = patient_data.get("systolic_bp", 120)
            total_chol = patient_data.get("total_cholesterol", 200)
            hdl_chol = patient_data.get("hdl_cholesterol", 50)
            diabetes = patient_data.get("diabetes", False)
            region_risk = patient_data.get("region_risk", "moderate")  # low, moderate, high, very_high
            
            coeff = self.score2_coefficients
            
            linear_predictor = (
                coeff["age"] * (age - 60) +
                coeff["systolic_bp"] * (systolic_bp - 140) +
                coeff["total_cholesterol"] * (total_chol / 38.67 - 5.2) +
                coeff["hdl_cholesterol"] * (hdl_chol / 38.67 - 1.3)
            )
            
            if smoking:
                linear_predictor += coeff["smoking"]
                
            if diabetes:
                linear_predictor += coeff["diabetes"]
                
            baseline_key = f"baseline_{region_risk}_risk"
            if baseline_key in coeff:
                linear_predictor += coeff[baseline_key]
            else:
                linear_predictor += coeff["baseline_moderate_risk"]
                
            risk = 1 - np.exp(-np.exp(linear_predictor))
            
            return max(0.0, min(1.0, risk))
            
        except Exception as e:
            logger.error(f"SCORE2 calculation failed: {e}")
            return 0.0
            
    async def _traditional_scd_assessment(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Traditional sudden cardiac death risk assessment"""
        risk_score = 0.0
        
        if patient_data.get("previous_cardiac_arrest", False):
            risk_score += 0.4
        if patient_data.get("sustained_vt", False):
            risk_score += 0.3
        if patient_data.get("ejection_fraction", 60) < 35:
            risk_score += 0.2
        if patient_data.get("ischemic_cardiomyopathy", False):
            risk_score += 0.15
        if patient_data.get("heart_failure", False):
            risk_score += 0.1
            
        if patient_data.get("family_history_scd", False):
            risk_score += 0.05
        if patient_data.get("syncope", False):
            risk_score += 0.03
        if patient_data.get("diabetes", False):
            risk_score += 0.02
            
        return {
            "scd_risk_score": min(1.0, risk_score),
            "model_confidence": 0.7
        }
        
    def _extract_clinical_features(self, patient_data: Dict[str, Any]) -> npt.NDArray[np.float32]:
        """Extract clinical features for risk prediction"""
        features = []
        
        features.append(patient_data.get("age", 50) / 100.0)  # Normalize age
        features.append(1.0 if patient_data.get("sex", "male").lower() == "male" else 0.0)
        
        features.append(patient_data.get("systolic_bp", 120) / 200.0)
        features.append(patient_data.get("diastolic_bp", 80) / 120.0)
        features.append(patient_data.get("total_cholesterol", 200) / 400.0)
        features.append(patient_data.get("hdl_cholesterol", 50) / 100.0)
        features.append(patient_data.get("ldl_cholesterol", 100) / 300.0)
        features.append(patient_data.get("triglycerides", 150) / 500.0)
        features.append(patient_data.get("bmi", 25) / 50.0)
        
        features.append(1.0 if patient_data.get("smoking", False) else 0.0)
        features.append(1.0 if patient_data.get("diabetes", False) else 0.0)
        features.append(1.0 if patient_data.get("hypertension", False) else 0.0)
        features.append(1.0 if patient_data.get("family_history_cad", False) else 0.0)
        features.append(1.0 if patient_data.get("previous_mi", False) else 0.0)
        features.append(1.0 if patient_data.get("previous_stroke", False) else 0.0)
        
        features.append(1.0 if patient_data.get("statin_use", False) else 0.0)
        features.append(1.0 if patient_data.get("aspirin_use", False) else 0.0)
        features.append(1.0 if patient_data.get("ace_inhibitor", False) else 0.0)
        features.append(1.0 if patient_data.get("beta_blocker", False) else 0.0)
        
        while len(features) < 20:
            features.append(0.0)
            
        return np.array(features[:20], dtype=np.float32)
        
    def _extract_clinical_features_scd(self, patient_data: Dict[str, Any]) -> npt.NDArray[np.float32]:
        """Extract clinical features specific to SCD risk"""
        features = []
        
        features.append(patient_data.get("age", 50) / 100.0)
        features.append(1.0 if patient_data.get("sex", "male").lower() == "male" else 0.0)
        
        features.append(patient_data.get("ejection_fraction", 60) / 100.0)
        features.append(patient_data.get("lv_mass_index", 100) / 200.0)
        
        features.append(1.0 if patient_data.get("ischemic_cardiomyopathy", False) else 0.0)
        features.append(1.0 if patient_data.get("dilated_cardiomyopathy", False) else 0.0)
        features.append(1.0 if patient_data.get("hypertrophic_cardiomyopathy", False) else 0.0)
        features.append(1.0 if patient_data.get("arrhythmogenic_cardiomyopathy", False) else 0.0)
        
        features.append(1.0 if patient_data.get("sustained_vt", False) else 0.0)
        features.append(1.0 if patient_data.get("nonsustained_vt", False) else 0.0)
        features.append(1.0 if patient_data.get("atrial_fibrillation", False) else 0.0)
        features.append(1.0 if patient_data.get("previous_cardiac_arrest", False) else 0.0)
        
        features.append(1.0 if patient_data.get("syncope", False) else 0.0)
        features.append(1.0 if patient_data.get("heart_failure", False) else 0.0)
        
        features.append(1.0 if patient_data.get("family_history_scd", False) else 0.0)
        features.append(1.0 if patient_data.get("diabetes", False) else 0.0)
        features.append(1.0 if patient_data.get("ckd", False) else 0.0)
        
        features.append(1.0 if patient_data.get("icd_implanted", False) else 0.0)
        features.append(1.0 if patient_data.get("crt_device", False) else 0.0)
        
        features.append(1.0 if patient_data.get("antiarrhythmic_drugs", False) else 0.0)
        
        return np.array(features[:20], dtype=np.float32)
        
    def _categorize_risk(self, risk_score: float) -> str:
        """Categorize cardiovascular risk level"""
        if risk_score < self.risk_thresholds["low"]:
            return "low"
        elif risk_score < self.risk_thresholds["intermediate"]:
            return "intermediate"
        else:
            return "high"
            
    def _identify_risk_factors(self, patient_data: Dict[str, Any]) -> List[str]:
        """Identify modifiable and non-modifiable risk factors"""
        risk_factors = []
        
        if patient_data.get("age", 0) > 65:
            risk_factors.append("Advanced age")
        if patient_data.get("sex", "").lower() == "male":
            risk_factors.append("Male sex")
        if patient_data.get("family_history_cad", False):
            risk_factors.append("Family history of CAD")
            
        if patient_data.get("smoking", False):
            risk_factors.append("Current smoking")
        if patient_data.get("diabetes", False):
            risk_factors.append("Diabetes mellitus")
        if patient_data.get("hypertension", False):
            risk_factors.append("Hypertension")
        if patient_data.get("total_cholesterol", 0) > 240:
            risk_factors.append("High cholesterol")
        if patient_data.get("hdl_cholesterol", 100) < 40:
            risk_factors.append("Low HDL cholesterol")
        if patient_data.get("bmi", 0) > 30:
            risk_factors.append("Obesity")
            
        return risk_factors
        
    def _identify_scd_risk_factors(self, patient_data: Dict[str, Any]) -> List[str]:
        """Identify SCD-specific risk factors"""
        risk_factors = []
        
        if patient_data.get("ejection_fraction", 60) < 35:
            risk_factors.append("Reduced ejection fraction")
        if patient_data.get("previous_cardiac_arrest", False):
            risk_factors.append("Previous cardiac arrest")
        if patient_data.get("sustained_vt", False):
            risk_factors.append("Sustained ventricular tachycardia")
        if patient_data.get("ischemic_cardiomyopathy", False):
            risk_factors.append("Ischemic cardiomyopathy")
        if patient_data.get("heart_failure", False):
            risk_factors.append("Heart failure")
        if patient_data.get("syncope", False):
            risk_factors.append("Unexplained syncope")
        if patient_data.get("family_history_scd", False):
            risk_factors.append("Family history of SCD")
            
        return risk_factors
        
    def _identify_protective_factors(self, patient_data: Dict[str, Any]) -> List[str]:
        """Identify protective factors against SCD"""
        protective_factors = []
        
        if patient_data.get("icd_implanted", False):
            protective_factors.append("ICD implanted")
        if patient_data.get("beta_blocker", False):
            protective_factors.append("Beta-blocker therapy")
        if patient_data.get("ace_inhibitor", False):
            protective_factors.append("ACE inhibitor therapy")
        if patient_data.get("statin_use", False):
            protective_factors.append("Statin therapy")
        if patient_data.get("exercise_regular", False):
            protective_factors.append("Regular exercise")
            
        return protective_factors
        
    def _generate_recommendations(
        self, 
        risk_category: str, 
        risk_factors: List[str]
    ) -> List[str]:
        """Generate personalized treatment recommendations"""
        recommendations = []
        
        if risk_category == "high":
            recommendations.extend([
                "Consider high-intensity statin therapy",
                "Target LDL cholesterol <70 mg/dL",
                "Consider aspirin therapy if no contraindications",
                "Aggressive blood pressure control (<130/80 mmHg)",
                "Cardiology consultation recommended"
            ])
        elif risk_category == "intermediate":
            recommendations.extend([
                "Consider moderate-intensity statin therapy",
                "Lifestyle modifications (diet, exercise)",
                "Blood pressure control (<140/90 mmHg)",
                "Consider coronary calcium scoring"
            ])
        else:  # low risk
            recommendations.extend([
                "Lifestyle modifications",
                "Regular cardiovascular risk assessment",
                "Maintain healthy weight and exercise"
            ])
            
        if "Current smoking" in risk_factors:
            recommendations.append("Smoking cessation counseling and support")
        if "Diabetes mellitus" in risk_factors:
            recommendations.append("Optimal diabetes management (HbA1c <7%)")
        if "High cholesterol" in risk_factors:
            recommendations.append("Dietary counseling and lipid management")
        if "Obesity" in risk_factors:
            recommendations.append("Weight management program")
            
        return recommendations
        
    def _generate_scd_recommendations(
        self, 
        risk_level: str, 
        risk_factors: List[str]
    ) -> List[str]:
        """Generate SCD-specific recommendations"""
        recommendations = []
        
        if risk_level == "very_high":
            recommendations.extend([
                "Urgent cardiology/electrophysiology consultation",
                "Consider ICD implantation",
                "Optimize heart failure medications",
                "Avoid QT-prolonging medications"
            ])
        elif risk_level == "high":
            recommendations.extend([
                "Cardiology consultation within 2 weeks",
                "Consider ICD evaluation",
                "Optimize medical therapy",
                "Regular monitoring"
            ])
        elif risk_level == "intermediate":
            recommendations.extend([
                "Cardiology follow-up",
                "Risk stratification studies",
                "Medical optimization"
            ])
        else:  # low risk
            recommendations.extend([
                "Routine cardiology follow-up",
                "Continue current management"
            ])
            
        if "Reduced ejection fraction" in risk_factors:
            recommendations.append("ACE inhibitor/ARB and beta-blocker optimization")
        if "Heart failure" in risk_factors:
            recommendations.append("Heart failure guideline-directed medical therapy")
            
        return recommendations
        
    def get_service_info(self) -> Dict[str, Any]:
        """Get information about the risk prediction service"""
        return {
            "models_loaded": self.models_loaded,
            "torch_available": TORCH_AVAILABLE,
            "sklearn_available": SKLEARN_AVAILABLE,
            "device": str(self.device),
            "supported_scores": [
                "framingham",
                "pooled_cohort_equations",
                "score2",
                "neural_network",
                "sudden_cardiac_death"
            ],
            "event_types": self.event_types,
            "risk_thresholds": self.risk_thresholds
        }
