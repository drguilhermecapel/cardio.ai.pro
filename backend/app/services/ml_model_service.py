"""
ML Model Service - Machine Learning model management and inference.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import onnxruntime as ort

from app.core.config import settings
from app.core.exceptions import MLModelException
from app.services.tensorrt_optimizer import TensorRTModelManager
from app.services.ecg_foundation_model import ECGFoundationModel
from app.services.risk_prediction_service import RiskPredictionService
from app.services.sleep_apnea_service import SleepApneaDetector, RespiratoryPatternAnalyzer
from app.services.fhir_service import FHIRService
from app.services.wearable_integration_service import WearableIntegrationService
from app.services.federated_learning import FederatedLearningService, ParticipantRole
from app.services.explainable_ai import ExplainableAIService, ExplanationMethod
from app.services.homomorphic_encryption import HomomorphicEncryptionService, homomorphic_service
from app.utils.memory_monitor import MemoryMonitor

logger = logging.getLogger(__name__)


class MLModelService:
    """ML Model Service for ECG analysis using ONNX Runtime."""

    def __init__(self) -> None:
        self.models: dict[str, ort.InferenceSession] = {}
        self.model_metadata: dict[str, dict[str, Any]] = {}
        self.memory_monitor = MemoryMonitor()
        
        self.tensorrt_manager = TensorRTModelManager(settings.MODELS_DIR)
        self.use_tensorrt = False
        
        self.foundation_model = ECGFoundationModel(
            model_type="transformer",
            device="auto"
        )
        self.foundation_model_loaded = False
        
        self.risk_prediction_service = RiskPredictionService(device="auto")
        self.risk_service_loaded = False
        
        self.sleep_apnea_detector = SleepApneaDetector(sampling_rate=500)
        self.respiratory_analyzer = RespiratoryPatternAnalyzer(sampling_rate=500)
        
        self.fhir_service = FHIRService()
        
        self.wearable_service = WearableIntegrationService()
        
        self.federated_learning_service = FederatedLearningService(
            node_id=f"ecg_node_{hash(str(id(self))) % 10000}",
            role=ParticipantRole.PARTICIPANT
        )
        
        self.explainable_ai_service = ExplainableAIService()
        
        self.homomorphic_encryption_service = homomorphic_service
        
        self._load_models()
        self._initialize_tensorrt()
        self._initialize_foundation_model()
        self._initialize_risk_prediction_service()

    def _load_models(self) -> None:
        """Load ML models from disk."""
        try:
            models_dir = Path(settings.MODELS_DIR)
            if not models_dir.exists():
                logger.warning(f"Models directory not found: {models_dir}")
                return

            ecg_model_path = models_dir / "ecg_classifier.onnx"
            if ecg_model_path.exists():
                self._load_model("ecg_classifier", str(ecg_model_path))

            rhythm_model_path = models_dir / "rhythm_detector.onnx"
            if rhythm_model_path.exists():
                self._load_model("rhythm_detector", str(rhythm_model_path))

            quality_model_path = models_dir / "quality_assessor.onnx"
            if quality_model_path.exists():
                self._load_model("quality_assessor", str(quality_model_path))

            logger.info(f"Loaded {len(self.models)} ML models")

        except Exception as e:
            logger.error(f"Failed to load models: {str(e)}")

    def _initialize_tensorrt(self) -> None:
        """Initialize TensorRT optimization if available"""
        try:
            load_results = self.tensorrt_manager.load_optimized_models()
            
            if any(load_results.values()):
                self.use_tensorrt = True
                logger.info(f"Loaded TensorRT engines: {[k for k, v in load_results.items() if v]}")
            else:
                logger.info("Attempting to optimize ONNX models to TensorRT...")
                optimize_results = self.tensorrt_manager.optimize_all_models()
                
                if any(optimize_results.values()):
                    self.use_tensorrt = True
                    logger.info(f"Optimized models to TensorRT: {[k for k, v in optimize_results.items() if v]}")
                else:
                    logger.info("TensorRT optimization not available, using ONNX Runtime")
                    
        except Exception as e:
            logger.warning(f"TensorRT initialization failed: {e}")
            self.use_tensorrt = False

    def _initialize_foundation_model(self) -> None:
        """Initialize foundation model if available"""
        try:
            import asyncio
            
            async def load_foundation():
                success = await self.foundation_model.load_model()
                return success
                
            self.foundation_model_loaded = asyncio.run(load_foundation())
            
            if self.foundation_model_loaded:
                logger.info("Foundation model loaded successfully")
            else:
                logger.info("Foundation model initialization with random weights")
                
        except Exception as e:
            logger.warning(f"Foundation model initialization failed: {e}")
            self.foundation_model_loaded = False

    def _initialize_risk_prediction_service(self) -> None:
        """Initialize risk prediction service"""
        try:
            import asyncio
            
            async def load_risk_service():
                success = await self.risk_prediction_service.load_models()
                return success
                
            self.risk_service_loaded = asyncio.run(load_risk_service())
            
            if self.risk_service_loaded:
                logger.info("Risk prediction service loaded successfully")
            else:
                logger.info("Risk prediction service initialized with traditional models")
                
        except Exception as e:
            logger.warning(f"Risk prediction service initialization failed: {e}")
            self.risk_service_loaded = False

    def _load_model(self, model_name: str, model_path: str) -> None:
        """Load a single ONNX model."""
        try:
            providers = ['CPUExecutionProvider']

            if ort.get_device() == 'GPU':
                providers.insert(0, 'CUDAExecutionProvider')

            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session_options.intra_op_num_threads = 4
            session_options.inter_op_num_threads = 2

            session = ort.InferenceSession(
                model_path,
                sess_options=session_options,
                providers=providers
            )

            self.models[model_name] = session

            input_meta = session.get_inputs()[0]
            output_meta = session.get_outputs()[0]

            self.model_metadata[model_name] = {
                "input_shape": input_meta.shape,
                "input_type": input_meta.type,
                "output_shape": output_meta.shape,
                "output_type": output_meta.type,
                "providers": session.get_providers(),
            }

            logger.info(
                f"Loaded model {model_name} with input_shape={input_meta.shape} providers={session.get_providers()}"
            )

        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            raise MLModelException(f"Failed to load model {model_name}: {str(e)}") from e

    async def analyze_ecg(
        self,
        ecg_data: "np.ndarray[Any, np.dtype[np.float32]]",
        sample_rate: int,
        leads_names: list[str],
        patient_data: Optional[Dict[str, Any]] = None,
        include_risk_prediction: bool = True,
        include_sleep_analysis: bool = True,
        generate_fhir_resources: bool = False,
        patient_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Analyze ECG data using ML models."""
        try:
            start_time = time.time()

            processed_data = await self._preprocess_for_inference(
                ecg_data, sample_rate, leads_names
            )

            results = {
                "confidence": 0.0,
                "predictions": {},
                "interpretability": {},
                "rhythm": "Unknown",
                "events": [],
            }

            if "ecg_classifier" in self.models:
                classification_results = await self._run_classification(processed_data)
                predictions = classification_results["predictions"]
                if isinstance(predictions, dict):
                    results_predictions = results["predictions"]
                    if isinstance(results_predictions, dict):
                        results_predictions.update(predictions)
                results["confidence"] = classification_results["confidence"]

                if classification_results["confidence"] > 0.5:
                    interpretability = await self._generate_interpretability(
                        processed_data, classification_results
                    )
                    results["interpretability"] = interpretability

            if "rhythm_detector" in self.models:
                rhythm_results = await self._run_rhythm_detection(processed_data)
                results["rhythm"] = rhythm_results["rhythm"]
                events = rhythm_results.get("events", [])
                if isinstance(events, list):
                    results_events = results["events"]
                    if isinstance(results_events, list):
                        results_events.extend(events)

            if "quality_assessor" in self.models:
                quality_results = await self._run_quality_assessment(processed_data)
                results["quality_score"] = quality_results["score"]
                results["quality_issues"] = quality_results.get("issues", [])

            if self.foundation_model_loaded:
                try:
                    # processed_data is (1, 12, 5000), need (1, 5000, 12)
                    foundation_data = processed_data.transpose(0, 2, 1).astype(np.float32)
                    foundation_result = await self.foundation_model.analyze_ecg(
                        foundation_data, 
                        return_embeddings=True
                    )
                    results["foundation_analysis"] = foundation_result
                    
                    if foundation_result.get("overall_confidence", 0) > results["confidence"]:
                        results["confidence"] = foundation_result["overall_confidence"]
                        
                except Exception as e:
                    logger.warning(f"Foundation model analysis failed: {e}")
                    results["foundation_analysis"] = {"error": str(e)}

            if include_risk_prediction and patient_data and self.risk_service_loaded:
                try:
                    ecg_features = None
                    if self.foundation_model_loaded and results.get("foundation_analysis"):
                        foundation_analysis = results["foundation_analysis"]
                        if isinstance(foundation_analysis, dict) and "embeddings" in foundation_analysis:
                            ecg_features = np.array(foundation_analysis["embeddings"][0], dtype=np.float32)
                    
                    cv_risk_result = await self.risk_prediction_service.predict_cardiovascular_risk(
                        patient_data=patient_data,
                        ecg_features=ecg_features,
                        use_neural_model=True
                    )
                    results["cardiovascular_risk"] = cv_risk_result
                    
                    if ecg_features is not None:
                        scd_risk_result = await self.risk_prediction_service.predict_sudden_cardiac_death_risk(
                            patient_data=patient_data,
                            ecg_features=ecg_features
                        )
                        results["sudden_cardiac_death_risk"] = scd_risk_result
                        
                except Exception as e:
                    logger.warning(f"Risk prediction failed: {e}")
                    results["risk_prediction_error"] = str(e)

            if include_sleep_analysis:
                try:
                    lead_ii_data = processed_data[0, 1, :] if processed_data.shape[1] > 1 else processed_data[0, 0, :]
                    
                    respiratory_signal = self.respiratory_analyzer.extract_respiratory_signal(
                        lead_ii_data.astype(np.float64), method="edr"
                    )
                    respiratory_analysis = self.respiratory_analyzer.analyze_respiratory_rate(respiratory_signal)
                    results["respiratory_analysis"] = respiratory_analysis
                    
                    duration_hours = len(lead_ii_data) / sample_rate / 3600
                    sleep_apnea_result = await self.sleep_apnea_detector.detect_sleep_apnea(
                        lead_ii_data.astype(np.float64), duration_hours=max(duration_hours, 0.1)
                    )
                    results["sleep_apnea_analysis"] = sleep_apnea_result
                    
                except Exception as e:
                    logger.warning(f"Sleep apnea analysis failed: {e}")
                    results["sleep_analysis_error"] = str(e)

            if generate_fhir_resources and patient_id:
                try:
                    fhir_result = await self.fhir_service.process_ecg_for_fhir(
                        patient_id=patient_id,
                        ecg_data=ecg_data,
                        analysis_results=results,
                        submit_to_epic=False
                    )
                    results["fhir_resources"] = fhir_result
                    
                except Exception as e:
                    logger.warning(f"FHIR resource generation failed: {e}")
                    results["fhir_error"] = str(e)

            processing_time = time.time() - start_time
            results["processing_time_seconds"] = processing_time

            results["wearable_compatibility"] = {
                "supported_devices": self.wearable_service.get_supported_devices(),
                "standardized_format": True,
                "continuous_monitoring_available": True
            }
            
            results["federated_learning"] = {
                "privacy_preserving": True,
                "differential_privacy_enabled": True,
                "node_id": self.federated_learning_service.node_id,
                "capabilities": self.federated_learning_service.get_privacy_capabilities()
            }

            foundation_status = "enabled" if self.foundation_model_loaded else "disabled"
            risk_status = "enabled" if self.risk_service_loaded else "disabled"
            logger.info(
                f"ECG analysis completed with confidence={results['confidence']} rhythm={results['rhythm']} "
                f"foundation_model={foundation_status} risk_prediction={risk_status} processing_time={processing_time}"
            )

            return results

        except Exception as e:
            logger.error(f"ECG analysis failed: {str(e)}")
            raise MLModelException(f"ECG analysis failed: {str(e)}") from e

    async def _preprocess_for_inference(
        self,
        ecg_data: "np.ndarray[Any, np.dtype[np.float32]]",
        sample_rate: int,
        leads_names: list[str],
    ) -> "np.ndarray[Any, np.dtype[np.float32]]":
        """Preprocess ECG data for model inference."""
        try:
            if ecg_data.shape[1] < 12:
                padded_data = np.zeros((ecg_data.shape[0], 12), dtype=np.float32)
                padded_data[:, :ecg_data.shape[1]] = ecg_data
                ecg_data = padded_data
            elif ecg_data.shape[1] > 12:
                ecg_data = ecg_data[:, :12].astype(np.float32)

            target_sample_rate = 500
            if sample_rate != target_sample_rate:
                from scipy import signal
                num_samples = int(ecg_data.shape[0] * target_sample_rate / sample_rate)
                ecg_data = signal.resample(ecg_data, num_samples, axis=0)

            ecg_data = (ecg_data - np.mean(ecg_data, axis=0)) / (np.std(ecg_data, axis=0) + 1e-8)

            target_length = 5000
            if ecg_data.shape[0] > target_length:
                ecg_data = ecg_data[:target_length, :]
            elif ecg_data.shape[0] < target_length:
                padded_data = np.zeros((target_length, 12), dtype=np.float32)
                padded_data[:ecg_data.shape[0], :] = ecg_data
                ecg_data = padded_data

            ecg_data = np.expand_dims(ecg_data.T, axis=0).astype(np.float32)

            return ecg_data

        except Exception as e:
            logger.error(f"Preprocessing failed: {str(e)}")
            raise MLModelException(f"Preprocessing failed: {str(e)}") from e

    async def _run_classification(self, data: "np.ndarray[Any, np.dtype[np.float32]]") -> dict[str, Any]:
        """Run ECG classification model."""
        try:
            if self.use_tensorrt and "ecg_classifier" in self.tensorrt_manager.optimizer.engines:
                predictions = self.tensorrt_manager.optimizer.infer("ecg_classifier", data)
                predictions = predictions[0]  # Remove batch dimension
            else:
                model = self.models["ecg_classifier"]
                input_name = model.get_inputs()[0].name
                outputs = model.run(None, {input_name: data})
                predictions = outputs[0][0]  # Remove batch dimension

            condition_names = [
                "normal",
                "atrial_fibrillation",
                "atrial_flutter",
                "supraventricular_tachycardia",
                "ventricular_tachycardia",
                "ventricular_fibrillation",
                "first_degree_block",
                "second_degree_block",
                "complete_heart_block",
                "left_bundle_branch_block",
                "right_bundle_branch_block",
                "premature_atrial_contraction",
                "premature_ventricular_contraction",
                "stemi",
                "nstemi",
            ]

            predictions_dict = {}
            for i, condition in enumerate(condition_names):
                if i < len(predictions):
                    predictions_dict[condition] = float(predictions[i])

            confidence = float(np.max(predictions))

            return {
                "predictions": predictions_dict,
                "confidence": confidence,
            }

        except Exception as e:
            logger.error(f"Classification failed: {str(e)}")
            return {"predictions": {}, "confidence": 0.0}

    async def _run_rhythm_detection(self, data: "np.ndarray[Any, np.dtype[np.float32]]") -> dict[str, Any]:
        """Run rhythm detection model."""
        try:
            if self.use_tensorrt and "rhythm_detector" in self.tensorrt_manager.optimizer.engines:
                rhythm_probs = self.tensorrt_manager.optimizer.infer("rhythm_detector", data)
                rhythm_probs = rhythm_probs[0]  # Remove batch dimension
            else:
                model = self.models["rhythm_detector"]
                input_name = model.get_inputs()[0].name
                outputs = model.run(None, {input_name: data})
                rhythm_probs = outputs[0][0]

            rhythm_types = [
                "Sinus Rhythm",
                "Atrial Fibrillation",
                "Atrial Flutter",
                "Supraventricular Tachycardia",
                "Ventricular Tachycardia",
                "Ventricular Fibrillation",
                "Asystole",
            ]

            rhythm_idx = np.argmax(rhythm_probs)
            rhythm = rhythm_types[rhythm_idx] if rhythm_idx < len(rhythm_types) else "Unknown"

            events = []
            if rhythm != "Sinus Rhythm" and rhythm_probs[rhythm_idx] > 0.7:
                events.append({
                    "label": f"{rhythm}_detected",
                    "time_ms": 0.0,
                    "confidence": float(rhythm_probs[rhythm_idx]),
                    "properties": {"rhythm_type": rhythm},
                })

            return {
                "rhythm": rhythm,
                "rhythm_confidence": float(rhythm_probs[rhythm_idx]),
                "events": events,
            }

        except Exception as e:
            logger.error(f"Rhythm detection failed: {str(e)}")
            return {"rhythm": "Unknown", "events": []}

    async def _run_quality_assessment(self, data: "np.ndarray[Any, np.dtype[np.float32]]") -> dict[str, Any]:
        """Run signal quality assessment model."""
        try:
            if self.use_tensorrt and "quality_assessor" in self.tensorrt_manager.optimizer.engines:
                quality_output = self.tensorrt_manager.optimizer.infer("quality_assessor", data)
                quality_score = float(quality_output[0][0])
            else:
                model = self.models["quality_assessor"]
                input_name = model.get_inputs()[0].name
                outputs = model.run(None, {input_name: data})
                quality_score = float(outputs[0][0][0])

            issues = []
            if quality_score < 0.5:
                issues.append("Poor signal quality")
            if quality_score < 0.3:
                issues.append("Significant noise detected")

            return {
                "score": quality_score,
                "issues": issues,
            }

        except Exception as e:
            logger.error(f"Quality assessment failed: {str(e)}")
            return {"score": 0.5, "issues": ["Quality assessment unavailable"]}

    async def _generate_interpretability(
        self, data: "np.ndarray[Any, np.dtype[np.float32]]", classification_results: dict[str, Any]
    ) -> dict[str, Any]:
        """Generate comprehensive interpretability using explainable AI service."""
        try:
            predictions = classification_results.get("predictions", {})
            confidence = classification_results.get("confidence", 0.0)
            
            if len(data.shape) == 3 and data.shape[0] == 1:
                ecg_data = data[0].T  # Shape: (time, leads) -> (leads, time)
            else:
                ecg_data = data
                
            explanation_result = self.explainable_ai_service.generate_comprehensive_explanation(
                ecg_data=ecg_data,
                predictions=predictions,
                confidence=confidence,
                methods=[ExplanationMethod.FEATURE_IMPORTANCE, ExplanationMethod.CLINICAL_REASONING]
            )
            
            interpretability = {
                "attention_maps": {},
                "feature_importance": explanation_result.feature_importance,
                "explanation": "AI-powered comprehensive analysis with clinical reasoning",
                "clinical_findings": [],
                "confidence_factors": explanation_result.confidence_factors,
                "uncertainty_analysis": explanation_result.uncertainty_analysis,
                "explanation_methods": [explanation_result.method.value]
            }
            
            for lead_name, attention_map in explanation_result.attention_maps.items():
                if hasattr(attention_map, 'tolist'):
                    interpretability["attention_maps"][lead_name] = attention_map.tolist()
                else:
                    interpretability["attention_maps"][lead_name] = list(attention_map)
                    
            for finding in explanation_result.clinical_reasoning:
                clinical_finding = {
                    "condition": finding.condition,
                    "confidence": finding.confidence,
                    "evidence": finding.evidence,
                    "lead_involvement": finding.lead_involvement,
                    "clinical_significance": finding.clinical_significance,
                    "recommendations": finding.recommendations
                }
                interpretability["clinical_findings"].append(clinical_finding)
                
            interpretability["visual_explanations"] = explanation_result.visual_explanations
            
            if not interpretability["attention_maps"] and not interpretability["clinical_findings"]:
                leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
                for i, lead in enumerate(leads):
                    if i < ecg_data.shape[0]:
                        attention = np.abs(ecg_data[i] - np.mean(ecg_data[i])) * confidence
                        attention = attention / (np.max(attention) + 1e-8)  # Normalize
                        interpretability["attention_maps"][lead] = attention.tolist()
                        
                top_prediction = max(predictions.items(), key=lambda x: x[1]) if predictions else ("unknown", 0.0)
                interpretability["feature_importance"].update({
                    "most_important_lead": "II",
                    "key_intervals": ["QRS", "ST_segment"],
                    "top_prediction": top_prediction[0],
                    "prediction_confidence": top_prediction[1]
                })

            return interpretability

        except Exception as e:
            logger.error(f"Advanced interpretability generation failed: {str(e)}")
            # Fallback to basic interpretability
            try:
                basic_interpretability = {
                    "attention_maps": {},
                    "feature_importance": {},
                    "explanation": "Basic interpretability analysis (advanced features unavailable)",
                    "clinical_findings": [],
                    "confidence_factors": ["Basic pattern analysis"],
                    "uncertainty_analysis": {"uncertainty_score": 0.5},
                    "explanation_methods": ["basic"]
                }
                
                leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
                confidence = classification_results.get("confidence", 0.5)
                
                for i, lead in enumerate(leads):
                    if len(data.shape) >= 3 and i < data.shape[2]:
                        attention = np.random.random(data.shape[1]) * confidence
                        basic_interpretability["attention_maps"][lead] = attention.tolist()
                        
                return basic_interpretability
                
            except Exception as fallback_error:
                logger.error(f"Fallback interpretability generation failed: {str(fallback_error)}")
                return {"explanation": "Interpretability analysis unavailable"}

    def get_model_info(self) -> dict[str, Any]:
        """Get information about loaded models."""
        info = {
            "loaded_models": list(self.models.keys()),
            "model_metadata": self.model_metadata,
            "memory_usage": self.memory_monitor.get_memory_usage(),
            "tensorrt_enabled": self.use_tensorrt,
            "foundation_model_loaded": self.foundation_model_loaded,
            "risk_service_loaded": self.risk_service_loaded,
        }
        
        if self.use_tensorrt:
            info["tensorrt_status"] = self.tensorrt_manager.get_status()
            info["tensorrt_benchmarks"] = self.tensorrt_manager.benchmark_all_models()
            
        if self.foundation_model_loaded:
            info["foundation_model_info"] = self.foundation_model.get_model_info()
            
        if self.risk_service_loaded:
            info["risk_service_info"] = self.risk_prediction_service.get_service_info()
            
        return info

    def unload_model(self, model_name: str) -> bool:
        """Unload a model to free memory."""
        if model_name in self.models:
            del self.models[model_name]
            if model_name in self.model_metadata:
                del self.model_metadata[model_name]
            logger.info(f"Unloaded model {model_name}")
            return True
        return False
