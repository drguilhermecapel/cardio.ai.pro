"""
ECG Analysis Service - Core ECG processing and analysis functionality.
"""

import asyncio
import hashlib
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import neurokit2 as nk
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.constants import AnalysisStatus, ClinicalUrgency, DiagnosisCategory
from app.core.exceptions import ECGProcessingException
from app.models.ecg_analysis import ECGAnalysis, ECGAnnotation, ECGMeasurement
from app.repositories.ecg_repository import ECGRepository
from app.services.ml_model_service import MLModelService
from app.services.validation_service import ValidationService
from app.utils.ecg_processor import ECGProcessor
from app.utils.signal_quality import SignalQualityAnalyzer

logger = logging.getLogger(__name__)


class ECGAnalysisService:
    """ECG Analysis Service for processing and analyzing ECG data."""

    def __init__(
        self,
        db: AsyncSession,
        ml_service: MLModelService,
        validation_service: ValidationService,
    ) -> None:
        self.db = db
        self.repository = ECGRepository(db)
        self.ml_service = ml_service
        self.validation_service = validation_service
        self.processor = ECGProcessor()
        self.quality_analyzer = SignalQualityAnalyzer()

    async def create_analysis(
        self,
        patient_id: int,
        file_path: str,
        original_filename: str,
        created_by: int,
        metadata: dict[str, Any] | None = None,
    ) -> ECGAnalysis:
        """Create a new ECG analysis."""
        try:
            analysis_id = f"ECG_{uuid.uuid4().hex[:12].upper()}"

            file_hash, file_size = await self._calculate_file_info(file_path)

            ecg_metadata = await self.processor.extract_metadata(file_path)

            analysis = ECGAnalysis()
            analysis.analysis_id = analysis_id
            analysis.patient_id = patient_id
            analysis.created_by = created_by
            analysis.original_filename = original_filename
            analysis.file_path = file_path
            analysis.file_hash = file_hash
            analysis.file_size = file_size
            analysis.acquisition_date = ecg_metadata.get("acquisition_date", datetime.utcnow())
            analysis.sample_rate = ecg_metadata.get("sample_rate", settings.ECG_SAMPLE_RATE)
            analysis.duration_seconds = ecg_metadata.get("duration_seconds", 10.0)
            analysis.leads_count = ecg_metadata.get("leads_count", 12)
            analysis.leads_names = ecg_metadata.get("leads_names", settings.ECG_LEADS)
            analysis.device_manufacturer = ecg_metadata.get("device_manufacturer")
            analysis.device_model = ecg_metadata.get("device_model")
            analysis.device_serial = ecg_metadata.get("device_serial")
            analysis.status = AnalysisStatus.PENDING
            analysis.clinical_urgency = ClinicalUrgency.LOW
            analysis.requires_immediate_attention = False
            analysis.is_validated = False
            analysis.validation_required = True

            analysis = await self.repository.create_analysis(analysis)

            asyncio.create_task(self._process_analysis_async(analysis.id))

            logger.info(
                f"ECG analysis created: analysis_id={analysis_id}, patient_id={patient_id}, filename={original_filename}"
            )

            return analysis

        except Exception as e:
            logger.error(
                f"Failed to create ECG analysis: error={str(e)}, patient_id={patient_id}, filename={original_filename}"
            )
            raise ECGProcessingException(f"Failed to create analysis: {str(e)}") from e

    async def _process_analysis_async(self, analysis_id: int) -> None:
        """Process ECG analysis asynchronously."""
        analysis = None  # Initialize analysis variable to avoid UnboundLocalError
        try:
            await self.repository.update_analysis_status(
                analysis_id, AnalysisStatus.PROCESSING
            )

            start_time = datetime.utcnow()

            analysis = await self.repository.get_analysis_by_id(analysis_id)
            if not analysis:
                raise ECGProcessingException(f"Analysis {analysis_id} not found")

            ecg_data = await self.processor.load_ecg_file(analysis.file_path)
            preprocessed_data = await self.processor.preprocess_signal(ecg_data)

            quality_metrics = await self.quality_analyzer.analyze_quality(
                preprocessed_data
            )

            ai_results = await self.ml_service.analyze_ecg(
                preprocessed_data.astype(np.float32),
                analysis.sample_rate,
                analysis.leads_names,
            )

            measurements = await self._extract_measurements(
                preprocessed_data, analysis.sample_rate
            )

            annotations = await self._generate_annotations(
                preprocessed_data, ai_results, analysis.sample_rate
            )

            clinical_assessment = await self._assess_clinical_urgency(ai_results)

            end_time = datetime.utcnow()
            processing_duration_ms = int((end_time - start_time).total_seconds() * 1000)

            update_data = {
                "status": AnalysisStatus.COMPLETED,
                "processing_started_at": start_time,
                "processing_completed_at": end_time,
                "processing_duration_ms": processing_duration_ms,
                "ai_confidence": ai_results.get("confidence", 0.0),
                "ai_predictions": ai_results.get("predictions", {}),
                "ai_interpretability": ai_results.get("interpretability", {}),
                "heart_rate_bpm": measurements.get("heart_rate"),
                "rhythm": ai_results.get("rhythm"),
                "pr_interval_ms": measurements.get("pr_interval"),
                "qrs_duration_ms": measurements.get("qrs_duration"),
                "qt_interval_ms": measurements.get("qt_interval"),
                "qtc_interval_ms": measurements.get("qtc_interval"),
                "primary_diagnosis": clinical_assessment.get("primary_diagnosis"),
                "secondary_diagnoses": clinical_assessment.get("secondary_diagnoses", []),
                "diagnosis_category": clinical_assessment.get("category"),
                "icd10_codes": clinical_assessment.get("icd10_codes", []),
                "clinical_urgency": clinical_assessment.get("urgency", ClinicalUrgency.LOW),
                "requires_immediate_attention": clinical_assessment.get("critical", False),
                "recommendations": clinical_assessment.get("recommendations", []),
                "signal_quality_score": quality_metrics.get("overall_score", 0.0),
                "noise_level": quality_metrics.get("noise_level", 0.0),
                "baseline_wander": quality_metrics.get("baseline_wander", 0.0),
            }

            await self.repository.update_analysis(analysis_id, update_data)

            for measurement_data in measurements.get("detailed_measurements", []):
                measurement = ECGMeasurement()
                measurement.analysis_id = analysis_id
                for key, value in measurement_data.items():
                    setattr(measurement, key, value)
                await self.repository.create_measurement(measurement)

            for annotation_data in annotations:
                annotation = ECGAnnotation()
                annotation.analysis_id = analysis_id
                for key, value in annotation_data.items():
                    setattr(annotation, key, value)
                await self.repository.create_annotation(annotation)

            if clinical_assessment.get("critical", False):
                await self.validation_service.create_urgent_validation(analysis_id)

            logger.info(
                f"ECG analysis completed successfully: analysis_id={analysis.analysis_id}, processing_time_ms={processing_duration_ms}, confidence={ai_results.get('confidence')}, urgency={clinical_assessment.get('urgency')}"
            )

        except Exception as e:
            logger.error(
                f"ECG analysis processing failed: analysis_id={analysis_id}, error={str(e)}"
            )

            await self.repository.update_analysis(
                analysis_id,
                {
                    "status": AnalysisStatus.FAILED,
                    "error_message": str(e),
                    "retry_count": analysis.retry_count + 1 if analysis else 1,
                }
            )

            if analysis and analysis.retry_count < 3:
                logger.info(f"Retrying analysis {analysis_id} in 60 seconds")
                await asyncio.sleep(60)
                asyncio.create_task(self._process_analysis_async(analysis_id))

    async def _calculate_file_info(self, file_path: str) -> tuple[str, int]:
        """Calculate file hash and size."""
        path = Path(file_path)
        if not path.exists():
            raise ECGProcessingException(f"File not found: {file_path}")

        hash_sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)

        file_hash = hash_sha256.hexdigest()
        file_size = path.stat().st_size

        return file_hash, file_size

    async def _extract_measurements(
        self, ecg_data: np.ndarray[Any, np.dtype[np.float64]], sample_rate: int
    ) -> dict[str, Any]:
        """Extract clinical measurements from ECG data."""
        try:
            measurements: dict[str, Any] = {}
            detailed_measurements: list[dict[str, Any]] = []

            signals, info = nk.ecg_process(ecg_data, sampling_rate=sample_rate)

            heart_rate = np.mean(info["ECG_Rate"])
            measurements["heart_rate"] = int(heart_rate) if not np.isnan(heart_rate) else None

            if "ECG_R_Peaks" in info:
                r_peaks = info["ECG_R_Peaks"]
                if len(r_peaks) > 1:
                    rr_intervals = np.diff(r_peaks) / sample_rate * 1000  # ms

                    pr_interval = np.mean(rr_intervals) * 0.16  # Approximate
                    measurements["pr_interval"] = int(pr_interval)

                    qrs_duration = 100  # Default value - should be calculated from signal
                    measurements["qrs_duration"] = qrs_duration

                    qt_interval = np.mean(rr_intervals) * 0.4  # Approximate
                    measurements["qt_interval"] = int(qt_interval)

                    qtc_interval = qt_interval / np.sqrt(np.mean(rr_intervals) / 1000)
                    measurements["qtc_interval"] = int(qtc_interval)

            for i, lead_name in enumerate(["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]):
                if i < ecg_data.shape[1]:
                    lead_data = ecg_data[:, i]

                    detailed_measurements.append({
                        "measurement_type": "amplitude",
                        "lead_name": lead_name,
                        "value": float(np.max(lead_data) - np.min(lead_data)),
                        "unit": "mV",
                        "confidence": 0.9,
                        "source": "algorithm",
                    })

            measurements["detailed_measurements"] = detailed_measurements

            return measurements

        except Exception as e:
            logger.error(f"Failed to extract measurements: {str(e)}")
            return {"heart_rate": None, "detailed_measurements": []}

    async def _generate_annotations(
        self,
        ecg_data: np.ndarray[Any, np.dtype[np.float64]],
        ai_results: dict[str, Any],
        sample_rate: int,
    ) -> list[dict[str, Any]]:
        """Generate ECG annotations."""
        annotations = []

        try:
            signals, info = nk.ecg_process(ecg_data, sampling_rate=sample_rate)

            if "ECG_R_Peaks" in info:
                r_peaks = info["ECG_R_Peaks"]

                for peak_idx in r_peaks:
                    time_ms = (peak_idx / sample_rate) * 1000

                    annotations.append({
                        "annotation_type": "beat",
                        "label": "R_peak",
                        "time_ms": float(time_ms),
                        "confidence": 0.95,
                        "source": "algorithm",
                        "properties": {"peak_amplitude": float(ecg_data[peak_idx, 1])},
                    })

            if "events" in ai_results:
                for event in ai_results["events"]:
                    annotations.append({
                        "annotation_type": "event",
                        "label": event.get("label", "unknown"),
                        "time_ms": float(event.get("time_ms", 0)),
                        "confidence": float(event.get("confidence", 0.5)),
                        "source": "ai",
                        "properties": event.get("properties", {}),
                    })

            return annotations

        except Exception as e:
            logger.error(f"Failed to generate annotations: {str(e)}")
            return []

    async def _assess_clinical_urgency(
        self, ai_results: dict[str, Any]
    ) -> dict[str, Any]:
        """Assess clinical urgency based on AI results."""
        assessment = {
            "urgency": ClinicalUrgency.LOW,
            "critical": False,
            "primary_diagnosis": "Normal ECG",
            "secondary_diagnoses": [],
            "category": DiagnosisCategory.NORMAL,
            "icd10_codes": [],
            "recommendations": [],
        }

        try:
            predictions = ai_results.get("predictions", {})
            confidence = ai_results.get("confidence", 0.0)

            critical_conditions = [
                "ventricular_fibrillation",
                "ventricular_tachycardia",
                "complete_heart_block",
                "stemi",
                "asystole",
            ]

            for condition in critical_conditions:
                if predictions.get(condition, 0.0) > 0.7:
                    assessment["urgency"] = ClinicalUrgency.CRITICAL
                    assessment["critical"] = True
                    assessment["primary_diagnosis"] = condition.replace("_", " ").title()
                    assessment["category"] = DiagnosisCategory.ARRHYTHMIA
                    recommendations = assessment["recommendations"]
                    if isinstance(recommendations, list):
                        recommendations.append("Immediate medical attention required")
                    break

            if not assessment["critical"]:
                high_priority_conditions = [
                    "atrial_fibrillation",
                    "supraventricular_tachycardia",
                    "first_degree_block",
                ]

                for condition in high_priority_conditions:
                    if predictions.get(condition, 0.0) > 0.6:
                        assessment["urgency"] = ClinicalUrgency.HIGH
                        assessment["primary_diagnosis"] = condition.replace("_", " ").title()
                        assessment["category"] = DiagnosisCategory.ARRHYTHMIA
                        recommendations = assessment["recommendations"]
                        if isinstance(recommendations, list):
                            recommendations.append("Cardiology consultation recommended")
                        break

            if confidence < 0.7:
                recommendations = assessment["recommendations"]
                if isinstance(recommendations, list):
                    recommendations.append("Manual review recommended due to low AI confidence")

            return assessment

        except Exception as e:
            logger.error(f"Failed to assess clinical urgency: {str(e)}")
            return assessment

    async def get_analysis_by_id(self, analysis_id: int) -> ECGAnalysis | None:
        """Get analysis by ID."""
        return await self.repository.get_analysis_by_id(analysis_id)

    async def get_analyses_by_patient(
        self, patient_id: int, limit: int = 50, offset: int = 0
    ) -> list[ECGAnalysis]:
        """Get analyses by patient ID."""
        return await self.repository.get_analyses_by_patient(patient_id, limit, offset)

    async def search_analyses(
        self,
        filters: dict[str, Any],
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[ECGAnalysis], int]:
        """Search analyses with filters."""
        return await self.repository.search_analyses(filters, limit, offset)

    async def delete_analysis(self, analysis_id: int) -> bool:
        """Delete analysis (soft delete for audit trail)."""
        return await self.repository.delete_analysis(analysis_id)
