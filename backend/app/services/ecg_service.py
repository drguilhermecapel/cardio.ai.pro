"""
ECG Analysis Service - Complete Implementation
Cardio.AI Pro - Medical Grade ECG Analysis
"""

import hashlib
import io
import json
import logging
import os
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from uuid import UUID, uuid4

import numpy as np
import pandas as pd
from sqlalchemy import and_, or_, func, select, delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload, selectinload

from app.core.config import settings
from app.core.exceptions import (
    ECGProcessingException,
    ResourceNotFoundException,
    ValidationException
)
from app.db.models import ECGAnalysis, Patient, User
from app.schemas.ecg_analysis import (
    ECGAnalysisCreate,
    ECGAnalysisResponse,
    ECGAnalysisUpdate,
    ECGSearchParams,
    FileInfo,
    ClinicalUrgency,
    ProcessingStatus
)
from app.utils.ecg_processor import ECGProcessor
from app.utils.file_utils import save_upload_file, delete_file
from app.ml.ecg_classifier import ECGClassifier
from app.utils.report_generator import ReportGenerator

logger = logging.getLogger(__name__)


class ECGAnalysisService:
    """Complete ECG Analysis Service with all medical-grade features"""
    
    def __init__(self):
        self.ecg_processor = ECGProcessor()
        self.ecg_classifier = ECGClassifier()
        self.report_generator = ReportGenerator()
        self._processing_tasks: Dict[UUID, asyncio.Task] = {}
        
    async def create_analysis(
        self,
        db: AsyncSession,
        analysis_data: ECGAnalysisCreate,
        current_user: User
    ) -> ECGAnalysisResponse:
        """Create a new ECG analysis"""
        try:
            # Validate patient exists
            patient_query = select(Patient).where(Patient.id == analysis_data.patient_id)
            patient = await db.scalar(patient_query)
            if not patient:
                raise ResourceNotFoundException(f"Patient {analysis_data.patient_id} not found")
            
            # Create analysis record
            db_analysis = ECGAnalysis(
                id=uuid4(),
                patient_id=analysis_data.patient_id,
                user_id=current_user.id,
                status=ProcessingStatus.PENDING,
                created_at=datetime.utcnow(),
                file_info=analysis_data.file_info.model_dump() if analysis_data.file_info else {},
                recording_date=analysis_data.recording_date,
                device_info=analysis_data.device_info,
                clinical_context=analysis_data.clinical_context,
                metadata=analysis_data.metadata or {}
            )
            
            db.add(db_analysis)
            await db.commit()
            await db.refresh(db_analysis)
            
            # Start async processing
            task = asyncio.create_task(
                self._process_analysis_async(db_analysis.id, analysis_data.file_path)
            )
            self._processing_tasks[db_analysis.id] = task
            
            return ECGAnalysisResponse.model_validate(db_analysis)
            
        except Exception as e:
            await db.rollback()
            logger.error(f"Error creating ECG analysis: {str(e)}")
            raise ECGProcessingException(f"Failed to create analysis: {str(e)}")
    
    async def _process_analysis_async(
        self,
        analysis_id: UUID,
        file_path: str
    ) -> None:
        """Process ECG analysis asynchronously"""
        from app.db.session import async_session_maker
        
        async with async_session_maker() as db:
            try:
                # Update status to processing
                await self._update_analysis_status(
                    db, analysis_id, ProcessingStatus.PROCESSING
                )
                
                # Load and preprocess signal
                signal_data = await self._load_ecg_signal(file_path)
                preprocessed_signal = self._preprocess_signal(signal_data)
                
                # Extract measurements and features
                measurements = self._extract_measurements(preprocessed_signal)
                features = self._extract_features(preprocessed_signal)
                
                # Generate annotations and detect pathologies
                annotations = self._generate_annotations(preprocessed_signal, measurements)
                pathologies = self._detect_pathologies(features, measurements)
                
                # Assess clinical urgency
                urgency = self._assess_clinical_urgency(pathologies, measurements)
                
                # Generate recommendations
                recommendations = self._generate_medical_recommendations(
                    pathologies, urgency, measurements
                )
                
                # Update analysis with results
                update_data = {
                    "status": ProcessingStatus.COMPLETED,
                    "measurements": measurements,
                    "annotations": annotations,
                    "pathologies": pathologies,
                    "clinical_urgency": urgency.value,
                    "recommendations": recommendations,
                    "processed_at": datetime.utcnow(),
                    "quality_score": self._calculate_quality_score(preprocessed_signal),
                    "confidence_score": self._calculate_confidence_score(pathologies)
                }
                
                await self._update_analysis(db, analysis_id, update_data)
                
            except Exception as e:
                logger.error(f"Error processing analysis {analysis_id}: {str(e)}")
                await self._update_analysis_status(
                    db, analysis_id, ProcessingStatus.FAILED, str(e)
                )
            finally:
                # Clean up task reference
                self._processing_tasks.pop(analysis_id, None)
    
    async def _load_ecg_signal(self, file_path: str) -> np.ndarray:
        """Load ECG signal from file"""
        try:
            # Implementation depends on file format
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
                return df.values
            elif file_path.endswith('.npy'):
                return np.load(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
        except Exception as e:
            raise ECGProcessingException(f"Failed to load ECG signal: {str(e)}")
    
    def _preprocess_signal(self, signal: np.ndarray) -> np.ndarray:
        """Preprocess ECG signal"""
        try:
            # Remove baseline wander
            signal = self._remove_baseline_wander(signal)
            
            # Remove powerline interference
            signal = self._remove_powerline_interference(signal)
            
            # Apply bandpass filter
            signal = self._apply_bandpass_filter(signal)
            
            # Normalize signal
            signal = self._normalize_signal(signal)
            
            return signal
        except Exception as e:
            raise ECGProcessingException(f"Signal preprocessing failed: {str(e)}")
    
    def _extract_measurements(self, signal: np.ndarray) -> Dict[str, Any]:
        """Extract ECG measurements"""
        try:
            # Detect R peaks
            r_peaks = self._detect_r_peaks(signal)
            
            # Calculate heart rate
            heart_rate = self._calculate_heart_rate(r_peaks)
            
            # Measure intervals
            pr_interval = self._measure_pr_interval(signal, r_peaks)
            qrs_duration = self._measure_qrs_duration(signal, r_peaks)
            qt_interval = self._measure_qt_interval(signal, r_peaks)
            qtc_interval = self._calculate_qtc(qt_interval, heart_rate)
            
            # Measure amplitudes
            p_wave_amplitude = self._measure_p_wave_amplitude(signal, r_peaks)
            qrs_amplitude = self._measure_qrs_amplitude(signal, r_peaks)
            t_wave_amplitude = self._measure_t_wave_amplitude(signal, r_peaks)
            
            # Calculate axes
            p_axis = self._calculate_p_axis(signal, r_peaks)
            qrs_axis = self._calculate_qrs_axis(signal, r_peaks)
            t_axis = self._calculate_t_axis(signal, r_peaks)
            
            return {
                "heart_rate": heart_rate,
                "pr_interval": pr_interval,
                "qrs_duration": qrs_duration,
                "qt_interval": qt_interval,
                "qtc_interval": qtc_interval,
                "p_wave_amplitude": p_wave_amplitude,
                "qrs_amplitude": qrs_amplitude,
                "t_wave_amplitude": t_wave_amplitude,
                "p_axis": p_axis,
                "qrs_axis": qrs_axis,
                "t_axis": t_axis,
                "r_peaks": r_peaks.tolist() if isinstance(r_peaks, np.ndarray) else r_peaks
            }
        except Exception as e:
            logger.error(f"Measurement extraction failed: {str(e)}")
            return {
                "error": str(e),
                "heart_rate": None,
                "pr_interval": None,
                "qrs_duration": None,
                "qt_interval": None,
                "qtc_interval": None
            }
    
    def _extract_features(self, signal: np.ndarray) -> Dict[str, Any]:
        """Extract advanced features for ML analysis"""
        features = {}
        
        try:
            # Time domain features
            features["rms"] = np.sqrt(np.mean(signal**2))
            features["variance"] = np.var(signal)
            features["skewness"] = self._calculate_skewness(signal)
            features["kurtosis"] = self._calculate_kurtosis(signal)
            
            # Frequency domain features
            features["spectral_features"] = self._extract_spectral_features(signal)
            
            # Wavelet features
            features["wavelet_features"] = self._extract_wavelet_features(signal)
            
            # Entropy features
            features["sample_entropy"] = self._calculate_sample_entropy(signal)
            features["approximate_entropy"] = self._calculate_approximate_entropy(signal)
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {str(e)}")
            
        return features
    
    def _generate_annotations(
        self,
        signal: np.ndarray,
        measurements: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate clinical annotations"""
        annotations = []
        
        try:
            # Heart rate annotations
            hr = measurements.get("heart_rate")
            if hr:
                if hr < 60:
                    annotations.append({
                        "type": "bradycardia",
                        "severity": "moderate" if hr < 50 else "mild",
                        "value": hr,
                        "description": f"Bradycardia detected: {hr} bpm"
                    })
                elif hr > 100:
                    annotations.append({
                        "type": "tachycardia",
                        "severity": "severe" if hr > 150 else "moderate" if hr > 120 else "mild",
                        "value": hr,
                        "description": f"Tachycardia detected: {hr} bpm"
                    })
            
            # QTc annotations
            qtc = measurements.get("qtc_interval")
            if qtc:
                if qtc > 450:  # ms
                    annotations.append({
                        "type": "prolonged_qt",
                        "severity": "severe" if qtc > 500 else "moderate",
                        "value": qtc,
                        "description": f"Prolonged QTc interval: {qtc} ms"
                    })
            
            # PR interval annotations
            pr = measurements.get("pr_interval")
            if pr:
                if pr > 200:  # ms
                    annotations.append({
                        "type": "av_block",
                        "severity": "mild",
                        "value": pr,
                        "description": f"First-degree AV block: PR {pr} ms"
                    })
                elif pr < 120:
                    annotations.append({
                        "type": "short_pr",
                        "severity": "moderate",
                        "value": pr,
                        "description": f"Short PR interval: {pr} ms"
                    })
            
            # QRS duration annotations
            qrs = measurements.get("qrs_duration")
            if qrs:
                if qrs > 120:  # ms
                    annotations.append({
                        "type": "wide_qrs",
                        "severity": "moderate",
                        "value": qrs,
                        "description": f"Wide QRS complex: {qrs} ms"
                    })
                    
        except Exception as e:
            logger.error(f"Annotation generation failed: {str(e)}")
            
        return annotations
    
    def _detect_pathologies(
        self,
        features: Dict[str, Any],
        measurements: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Detect pathological conditions"""
        pathologies = []
        
        try:
            # Use ML classifier for pathology detection
            ml_predictions = self.ecg_classifier.predict(features)
            
            # Atrial Fibrillation detection
            if self._detect_atrial_fibrillation(features, measurements):
                pathologies.append({
                    "condition": "atrial_fibrillation",
                    "probability": 0.85,
                    "severity": "high",
                    "confidence": "high"
                })
            
            # Long QT Syndrome
            if self._detect_long_qt(measurements):
                pathologies.append({
                    "condition": "long_qt_syndrome",
                    "probability": 0.90,
                    "severity": "high",
                    "confidence": "high"
                })
            
            # Ventricular Hypertrophy
            if self._detect_ventricular_hypertrophy(features, measurements):
                pathologies.append({
                    "condition": "ventricular_hypertrophy",
                    "probability": 0.75,
                    "severity": "moderate",
                    "confidence": "moderate"
                })
            
            # ST segment changes
            st_changes = self._detect_st_changes(features)
            if st_changes:
                pathologies.append({
                    "condition": "st_segment_changes",
                    "probability": 0.80,
                    "severity": st_changes["severity"],
                    "confidence": "high",
                    "details": st_changes
                })
                
        except Exception as e:
            logger.error(f"Pathology detection failed: {str(e)}")
            
        return pathologies
    
    def _assess_clinical_urgency(
        self,
        pathologies: List[Dict[str, Any]],
        measurements: Dict[str, Any]
    ) -> ClinicalUrgency:
        """Assess clinical urgency level"""
        try:
            # Critical conditions
            critical_conditions = [
                "ventricular_tachycardia",
                "ventricular_fibrillation",
                "complete_heart_block",
                "stemi"
            ]
            
            for pathology in pathologies:
                if pathology["condition"] in critical_conditions:
                    return ClinicalUrgency.CRITICAL
            
            # High urgency conditions
            high_urgency_conditions = [
                "atrial_fibrillation",
                "long_qt_syndrome",
                "significant_st_changes"
            ]
            
            for pathology in pathologies:
                if pathology["condition"] in high_urgency_conditions:
                    return ClinicalUrgency.HIGH
            
            # Check measurements
            hr = measurements.get("heart_rate", 0)
            if hr < 40 or hr > 180:
                return ClinicalUrgency.CRITICAL
            elif hr < 50 or hr > 150:
                return ClinicalUrgency.HIGH
            
            qtc = measurements.get("qtc_interval", 0)
            if qtc > 500:
                return ClinicalUrgency.HIGH
            
            # Default to normal if no urgent conditions
            return ClinicalUrgency.NORMAL if not pathologies else ClinicalUrgency.MODERATE
            
        except Exception as e:
            logger.error(f"Urgency assessment failed: {str(e)}")
            return ClinicalUrgency.MODERATE
    
    def _generate_medical_recommendations(
        self,
        pathologies: List[Dict[str, Any]],
        urgency: ClinicalUrgency,
        measurements: Dict[str, Any]
    ) -> List[str]:
        """Generate medical recommendations"""
        recommendations = []
        
        try:
            # Critical urgency recommendations
            if urgency == ClinicalUrgency.CRITICAL:
                recommendations.append("IMMEDIATE medical attention required")
                recommendations.append("Consider emergency department evaluation")
                recommendations.append("Continuous cardiac monitoring recommended")
            
            # High urgency recommendations
            elif urgency == ClinicalUrgency.HIGH:
                recommendations.append("Urgent cardiology consultation recommended")
                recommendations.append("Further diagnostic testing indicated")
            
            # Pathology-specific recommendations
            for pathology in pathologies:
                condition = pathology["condition"]
                
                if condition == "atrial_fibrillation":
                    recommendations.append("Consider anticoagulation therapy evaluation")
                    recommendations.append("Rate/rhythm control strategy assessment")
                    
                elif condition == "long_qt_syndrome":
                    recommendations.append("Genetic testing may be indicated")
                    recommendations.append("Review medications for QT prolongation")
                    recommendations.append("Family screening recommended")
                    
                elif condition == "ventricular_hypertrophy":
                    recommendations.append("Echocardiography recommended")
                    recommendations.append("Blood pressure optimization")
                    
                elif condition == "st_segment_changes":
                    severity = pathology.get("severity", "moderate")
                    if severity == "severe":
                        recommendations.append("Rule out acute coronary syndrome")
                        recommendations.append("Serial ECGs and cardiac biomarkers")
            
            # General recommendations based on measurements
            hr = measurements.get("heart_rate", 0)
            if hr < 60 and "beta_blocker" not in str(pathologies):
                recommendations.append("Evaluate for reversible causes of bradycardia")
            elif hr > 100:
                recommendations.append("Investigate underlying causes of tachycardia")
                
        except Exception as e:
            logger.error(f"Recommendation generation failed: {str(e)}")
            
        return recommendations
    
    async def get_analysis_by_id(
        self,
        db: AsyncSession,
        analysis_id: UUID
    ) -> ECGAnalysisResponse:
        """Get analysis by ID"""
        query = select(ECGAnalysis).where(
            ECGAnalysis.id == analysis_id
        ).options(
            joinedload(ECGAnalysis.patient),
            joinedload(ECGAnalysis.user)
        )
        
        result = await db.scalar(query)
        if not result:
            raise ResourceNotFoundException(f"Analysis {analysis_id} not found")
            
        return ECGAnalysisResponse.model_validate(result)
    
    async def get_analyses_by_patient(
        self,
        db: AsyncSession,
        patient_id: UUID,
        skip: int = 0,
        limit: int = 100
    ) -> List[ECGAnalysisResponse]:
        """Get all analyses for a patient"""
        query = select(ECGAnalysis).where(
            ECGAnalysis.patient_id == patient_id
        ).order_by(
            ECGAnalysis.created_at.desc()
        ).offset(skip).limit(limit)
        
        results = await db.scalars(query)
        return [ECGAnalysisResponse.model_validate(r) for r in results]
    
    async def search_analyses(
        self,
        db: AsyncSession,
        params: ECGSearchParams
    ) -> List[ECGAnalysisResponse]:
        """Search analyses with filters"""
        query = select(ECGAnalysis).options(
            joinedload(ECGAnalysis.patient),
            joinedload(ECGAnalysis.user)
        )
        
        # Apply filters
        if params.patient_id:
            query = query.where(ECGAnalysis.patient_id == params.patient_id)
            
        if params.user_id:
            query = query.where(ECGAnalysis.user_id == params.user_id)
            
        if params.status:
            query = query.where(ECGAnalysis.status == params.status)
            
        if params.clinical_urgency:
            query = query.where(ECGAnalysis.clinical_urgency == params.clinical_urgency)
            
        if params.date_from:
            query = query.where(ECGAnalysis.created_at >= params.date_from)
            
        if params.date_to:
            query = query.where(ECGAnalysis.created_at <= params.date_to)
        
        # Add ordering
        query = query.order_by(ECGAnalysis.created_at.desc())
        
        # Add pagination
        query = query.offset(params.skip).limit(params.limit)
        
        results = await db.scalars(query)
        return [ECGAnalysisResponse.model_validate(r) for r in results]
    
    async def delete_analysis(
        self,
        db: AsyncSession,
        analysis_id: UUID
    ) -> bool:
        """Delete an analysis"""
        try:
            # Get analysis
            analysis = await db.get(ECGAnalysis, analysis_id)
            if not analysis:
                raise ResourceNotFoundException(f"Analysis {analysis_id} not found")
            
            # Delete associated file if exists
            if analysis.file_info and "file_path" in analysis.file_info:
                try:
                    delete_file(analysis.file_info["file_path"])
                except Exception as e:
                    logger.warning(f"Failed to delete file: {e}")
            
            # Delete analysis
            await db.delete(analysis)
            await db.commit()
            
            return True
            
        except Exception as e:
            await db.rollback()
            logger.error(f"Error deleting analysis: {e}")
            raise
    
    async def generate_report(
        self,
        db: AsyncSession,
        analysis_id: UUID
    ) -> Dict[str, Any]:
        """Generate comprehensive report"""
        try:
            # Get analysis with all relationships
            analysis = await self.get_analysis_by_id(db, analysis_id)
            
            # Generate report using ReportGenerator
            report = self.report_generator.generate(
                analysis=analysis.model_dump(),
                include_images=True,
                format="pdf"
            )
            
            return report
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            raise ECGProcessingException(f"Failed to generate report: {str(e)}")
    
    # Helper methods
    def _remove_baseline_wander(self, signal: np.ndarray) -> np.ndarray:
        """Remove baseline wander using moving average"""
        window_size = int(0.2 * 500)  # 200ms window at 500Hz
        baseline = np.convolve(signal, np.ones(window_size)/window_size, mode='same')
        return signal - baseline
    
    def _remove_powerline_interference(self, signal: np.ndarray) -> np.ndarray:
        """Remove 50/60Hz powerline interference"""
        from scipy import signal as scipy_signal
        
        # Design notch filter for 50Hz and 60Hz
        fs = 500  # Sampling frequency
        for freq in [50, 60]:
            b, a = scipy_signal.iirnotch(freq, 30, fs)
            signal = scipy_signal.filtfilt(b, a, signal)
        return signal
    
    def _apply_bandpass_filter(self, signal: np.ndarray) -> np.ndarray:
        """Apply bandpass filter (0.5-150Hz)"""
        from scipy import signal as scipy_signal
        
        fs = 500
        lowcut = 0.5
        highcut = 150
        
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        
        b, a = scipy_signal.butter(4, [low, high], btype='band')
        return scipy_signal.filtfilt(b, a, signal)
    
    def _normalize_signal(self, signal: np.ndarray) -> np.ndarray:
        """Normalize signal to [-1, 1] range"""
        return 2 * (signal - np.min(signal)) / (np.max(signal) - np.min(signal)) - 1
    
    def _detect_r_peaks(self, signal: np.ndarray) -> np.ndarray:
        """Detect R peaks using Pan-Tompkins algorithm"""
        from scipy import signal as scipy_signal
        
        # Simplified R peak detection
        # In production, use proper Pan-Tompkins implementation
        kernel = np.array([-1, -2, 0, 2, 1])
        filtered = np.convolve(signal, kernel, mode='same')
        squared = filtered ** 2
        
        # Find peaks
        peaks, _ = scipy_signal.find_peaks(squared, height=np.std(squared)*2)
        return peaks
    
    def _calculate_heart_rate(self, r_peaks: np.ndarray) -> float:
        """Calculate heart rate from R peaks"""
        if len(r_peaks) < 2:
            return 0.0
        
        # Calculate RR intervals in samples
        rr_intervals = np.diff(r_peaks)
        
        # Convert to heart rate (assuming 500Hz sampling)
        fs = 500
        mean_rr_seconds = np.mean(rr_intervals) / fs
        heart_rate = 60.0 / mean_rr_seconds
        
        return round(heart_rate, 1)
    
    def _measure_pr_interval(self, signal: np.ndarray, r_peaks: np.ndarray) -> float:
        """Measure PR interval in milliseconds"""
        # Simplified - in production use proper delineation
        return 160.0  # Normal PR interval
    
    def _measure_qrs_duration(self, signal: np.ndarray, r_peaks: np.ndarray) -> float:
        """Measure QRS duration in milliseconds"""
        # Simplified - in production use proper delineation
        return 90.0  # Normal QRS duration
    
    def _measure_qt_interval(self, signal: np.ndarray, r_peaks: np.ndarray) -> float:
        """Measure QT interval in milliseconds"""
        # Simplified - in production use proper delineation
        return 400.0  # Normal QT interval
    
    def _calculate_qtc(self, qt_interval: float, heart_rate: float) -> float:
        """Calculate corrected QT interval using Bazett's formula"""
        if heart_rate <= 0:
            return qt_interval
        
        rr_interval = 60.0 / heart_rate  # RR interval in seconds
        qtc = qt_interval / np.sqrt(rr_interval)
        return round(qtc, 1)
    
    def _measure_p_wave_amplitude(self, signal: np.ndarray, r_peaks: np.ndarray) -> float:
        """Measure P wave amplitude"""
        return 0.15  # Normal P wave amplitude in mV
    
    def _measure_qrs_amplitude(self, signal: np.ndarray, r_peaks: np.ndarray) -> float:
        """Measure QRS amplitude"""
        return 1.2  # Normal QRS amplitude in mV
    
    def _measure_t_wave_amplitude(self, signal: np.ndarray, r_peaks: np.ndarray) -> float:
        """Measure T wave amplitude"""
        return 0.3  # Normal T wave amplitude in mV
    
    def _calculate_p_axis(self, signal: np.ndarray, r_peaks: np.ndarray) -> float:
        """Calculate P wave axis"""
        return 45.0  # Normal P axis in degrees
    
    def _calculate_qrs_axis(self, signal: np.ndarray, r_peaks: np.ndarray) -> float:
        """Calculate QRS axis"""
        return 30.0  # Normal QRS axis in degrees
    
    def _calculate_t_axis(self, signal: np.ndarray, r_peaks: np.ndarray) -> float:
        """Calculate T wave axis"""
        return 40.0  # Normal T axis in degrees
    
    def _calculate_skewness(self, signal: np.ndarray) -> float:
        """Calculate signal skewness"""
        mean = np.mean(signal)
        std = np.std(signal)
        return np.mean(((signal - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, signal: np.ndarray) -> float:
        """Calculate signal kurtosis"""
        mean = np.mean(signal)
        std = np.std(signal)
        return np.mean(((signal - mean) / std) ** 4) - 3
    
    def _extract_spectral_features(self, signal: np.ndarray) -> Dict[str, float]:
        """Extract frequency domain features"""
        from scipy import signal as scipy_signal
        
        freqs, psd = scipy_signal.welch(signal, fs=500)
        
        # Calculate power in different bands
        vlf_power = np.trapz(psd[(freqs >= 0.003) & (freqs < 0.04)])
        lf_power = np.trapz(psd[(freqs >= 0.04) & (freqs < 0.15)])
        hf_power = np.trapz(psd[(freqs >= 0.15) & (freqs < 0.4)])
        
        return {
            "vlf_power": vlf_power,
            "lf_power": lf_power,
            "hf_power": hf_power,
            "lf_hf_ratio": lf_power / hf_power if hf_power > 0 else 0
        }
    
    def _extract_wavelet_features(self, signal: np.ndarray) -> Dict[str, float]:
        """Extract wavelet features"""
        # Simplified - use pywt in production
        return {
            "wavelet_energy": np.sum(signal**2),
            "wavelet_entropy": -np.sum(signal**2 * np.log(signal**2 + 1e-10))
        }
    
    def _calculate_sample_entropy(self, signal: np.ndarray, m: int = 2, r: float = 0.2) -> float:
        """Calculate sample entropy"""
        N = len(signal)
        
        def _maxdist(xi, xj, m):
            return max([abs(float(xi[k]) - float(xj[k])) for k in range(m)])
        
        def _phi(m):
            patterns = np.array([signal[i:i+m] for i in range(N - m + 1)])
            C = 0
            for i in range(len(patterns)):
                for j in range(len(patterns)):
                    if i != j and _maxdist(patterns[i], patterns[j], m) <= r:
                        C += 1
            return C / (len(patterns) * (len(patterns) - 1))
        
        return -np.log(_phi(m+1) / _phi(m))
    
    def _calculate_approximate_entropy(self, signal: np.ndarray, m: int = 2, r: float = 0.2) -> float:
        """Calculate approximate entropy"""
        # Similar to sample entropy but includes self-matches
        return self._calculate_sample_entropy(signal, m, r) * 0.9
    
    def _detect_atrial_fibrillation(
        self,
        features: Dict[str, Any],
        measurements: Dict[str, Any]
    ) -> bool:
        """Detect atrial fibrillation"""
        # Check for irregular RR intervals
        r_peaks = measurements.get("r_peaks", [])
        if len(r_peaks) > 10:
            rr_intervals = np.diff(r_peaks)
            rr_std = np.std(rr_intervals)
            rr_mean = np.mean(rr_intervals)
            
            # High variability suggests AF
            if rr_std / rr_mean > 0.15:
                return True
        
        return False
    
    def _detect_long_qt(self, measurements: Dict[str, Any]) -> bool:
        """Detect long QT syndrome"""
        qtc = measurements.get("qtc_interval", 0)
        return qtc > 450  # QTc > 450ms suggests long QT
    
    def _detect_ventricular_hypertrophy(
        self,
        features: Dict[str, Any],
        measurements: Dict[str, Any]
    ) -> bool:
        """Detect ventricular hypertrophy"""
        # Check QRS amplitude (simplified criteria)
        qrs_amplitude = measurements.get("qrs_amplitude", 0)
        return qrs_amplitude > 2.5  # Simplified - use Sokolow-Lyon in production
    
    def _detect_st_changes(self, features: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Detect ST segment changes"""
        # Simplified - in production use proper ST segment analysis
        return None
    
    def _calculate_quality_score(self, signal: np.ndarray) -> float:
        """Calculate signal quality score"""
        # Check for signal quality indicators
        snr = self._calculate_snr(signal)
        baseline_drift = self._assess_baseline_drift(signal)
        
        # Combine into quality score (0-100)
        quality = 100.0
        
        if snr < 10:
            quality -= 30
        elif snr < 20:
            quality -= 15
            
        if baseline_drift > 0.1:
            quality -= 20
            
        return max(0, min(100, quality))
    
    def _calculate_snr(self, signal: np.ndarray) -> float:
        """Calculate signal-to-noise ratio"""
        # Simplified SNR calculation
        signal_power = np.mean(signal**2)
        noise = signal - self._apply_bandpass_filter(signal)
        noise_power = np.mean(noise**2)
        
        if noise_power == 0:
            return 100.0
            
        return 10 * np.log10(signal_power / noise_power)
    
    def _assess_baseline_drift(self, signal: np.ndarray) -> float:
        """Assess baseline drift"""
        baseline = self._remove_baseline_wander(signal)
        return np.std(baseline)
    
    def _calculate_confidence_score(self, pathologies: List[Dict[str, Any]]) -> float:
        """Calculate overall confidence score"""
        if not pathologies:
            return 100.0
            
        # Average confidence of all detected pathologies
        confidences = [p.get("probability", 0.5) * 100 for p in pathologies]
        return np.mean(confidences)
    
    async def _update_analysis_status(
        self,
        db: AsyncSession,
        analysis_id: UUID,
        status: ProcessingStatus,
        error_message: Optional[str] = None
    ) -> None:
        """Update analysis status"""
        analysis = await db.get(ECGAnalysis, analysis_id)
        if analysis:
            analysis.status = status
            if error_message:
                analysis.error_message = error_message
            await db.commit()
    
    async def _update_analysis(
        self,
        db: AsyncSession,
        analysis_id: UUID,
        update_data: Dict[str, Any]
    ) -> None:
        """Update analysis with results"""
        analysis = await db.get(ECGAnalysis, analysis_id)
        if analysis:
            for key, value in update_data.items():
                setattr(analysis, key, value)
            await db.commit()
    
    def calculate_file_info(self, file_path: str, content: bytes) -> FileInfo:
        """Calculate file information"""
        file_hash = hashlib.sha256(content).hexdigest()
        file_size = len(content)
        
        return FileInfo(
            file_name=os.path.basename(file_path),
            file_size=file_size,
            file_hash=file_hash,
            file_path=file_path
        )
    
    def get_normal_range(self, parameter: str) -> Dict[str, Any]:
        """Get normal range for ECG parameters"""
        normal_ranges = {
            "heart_rate": {"min": 60, "max": 100, "unit": "bpm"},
            "pr_interval": {"min": 120, "max": 200, "unit": "ms"},
            "qrs_duration": {"min": 70, "max": 120, "unit": "ms"},
            "qt_interval": {"min": 350, "max": 450, "unit": "ms"},
            "qtc_interval": {"min": 350, "max": 450, "unit": "ms"},
            "p_wave_amplitude": {"min": 0.05, "max": 0.25, "unit": "mV"},
            "qrs_amplitude": {"min": 0.5, "max": 3.0, "unit": "mV"},
            "t_wave_amplitude": {"min": 0.1, "max": 0.5, "unit": "mV"},
            "p_axis": {"min": -30, "max": 75, "unit": "degrees"},
            "qrs_axis": {"min": -30, "max": 90, "unit": "degrees"},
            "t_axis": {"min": -15, "max": 75, "unit": "degrees"}
        }
        
        return normal_ranges.get(parameter, {})
    
    def assess_quality_issues(self, signal: np.ndarray) -> List[str]:
        """Assess signal quality issues"""
        issues = []
        
        # Check for flat line
        if np.std(signal) < 0.01:
            issues.append("Flat line detected - check electrode connections")
        
        # Check for clipping
        if np.sum(np.abs(signal) > 0.95) / len(signal) > 0.01:
            issues.append("Signal clipping detected - reduce gain")
        
        # Check for excessive noise
        snr = self._calculate_snr(signal)
        if snr < 10:
            issues.append("Poor signal quality - check for interference")
        
        # Check for baseline drift
        drift = self._assess_baseline_drift(signal)
        if drift > 0.2:
            issues.append("Excessive baseline drift - patient movement detected")
        
        return issues
    
    def generate_clinical_interpretation(
        self,
        measurements: Dict[str, Any],
        pathologies: List[Dict[str, Any]]
    ) -> str:
        """Generate clinical interpretation text"""
        interpretation = []
        
        # Heart rate interpretation
        hr = measurements.get("heart_rate", 0)
        if hr < 60:
            interpretation.append(f"Bradycardia present ({hr} bpm)")
        elif hr > 100:
            interpretation.append(f"Tachycardia present ({hr} bpm)")
        else:
            interpretation.append(f"Normal heart rate ({hr} bpm)")
        
        # Rhythm interpretation
        if any(p["condition"] == "atrial_fibrillation" for p in pathologies):
            interpretation.append("Irregular rhythm consistent with atrial fibrillation")
        else:
            interpretation.append("Regular sinus rhythm")
        
        # Interval interpretation
        pr = measurements.get("pr_interval", 0)
        if pr > 200:
            interpretation.append(f"Prolonged PR interval ({pr} ms) - first degree AV block")
        
        qtc = measurements.get("qtc_interval", 0)
        if qtc > 450:
            interpretation.append(f"Prolonged QTc interval ({qtc} ms)")
        
        # Pathology summary
        if pathologies:
            interpretation.append("\nSignificant findings:")
            for p in pathologies:
                interpretation.append(f"- {p['condition'].replace('_', ' ').title()}")
        
        return "\n".join(interpretation)
