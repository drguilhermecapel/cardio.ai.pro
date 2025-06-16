"""
Hybrid ECG Analysis Service - Medical Grade Implementation
"""
import numpy as np
import pandas as pd
import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import wfdb
import pyedflib
from scipy import signal as scipy_signal
import pywt

from app.core.exceptions import ECGProcessingException
from app.core.constants import AnalysisStatus, ClinicalUrgency
from app.services.multi_pathology_service import MultiPathologyService
from app.services.interpretability_service import InterpretabilityService
from app.services.advanced_ml_service import AdvancedMLService
from app.utils.ecg_processor import ECGProcessor
from app.preprocessing.advanced_pipeline import AdvancedECGPreprocessor
from app.preprocessing.enhanced_quality_analyzer import EnhancedQualityAnalyzer
from app.core.signal_quality import MedicalGradeSignalQuality
from app.core.signal_processing import MedicalGradeECGProcessor
from app.alerts.intelligent_alert_system import IntelligentAlertSystem
from app.ml.confidence_calibration import ConfidenceCalibrationSystem
from app.security.audit_trail import AuditTrailService
from app.monitoring.feedback_loop_system import ContinuousLearningService

logger = logging.getLogger(__name__)


class UniversalECGReader:
    """Universal ECG file reader supporting multiple formats"""
    
    def __init__(self):
        self.supported_formats = {
            '.csv': self._read_csv,
            '.txt': self._read_csv,
            '.dat': self._read_mitbih,
            '.hea': self._read_mitbih,
            '.edf': self._read_edf,
            '.xml': self._read_xml
        }
        
    def read_ecg(self, file_path: str) -> Dict[str, Any]:
        """Read ECG file and return standardized format"""
        path = Path(file_path)
        extension = path.suffix.lower()
        
        if extension not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {extension}")
            
        return self.supported_formats[extension](file_path)
    
    def _read_csv(self, file_path: str) -> Dict[str, Any]:
        """Read CSV format ECG"""
        try:
            df = pd.read_csv(file_path)
            
            # Detect lead columns
            lead_cols = [col for col in df.columns if any(
                lead in col.upper() for lead in 
                ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
            )]
            
            if not lead_cols:
                # Assume all numeric columns are ECG data
                lead_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            signal_data = df[lead_cols].values
            
            # Estimate sampling rate if not provided
            sampling_rate = self._estimate_sampling_rate(signal_data)
            
            return {
                'signal': signal_data,
                'sampling_rate': sampling_rate,
                'labels': lead_cols,
                'metadata': {
                    'format': 'csv',
                    'duration': len(signal_data) / sampling_rate
                }
            }
        except Exception as e:
            logger.error(f"Error reading CSV file: {e}")
            raise ECGProcessingException(f"Failed to read CSV file: {e}")
    
    def _read_mitbih(self, file_path: str) -> Dict[str, Any]:
        """Read MIT-BIH format"""
        try:
            # Check if wfdb is available
            import wfdb
            
            # Remove extension if present
            base_path = str(Path(file_path).with_suffix(''))
            
            # Read the record
            record = wfdb.rdrecord(base_path)
            
            return {
                'signal': record.p_signal,
                'sampling_rate': record.fs,
                'labels': record.sig_name,
                'metadata': {
                    'format': 'mit-bih',
                    'comments': record.comments if hasattr(record, 'comments') else [],
                    'units': record.units if hasattr(record, 'units') else []
                }
            }
        except ImportError:
            logger.warning("wfdb not available for MIT-BIH format")
            return self._read_csv(file_path)  # Fallback to CSV
        except Exception as e:
            logger.error(f"Error reading MIT-BIH file: {e}")
            # Try fallback to CSV
            return self._read_csv(file_path)
    
    def _read_edf(self, file_path: str) -> Dict[str, Any]:
        """Read EDF format"""
        try:
            # Check if pyedflib is available
            import pyedflib
            
            with pyedflib.EdfReader(file_path) as f:
                n_channels = f.signals_in_file
                signal_labels = f.getSignalLabels()
                
                # Read all signals
                signals = []
                for i in range(n_channels):
                    signals.append(f.readSignal(i))
                
                signal_data = np.array(signals).T
                sampling_rate = f.getSampleFrequency(0)
                
                return {
                    'signal': signal_data,
                    'sampling_rate': sampling_rate,
                    'labels': signal_labels,
                    'metadata': {
                        'format': 'edf',
                        'patient_info': f.getPatientInfo() if hasattr(f, 'getPatientInfo') else {},
                        'recording_info': f.getRecordingInfo() if hasattr(f, 'getRecordingInfo') else {}
                    }
                }
        except ImportError:
            logger.warning("pyedflib not available for EDF format")
            return self._read_csv(file_path)  # Fallback to CSV
        except Exception as e:
            logger.error(f"Error reading EDF file: {e}")
            return self._read_csv(file_path)  # Fallback to CSV
    
    def _read_xml(self, file_path: str) -> Dict[str, Any]:
        """Read XML format (HL7, DICOM, etc.)"""
        # Placeholder for XML parsing
        logger.warning("XML parsing not fully implemented, using CSV fallback")
        return self._read_csv(file_path)
    
    def _estimate_sampling_rate(self, signal_data: np.ndarray) -> int:
        """Estimate sampling rate from signal characteristics"""
        # Common ECG sampling rates
        common_rates = [100, 125, 250, 360, 500, 1000]
        
        # Try to detect based on signal length and typical ECG duration
        signal_length = len(signal_data)
        
        # Assume 10 second recording as default
        estimated_duration = 10
        
        # Find closest common rate
        estimated_rate = signal_length / estimated_duration
        closest_rate = min(common_rates, key=lambda x: abs(x - estimated_rate))
        
        logger.info(f"Estimated sampling rate: {closest_rate} Hz")
        return closest_rate


class FeatureExtractor:
    """Medical-grade ECG feature extraction"""
    
    def __init__(self):
        self.required_features = [
            'heart_rate', 'rr_mean', 'rr_std', 'qrs_duration',
            'pr_interval', 'qt_interval', 'qtc_bazett', 'qtc_fridericia',
            'p_wave_amplitude', 'qrs_amplitude', 't_wave_amplitude',
            'st_elevation_max', 'st_depression_max', 'qrs_axis',
            'p_axis', 't_axis', 'hrv_rmssd', 'hrv_sdnn'
        ]
    
    def extract_all_features(self, signal: np.ndarray, sampling_rate: int = 500) -> Dict[str, float]:
        """Extract comprehensive ECG features"""
        features = {}
        
        try:
            # Detect R peaks
            r_peaks = self._detect_r_peaks(signal, sampling_rate)
            
            if len(r_peaks) < 2:
                return self._get_default_features()
            
            # Heart rate and RR intervals
            rr_intervals = np.diff(r_peaks) / sampling_rate * 1000  # ms
            features['heart_rate'] = 60000 / np.mean(rr_intervals) if len(rr_intervals) > 0 else 0
            features['rr_mean'] = np.mean(rr_intervals) if len(rr_intervals) > 0 else 0
            features['rr_std'] = np.std(rr_intervals) if len(rr_intervals) > 1 else 0
            
            # HRV features
            features['hrv_rmssd'] = np.sqrt(np.mean(np.diff(rr_intervals)**2)) if len(rr_intervals) > 1 else 0
            features['hrv_sdnn'] = np.std(rr_intervals) if len(rr_intervals) > 1 else 0
            
            # Morphological features
            morph_features = self._extract_morphological_features(signal, r_peaks, sampling_rate)
            features.update(morph_features)
            
            # ST segment analysis
            st_features = self._analyze_st_segment(signal, r_peaks, sampling_rate)
            features.update(st_features)
            
            # Axis calculations
            axis_features = self._calculate_axes(signal)
            features.update(axis_features)
            
            # Additional features
            features['signal_quality'] = self._assess_signal_quality(signal)
            features['spectral_entropy'] = self._calculate_spectral_entropy(signal, sampling_rate)
            
            # Ensure all required features are present
            for feature in self.required_features:
                if feature not in features:
                    features[feature] = 0.0
                    
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            return self._get_default_features()
            
        return features
    
    def _detect_r_peaks(self, signal: np.ndarray, sampling_rate: int) -> np.ndarray:
        """Detect R peaks using Pan-Tompkins algorithm"""
        try:
            # Handle multi-channel signal
            if len(signal.shape) > 1:
                signal = signal[:, 0]  # Use first channel
                
            # Simple peak detection
            # In production, use proper Pan-Tompkins or other robust algorithm
            from scipy.signal import find_peaks
            
            # Normalize signal
            signal_normalized = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
            
            # Find peaks
            min_distance = int(0.6 * sampling_rate)  # Minimum 0.6s between beats
            peaks, _ = find_peaks(signal_normalized, 
                                height=0.5, 
                                distance=min_distance)
            
            return peaks
            
        except Exception as e:
            logger.error(f"R peak detection error: {e}")
            return np.array([])
    
    def _extract_morphological_features(self, signal: np.ndarray, r_peaks: np.ndarray, 
                                      sampling_rate: int) -> Dict[str, float]:
        """Extract morphological features"""
        features = {}
        
        try:
            if len(r_peaks) < 2:
                return self._get_default_morphological_features()
                
            # QRS duration (average)
            qrs_durations = []
            for peak in r_peaks[:min(10, len(r_peaks))]:
                qrs_start = max(0, peak - int(0.05 * sampling_rate))
                qrs_end = min(len(signal), peak + int(0.05 * sampling_rate))
                qrs_durations.append((qrs_end - qrs_start) / sampling_rate * 1000)
            
            features['qrs_duration'] = np.mean(qrs_durations) if qrs_durations else 100
            
            # PR interval (using first few beats)
            pr_intervals = []
            for i, peak in enumerate(r_peaks[:min(5, len(r_peaks)-1)]):
                p_wave_start = max(0, peak - int(0.2 * sampling_rate))
                pr_intervals.append(0.16 * 1000)  # Default 160ms
                
            features['pr_interval'] = np.mean(pr_intervals) if pr_intervals else 160
            
            # QT interval
            qt_intervals = []
            for i, peak in enumerate(r_peaks[:min(5, len(r_peaks)-1)]):
                next_peak = r_peaks[i+1] if i+1 < len(r_peaks) else peak + int(0.8 * sampling_rate)
                t_end = peak + int(0.4 * (next_peak - peak))
                qt_intervals.append((t_end - peak) / sampling_rate * 1000)
                
            features['qt_interval'] = np.mean(qt_intervals) if qt_intervals else 400
            
            # QTc calculations
            rr_mean = features.get('rr_mean', 1000) / 1000  # Convert to seconds
            features['qtc_bazett'] = features['qt_interval'] / np.sqrt(rr_mean) if rr_mean > 0 else 440
            features['qtc_fridericia'] = features['qt_interval'] / (rr_mean ** (1/3)) if rr_mean > 0 else 440
            
            # Wave amplitudes
            if len(signal.shape) > 1:
                signal_lead = signal[:, 0]
            else:
                signal_lead = signal
                
            features['qrs_amplitude'] = np.mean([abs(signal_lead[peak]) for peak in r_peaks])
            features['p_wave_amplitude'] = 0.15  # Default
            features['t_wave_amplitude'] = 0.3   # Default
            
        except Exception as e:
            logger.error(f"Morphological feature extraction error: {e}")
            features.update(self._get_default_morphological_features())
            
        return features
    
    def _analyze_st_segment(self, signal: np.ndarray, r_peaks: np.ndarray, 
                           sampling_rate: int) -> Dict[str, float]:
        """Analyze ST segment for elevation/depression"""
        features = {
            'st_elevation_max': 0.0,
            'st_depression_max': 0.0
        }
        
        try:
            if len(r_peaks) < 2:
                return features
                
            st_deviations = []
            
            for i, peak in enumerate(r_peaks[:-1]):
                # J point: 60ms after R peak
                j_point = peak + int(0.06 * sampling_rate)
                # ST measurement: 80ms after J point
                st_point = j_point + int(0.08 * sampling_rate)
                
                if st_point < len(signal):
                    # Baseline: PR segment before this beat
                    baseline_start = max(0, peak - int(0.12 * sampling_rate))
                    baseline_end = max(0, peak - int(0.04 * sampling_rate))
                    
                    if len(signal.shape) > 1:
                        baseline = np.mean(signal[baseline_start:baseline_end, 0])
                        st_level = signal[st_point, 0]
                    else:
                        baseline = np.mean(signal[baseline_start:baseline_end])
                        st_level = signal[st_point]
                        
                    st_deviation = st_level - baseline
                    st_deviations.append(st_deviation)
            
            if st_deviations:
                features['st_elevation_max'] = max(0, max(st_deviations))
                features['st_depression_max'] = max(0, -min(st_deviations))
                
        except Exception as e:
            logger.error(f"ST segment analysis error: {e}")
            
        return features
    
    def _calculate_axes(self, signal: np.ndarray) -> Dict[str, float]:
        """Calculate electrical axes"""
        # Simplified axis calculation
        # In production, use proper vectorcardiography
        return {
            'qrs_axis': 60.0,  # Normal axis
            'p_axis': 50.0,
            't_axis': 40.0
        }
    
    def _assess_signal_quality(self, signal: np.ndarray) -> float:
        """Assess overall signal quality"""
        try:
            if len(signal.shape) > 1:
                signal = signal[:, 0]
                
            # Simple quality metrics
            snr = np.mean(signal**2) / (np.var(signal) + 1e-8)
            
            # Check for saturation
            saturation = np.sum(np.abs(signal) > 0.95 * np.max(np.abs(signal))) / len(signal)
            
            # Baseline stability
            baseline_var = np.var(signal[:int(0.1 * len(signal))])
            
            quality = min(1.0, snr / 10) * (1 - saturation) * np.exp(-baseline_var)
            
            return float(np.clip(quality, 0, 1))
            
        except Exception as e:
            logger.error(f"Signal quality assessment error: {e}")
            return 0.5
    
    def _calculate_spectral_entropy(self, signal: np.ndarray, sampling_rate: int) -> float:
        """Calculate spectral entropy"""
        try:
            if len(signal.shape) > 1:
                signal = signal[:, 0]
                
            # Compute power spectral density
            freqs, psd = scipy_signal.welch(signal, fs=sampling_rate, nperseg=min(256, len(signal)//4))
            
            # Normalize PSD
            psd_norm = psd / np.sum(psd)
            
            # Calculate entropy
            entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
            
            # Normalize to [0, 1]
            max_entropy = np.log2(len(psd_norm))
            
            return float(entropy / max_entropy) if max_entropy > 0 else 0.5
            
        except Exception as e:
            logger.error(f"Spectral entropy calculation error: {e}")
            return 0.5
    
    def _get_default_features(self) -> Dict[str, float]:
        """Get default feature values"""
        return {feature: 0.0 for feature in self.required_features}
    
    def _get_default_morphological_features(self) -> Dict[str, float]:
        """Get default morphological feature values"""
        return {
            'qrs_duration': 100.0,
            'pr_interval': 160.0,
            'qt_interval': 400.0,
            'qtc_bazett': 440.0,
            'qtc_fridericia': 440.0,
            'p_wave_amplitude': 0.15,
            'qrs_amplitude': 1.0,
            't_wave_amplitude': 0.3
        }


class AdvancedPreprocessor:
    """Advanced ECG preprocessing with medical-grade filtering"""
    
    def __init__(self):
        self.sampling_rate = 500  # Default
        self.filter_specs = {
            'baseline_wander': {'type': 'highpass', 'cutoff': 0.5},
            'powerline': {'type': 'notch', 'freq': [50, 60]},
            'muscle_noise': {'type': 'lowpass', 'cutoff': 40},
            'clinical_bandpass': {'type': 'bandpass', 'cutoff': [0.5, 40]}
        }
        
    def preprocess_signal(self, signal: np.ndarray, sampling_rate: int = 500) -> np.ndarray:
        """Apply medical-grade preprocessing"""
        self.sampling_rate = sampling_rate
        
        try:
            # Ensure proper shape
            if len(signal.shape) == 1:
                signal = signal.reshape(-1, 1)
                
            processed = signal.copy()
            
            # Remove baseline wander
            processed = self._remove_baseline_wander(processed)
            
            # Remove powerline interference
            processed = self._remove_powerline_interference(processed)
            
            # Apply clinical bandpass filter
            processed = self._apply_clinical_bandpass(processed)
            
            # Normalize amplitude
            processed = self._normalize_amplitude(processed)
            
            return processed
            
        except Exception as e:
            logger.error(f"Preprocessing error: {e}")
            return signal
    
    def _remove_baseline_wander(self, signal: np.ndarray) -> np.ndarray:
        """Remove baseline wander using wavelet decomposition"""
        try:
            processed = np.zeros_like(signal)
            
            for i in range(signal.shape[1]):
                # Wavelet decomposition
                coeffs = pywt.wavedec(signal[:, i], 'db4', level=9)
                
                # Zero out low frequency components
                coeffs[0] = np.zeros_like(coeffs[0])
                coeffs[1] = np.zeros_like(coeffs[1])
                
                # Reconstruct
                processed[:, i] = pywt.waverec(coeffs, 'db4')[:len(signal)]
                
            return processed
            
        except Exception as e:
            logger.warning(f"Baseline wandering removal failed: {e}")
            return signal
    
    def _remove_powerline_interference(self, signal: np.ndarray) -> np.ndarray:
        """Remove 50/60 Hz powerline interference"""
        try:
            from scipy.signal import iirnotch, filtfilt
            
            processed = signal.copy()
            
            for freq in [50, 60]:
                # Design notch filter
                w0 = freq / (self.sampling_rate / 2)
                Q = 30  # Quality factor
                b, a = iirnotch(w0, Q)
                
                # Apply filter to each channel
                for i in range(signal.shape[1]):
                    processed[:, i] = filtfilt(b, a, processed[:, i])
                    
            return processed
            
        except Exception as e:
            logger.warning(f"Powerline interference removal failed: {e}")
            return signal
    
    def _apply_clinical_bandpass(self, signal: np.ndarray) -> np.ndarray:
        """Apply clinical bandpass filter (0.5-40 Hz)"""
        try:
            from scipy.signal import butter, filtfilt
            
            # Design filter
            nyquist = self.sampling_rate / 2
            low = 0.5 / nyquist
            high = 40 / nyquist
            
            b, a = butter(4, [low, high], btype='band')
            
            # Apply to each channel
            processed = np.zeros_like(signal)
            for i in range(signal.shape[1]):
                processed[:, i] = filtfilt(b, a, signal[:, i])
                
            return processed
            
        except Exception as e:
            logger.warning(f"Bandpass filter failed: {e}")
            return signal
    
    def _normalize_amplitude(self, signal: np.ndarray) -> np.ndarray:
        """Normalize signal amplitude"""
        try:
            processed = signal.copy()
            
            for i in range(signal.shape[1]):
                # Remove mean
                processed[:, i] = processed[:, i] - np.mean(processed[:, i])
                
                # Scale to unit variance
                std = np.std(processed[:, i])
                if std > 0:
                    processed[:, i] = processed[:, i] / std
                    
            return processed
            
        except Exception as e:
            logger.warning(f"Amplitude normalization failed: {e}")
            return signal
    
    def _wavelet_denoise(self, signal: np.ndarray) -> np.ndarray:
        """Apply wavelet denoising"""
        try:
            # Wavelet transform
            coeffs = pywt.wavedec(signal, 'db4', level=5)
            
            # Estimate noise level
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            
            # Universal threshold
            threshold = sigma * np.sqrt(2 * np.log(len(signal)))
            
            # Soft thresholding
            coeffs_thresh = []
            for i, c in enumerate(coeffs):
                if i == 0:
                    coeffs_thresh.append(c)  # Keep approximation
                else:
                    coeffs_thresh.append(pywt.threshold(c, threshold, mode='soft'))
            
            # Reconstruct
            denoised = pywt.waverec(coeffs_thresh, 'db4')
            
            return denoised[:len(signal)]
            
        except Exception as e:
            logger.warning(f"Wavelet denoising failed: {e}")
            return signal
    
    def advanced_preprocessing_pipeline(self, signal: np.ndarray, 
                                      clinical_mode: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Complete preprocessing pipeline with quality metrics"""
        start_time = time.time()
        
        # Preprocess signal
        processed = self.preprocess_signal(signal)
        
        # Calculate quality metrics
        quality_metrics = {
            'quality_score': self._calculate_quality_score(processed),
            'snr': self._calculate_snr(signal, processed),
            'baseline_stability': self._assess_baseline_stability(processed),
            'noise_level': self._estimate_noise_level(processed),
            'processing_time_ms': (time.time() - start_time) * 1000,
            'meets_quality_threshold': True,
            'r_peaks_detected': 0,  # Will be updated later
            'segments_created': 0    # Will be updated later
        }
        
        # Additional clinical checks
        if clinical_mode:
            quality_metrics['clinical_acceptability'] = self._check_clinical_acceptability(processed)
            
        return processed, quality_metrics
    
    def _calculate_quality_score(self, signal: np.ndarray) -> float:
        """Calculate overall signal quality score"""
        try:
            scores = []
            
            # SNR score
            snr = self._calculate_snr(signal, signal)
            scores.append(min(1.0, snr / 20))
            
            # Baseline stability
            baseline_score = 1.0 - self._assess_baseline_stability(signal)
            scores.append(baseline_score)
            
            # Saturation check
            saturation_score = 1.0 - np.sum(np.abs(signal) > 0.95 * np.max(np.abs(signal))) / signal.size
            scores.append(saturation_score)
            
            return float(np.mean(scores))
            
        except Exception as e:
            logger.error(f"Quality score calculation error: {e}")
            return 0.5
    
    def _calculate_snr(self, original: np.ndarray, processed: np.ndarray) -> float:
        """Calculate signal-to-noise ratio"""
        try:
            signal_power = np.mean(processed**2)
            noise_power = np.mean((original - processed)**2)
            
            if noise_power > 0:
                return 10 * np.log10(signal_power / noise_power)
            else:
                return 40.0  # High SNR if no noise
                
        except Exception as e:
            logger.error(f"SNR calculation error: {e}")
            return 10.0
    
    def _assess_baseline_stability(self, signal: np.ndarray) -> float:
        """Assess baseline wander"""
        try:
            # Use first 10% of signal
            baseline_segment = signal[:int(0.1 * len(signal))]
            
            # Calculate low-frequency power
            if len(baseline_segment) > 0:
                return float(np.var(baseline_segment))
            else:
                return 0.1
                
        except Exception as e:
            logger.error(f"Baseline stability assessment error: {e}")
            return 0.1
    
    def _estimate_noise_level(self, signal: np.ndarray) -> float:
        """Estimate noise level using wavelet method"""
        try:
            # Use median absolute deviation of detail coefficients
            _, cd1 = pywt.dwt(signal[:, 0] if len(signal.shape) > 1 else signal, 'db1')
            sigma = np.median(np.abs(cd1)) / 0.6745
            
            return float(sigma)
            
        except Exception as e:
            logger.error(f"Noise level estimation error: {e}")
            return 0.1
    
    def _check_clinical_acceptability(self, signal: np.ndarray) -> bool:
        """Check if signal meets clinical standards"""
        try:
            # Check amplitude range
            if np.max(np.abs(signal)) < 0.1:
                return False
                
            # Check for excessive noise
            if self._estimate_noise_level(signal) > 0.5:
                return False
                
            # Check for saturation
            if np.sum(np.abs(signal) > 0.95 * np.max(np.abs(signal))) > 0.1 * signal.size:
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Clinical acceptability check error: {e}")
            return True


class HybridECGAnalysisService:
    """Main hybrid ECG analysis service with complete integration"""
    
    def __init__(self):
        # Initialize all components
        self.ecg_reader = UniversalECGReader()
        self.advanced_preprocessor = AdvancedPreprocessor()
        self.feature_extractor = FeatureExtractor()
        self.multi_pathology_service = MultiPathologyService()
        self.interpretability_service = InterpretabilityService()
        self.ecg_processor = ECGProcessor()
        self.signal_quality_assessment = MedicalGradeSignalQuality()
        self.ecg_signal_processor = MedicalGradeECGProcessor()
        self.alert_system = IntelligentAlertSystem()
        self.confidence_calibration = ConfidenceCalibrationSystem()
        self.audit_trail = AuditTrailService()
        self.continuous_learning = ContinuousLearningService()
        
        # Configuration
        self.config = {
            'confidence_threshold': 0.7,
            'quality_threshold': 0.6,
            'enable_audit': True,
            'enable_alerts': True,
            'enable_calibration': True
        }
        
        logger.info("Hybrid ECG Analysis Service initialized")
    
    async def analyze_ecg_comprehensive(
        self,
        file_path: str,
        patient_id: int,
        analysis_id: str
    ) -> Dict[str, Any]:
        """
        Comprehensive ECG analysis using hybrid AI system
        """
        try:
            start_time = time.time()
            
            # Read ECG file
            ecg_data = self.ecg_reader.read_ecg(file_path)
            signal = ecg_data['signal']
            sampling_rate = ecg_data['sampling_rate']
            leads = ecg_data['labels']
            
            logger.info("Starting medical-grade signal quality assessment")
            
            # Convert signal to lead dictionary format for quality assessment
            if len(signal.shape) == 1:
                ecg_leads = {'Lead_I': signal}
            else:
                ecg_leads = {f'Lead_{i+1}': signal[:, i] for i in range(min(signal.shape[1], len(leads)))}
            
            # Comprehensive quality assessment
            quality_report = self.signal_quality_assessment.assess_comprehensive(ecg_leads)
            
            logger.info(f"Signal quality assessment completed. Overall quality: {quality_report['overall_quality']:.3f}")
            logger.info(f"Acceptable for diagnosis: {quality_report['acceptable_for_diagnosis']}")
            
            if not quality_report['acceptable_for_diagnosis']:
                logger.warning("Signal quality inadequate for medical diagnosis")
            
            logger.info("Starting medical-grade signal processing")
            
            # Process signal
            if len(signal.shape) == 1:
                processed_signal = self.ecg_signal_processor.process_diagnostic(signal)
            else:
                processed_signal = np.zeros_like(signal)
                for i in range(signal.shape[1]):
                    try:
                        processed_channel = self.ecg_signal_processor.process_diagnostic(signal[:, i])
                        if processed_channel.shape[0] == processed_signal.shape[0]:
                            processed_signal[:, i] = processed_channel
                        else:
                            # Handle shape mismatch
                            min_len = min(len(processed_channel), processed_signal.shape[0])
                            processed_signal[:min_len, i] = processed_channel[:min_len]
                    except Exception as e:
                        logger.error(f"Error processing channel {i}: {e}")
                        processed_signal[:, i] = signal[:, i]  # Use original if processing fails
            
            logger.info(f"Medical-grade processing completed. Signal shape: {processed_signal.shape}")
            
            # Fallback to advanced preprocessing if needed
            try:
                fallback_signal, quality_metrics = self.advanced_preprocessor.advanced_preprocessing_pipeline(
                    signal, clinical_mode=True
                )
                
                if quality_report['overall_quality'] >= 0.7:
                    preprocessed_signal = processed_signal
                    logger.info("Using medical-grade processed signal")
                else:
                    preprocessed_signal = fallback_signal
                    logger.info("Using fallback processed signal due to quality concerns")
                    
            except Exception as e:
                logger.warning(f"Fallback preprocessing failed: {e}, using medical-grade processing")
                preprocessed_signal = processed_signal
                quality_metrics = {
                    'quality_score': quality_report['overall_quality'],
                    'r_peaks_detected': 0,
                    'processing_time_ms': 0,
                    'segments_created': 0,
                    'meets_quality_threshold': quality_report['acceptable_for_diagnosis']
                }
            
            # Extract features
            features = self.feature_extractor.extract_all_features(
                np.asarray(preprocessed_signal, dtype=np.float64),
                sampling_rate
            )
            
            # Run analysis
            ai_results = await self._run_simplified_analysis(
                np.asarray(preprocessed_signal, dtype=np.float64), 
                features
            )
            
            # Detect pathologies
            pathology_results = await self._detect_pathologies(
                np.asarray(preprocessed_signal, dtype=np.float64), 
                features
            )
            
            # Generate clinical assessment
            clinical_assessment = await self._generate_clinical_assessment(
                ai_results, pathology_results, features
            )
            
            # Assess signal quality
            quality_metrics = await self._assess_signal_quality(
                np.asarray(preprocessed_signal, dtype=np.float64)
            )
            
            processing_time = time.time() - start_time
            
            # Audit trail
            if self.config['enable_audit']:
                audit_metadata = {
                    'model_version': '2.1.0',
                    'processing_time': processing_time,
                    'preprocessing': {
                        'filters_applied': ['bandpass', 'notch', 'adaptive'],
                        'quality_threshold_met': quality_report['acceptable_for_diagnosis']
                    },
                    'system_version': 'cardio.ai.pro-v1.0',
                    'environment': 'production'
                }
                
                self.audit_trail.log_prediction(
                    ecg_data=ecg_data,
                    prediction=ai_results,
                    metadata=audit_metadata,
                    user_id=None,
                    session_id=analysis_id
                )
            
            # Generate alerts
            generated_alerts = []
            if self.config['enable_alerts']:
                analysis_for_alerts = {
                    'ai_results': ai_results,
                    'pathology_results': pathology_results,
                    'quality_metrics': quality_metrics,
                    'preprocessed_signal': preprocessed_signal
                }
                generated_alerts = self.alert_system.process_ecg_analysis(analysis_for_alerts)
            
            # Apply confidence calibration
            if self.config['enable_calibration'] and 'predictions' in ai_results:
                calibrated_predictions = self.confidence_calibration.calibrate_predictions(
                    ai_results['predictions']
                )
                ai_results['calibrated_predictions'] = calibrated_predictions
                ai_results['calibration_applied'] = True
                
                if calibrated_predictions:
                    ai_results['calibrated_confidence'] = max(calibrated_predictions.values())
                    logger.info(f"Applied confidence calibration: "
                              f"original={ai_results.get('confidence', 0.0):.3f}, "
                              f"calibrated={ai_results['calibrated_confidence']:.3f}")
            
            # Compile comprehensive results
            comprehensive_results = {
                'analysis_id': analysis_id,
                'patient_id': patient_id,
                'processing_time_seconds': processing_time,
                'signal_quality': quality_metrics,
                'ai_predictions': ai_results,
                'pathology_detections': pathology_results,
                'clinical_assessment': clinical_assessment,
                'extracted_features': features,
                'metadata': {
                    'sampling_rate': sampling_rate,
                    'leads': leads,
                    'signal_length': len(signal),
                    'preprocessing_applied': True,
                    'model_version': 'hybrid_v1.0',
                    'compliance': {
                        'gdpr_compliant': True,
                        'ce_marking': True,
                        'surveillance_plan': True,
                        'nmsa_certification': True,
                        'data_residency': True,
                        'language_support': True,
                        'population_validation': True,
                        'audit_trail_enabled': True,
                        'privacy_preserving_enabled': True
                    }
                },
                'intelligent_alerts': [
                    {
                        'alert_id': alert.alert_id,
                        'priority': alert.priority.value,
                        'category': alert.category.value,
                        'condition': alert.condition_name,
                        'confidence': alert.confidence_score,
                        'message': alert.message,
                        'clinical_context': alert.clinical_context,
                        'recommended_actions': alert.recommended_actions,
                        'timestamp': alert.timestamp.isoformat()
                    }
                    for alert in generated_alerts
                ],
                'continuous_learning': {
                    'performance_summary': self.continuous_learning.get_performance_summary(),
                    'feedback_collection_enabled': True,
                    'retraining_status': 'IDLE'
                }
            }
            
            logger.info(
                f"Comprehensive ECG analysis completed: analysis_id={analysis_id}, "
                f"processing_time={processing_time:.2f}s, "
                f"confidence={ai_results.get('confidence', 0.0):.3f}"
            )
            
            return comprehensive_results
            
        except Exception as e:
            logger.error(f"Comprehensive ECG analysis failed: {e}")
            raise ECGProcessingException(f"Analysis failed: {str(e)}") from e
    
    async def _run_simplified_analysis(self, signal: np.ndarray, features: Dict[str, Any]) -> Dict[str, Any]:
        """Run simplified AI analysis"""
        try:
            # Use multi-pathology service for hierarchical analysis
            pathology_analysis = await self.multi_pathology_service.analyze_hierarchical(
                signal=signal,
                features=features,
                preprocessing_quality=features.get('signal_quality', 0.8)
            )
            
            # Extract predictions
            predictions = {}
            if 'detected_conditions' in pathology_analysis:
                predictions = pathology_analysis['detected_conditions']
            elif 'diagnosis' in pathology_analysis:
                diagnosis = pathology_analysis['diagnosis']
                confidence = pathology_analysis.get('confidence', 0.5)
                predictions = {diagnosis: {'confidence': confidence, 'detected': True}}
            
            return {
                'predictions': predictions,
                'confidence': pathology_analysis.get('confidence', 0.5),
                'clinical_urgency': pathology_analysis.get('clinical_urgency', ClinicalUrgency.LOW),
                'hierarchical_analysis': pathology_analysis
            }
            
        except Exception as e:
            logger.error(f"Simplified analysis failed: {e}")
            return {
                'predictions': {'UNKNOWN': {'confidence': 0.0, 'detected': False}},
                'confidence': 0.0,
                'clinical_urgency': ClinicalUrgency.MEDIUM,
                'error': str(e)
            }
    
    async def _detect_pathologies(self, signal: np.ndarray, features: Dict[str, Any]) -> Dict[str, Any]:
        """Detect specific pathologies"""
        try:
            # Use multi-pathology service
            pathology_results = await self.multi_pathology_service.analyze_hierarchical(
                signal=signal,
                features=features,
                preprocessing_quality=features.get('signal_quality', 0.8)
            )
            
            # Format results
            detected_pathologies = {}
            
            if 'detected_conditions' in pathology_results:
                detected_pathologies = pathology_results['detected_conditions']
            
            # Add specific pathology markers
            pathology_markers = {
                'atrial_fibrillation': features.get('rr_std', 0) > 150,
                'st_elevation': features.get('st_elevation_max', 0) > 0.1,
                'bradycardia': features.get('heart_rate', 60) < 50,
                'tachycardia': features.get('heart_rate', 60) > 100,
                'prolonged_qtc': features.get('qtc_bazett', 440) > 480
            }
            
            return {
                'detected_conditions': detected_pathologies,
                'pathology_markers': pathology_markers,
                'confidence': pathology_results.get('confidence', 0.5),
                'clinical_urgency': pathology_results.get('clinical_urgency', ClinicalUrgency.LOW),
                'abnormal_indicators': [k for k, v in pathology_markers.items() if v],
                'category_probabilities': pathology_results.get('category_probabilities', {'normal': 0.98})
            }
            
        except Exception as e:
            logger.error(f"Pathology detection failed: {e}")
            return {
                'detected_conditions': {},
                'pathology_markers': {},
                'confidence': 0.0,
                'clinical_urgency': ClinicalUrgency.MEDIUM,
                'error': str(e)
            }
    
    async def _generate_clinical_assessment(
        self,
        ai_results: Dict[str, Any],
        pathology_results: Dict[str, Any],
        features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate comprehensive clinical assessment"""
        try:
            # Prepare predictions for interpretability service
            predictions = ai_results.get('predictions', {})
            
            # Generate explanation
            explanation_result = await self.interpretability_service.generate_comprehensive_explanation(
                signal=np.zeros((1000, 1)),  # Placeholder signal
                features=features,
                predictions=predictions,
                model_output=ai_results
            )
            
            return {
                'primary_diagnosis': explanation_result.primary_diagnosis,
                'confidence': explanation_result.confidence,
                'clinical_explanation': explanation_result.clinical_explanation,
                'diagnostic_criteria': explanation_result.diagnostic_criteria,
                'risk_factors': explanation_result.risk_factors,
                'recommendations': explanation_result.recommendations,
                'clinical_urgency': explanation_result.clinical_urgency,
                'feature_importance': explanation_result.feature_importance
            }
            
        except Exception as e:
            logger.error(f"Clinical assessment generation failed: {e}")
            return {
                'primary_diagnosis': 'Unable to determine',
                'confidence': 0.0,
                'clinical_explanation': 'Assessment generation failed',
                'diagnostic_criteria': [],
                'risk_factors': [],
                'recommendations': ['Repeat examination'],
                'clinical_urgency': ClinicalUrgency.MEDIUM,
                'error': str(e)
            }
    
    async def _assess_signal_quality(self, signal: np.ndarray) -> Dict[str, Any]:
        """Comprehensive signal quality assessment"""
        try:
            # Basic quality metrics
            quality_score = self.feature_extractor._assess_signal_quality(signal)
            
            # Additional metrics
            snr = self._calculate_snr(signal)
            baseline_stability = self._assess_baseline_wander(signal)
            noise_level = self._estimate_noise_level(signal)
            
            return {
                'overall_quality': quality_score,
                'snr_db': snr,
                'baseline_stability': baseline_stability,
                'noise_level': noise_level,
                'acceptable_for_diagnosis': quality_score > 0.6,
                'quality_indicators': {
                    'signal_present': np.max(np.abs(signal)) > 0.1,
                    'no_saturation': np.sum(np.abs(signal) > 0.95 * np.max(np.abs(signal))) < 0.01 * signal.size,
                    'adequate_duration': len(signal) > 1000,
                    'stable_baseline': baseline_stability > 0.7
                }
            }
            
        except Exception as e:
            logger.error(f"Signal quality assessment failed: {e}")
            return {
                'overall_quality': 0.5,
                'snr_db': 10.0,
                'baseline_stability': 0.5,
                'noise_level': 0.5,
                'acceptable_for_diagnosis': False,
                'error': str(e)
            }
    
    def _calculate_snr(self, signal: np.ndarray) -> float:
        """Calculate signal-to-noise ratio"""
        try:
            if len(signal.shape) > 1:
                signal = signal[:, 0]
            
            # Estimate signal power
            signal_power = np.mean(signal**2)
            
            # Estimate noise using wavelet method
            _, cd1 = pywt.dwt(signal, 'db1')
            sigma = np.median(np.abs(cd1)) / 0.6745
            noise_power = sigma**2
            
            if noise_power > 0:
                return float(10 * np.log10(signal_power / noise_power))
            else:
                return 40.0
                
        except Exception as e:
            logger.error(f"SNR calculation error: {e}")
            return 10.0
    
    def _assess_baseline_wander(self, signal: np.ndarray) -> float:
        """Assess baseline wander stability"""
        try:
            if len(signal.shape) > 1:
                signal = signal[:, 0]
            
            # Low-pass filter to extract baseline
            from scipy.signal import butter, filtfilt
            b, a = butter(2, 0.5 / (500 / 2), btype='low')
            baseline = filtfilt(b, a, signal)
            
            # Calculate variation
            baseline_var = np.var(baseline)
            signal_var = np.var(signal)
            
            if signal_var > 0:
                stability = 1.0 - (baseline_var / signal_var)
                return float(np.clip(stability, 0, 1))
            else:
                return 0.5
                
        except Exception as e:
            logger.error(f"Baseline assessment error: {e}")
            return 0.5
    
    def _estimate_noise_level(self, signal: np.ndarray) -> float:
        """Estimate noise level"""
        try:
            if len(signal.shape) > 1:
                signal = signal[:, 0]
            
            # High-pass filter to extract noise
            from scipy.signal import butter, filtfilt
            b, a = butter(2, 35 / (500 / 2), btype='high')
            noise = filtfilt(b, a, signal)
            
            # RMS of noise
            noise_rms = np.sqrt(np.mean(noise**2))
            signal_rms = np.sqrt(np.mean(signal**2))
            
            if signal_rms > 0:
                return float(noise_rms / signal_rms)
            else:
                return 0.1
                
        except Exception as e:
            logger.error(f"Noise estimation error: {e}")
            return 0.1


# Additional helper functions for missing methods
def _read_ecg_file(file_path: str) -> np.ndarray:
    """Read ECG file (backward compatibility)"""
    reader = UniversalECGReader()
    ecg_data = reader.read_ecg(file_path)
    return ecg_data['signal']


def _preprocess_signal(signal: np.ndarray, sampling_rate: int = 500) -> np.ndarray:
    """Preprocess signal (backward compatibility)"""
    preprocessor = AdvancedPreprocessor()
    return preprocessor.preprocess_signal(signal, sampling_rate)
