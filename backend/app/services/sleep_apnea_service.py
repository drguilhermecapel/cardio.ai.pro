"""
Sleep Apnea and Respiratory Analysis Service
Implements respiratory pattern detection and sleep disorder analysis from ECG signals
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

try:
    import scipy.signal
    from scipy.stats import entropy
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("SciPy not available. Some respiratory analysis features will be limited.")

try:
    import neurokit2 as nk
    NEUROKIT_AVAILABLE = True
except ImportError:
    NEUROKIT_AVAILABLE = False
    logger.warning("NeuroKit2 not available. Advanced HRV analysis will be limited.")


class RespiratoryPatternAnalyzer:
    """Analyzes respiratory patterns from ECG signals"""
    
    def __init__(self, sampling_rate: int = 500):
        self.sampling_rate = sampling_rate
        self.respiratory_bands = {
            "normal": (0.15, 0.4),      # 9-24 breaths/min
            "bradypnea": (0.05, 0.15),  # 3-9 breaths/min
            "tachypnea": (0.4, 0.8),    # 24-48 breaths/min
        }
        
    def extract_respiratory_signal(
        self, 
        ecg_signal: npt.NDArray[np.float64],
        method: str = "edr"
    ) -> npt.NDArray[np.float64]:
        """Extract respiratory signal from ECG using ECG-derived respiration (EDR)"""
        try:
            if method == "edr":
                return self._extract_edr_signal(ecg_signal)
            elif method == "rsa":
                return self._extract_rsa_signal(ecg_signal)
            elif method == "amplitude_modulation":
                return self._extract_amplitude_modulation(ecg_signal)
            else:
                raise ValueError(f"Unknown respiratory extraction method: {method}")
                
        except Exception as e:
            logger.error(f"Respiratory signal extraction failed: {e}")
            return np.zeros_like(ecg_signal)
            
    def analyze_respiratory_rate(
        self, 
        respiratory_signal: npt.NDArray[np.float64]
    ) -> Dict[str, Any]:
        """Analyze respiratory rate from respiratory signal"""
        try:
            if len(respiratory_signal) < 100:
                return {"respiratory_rate": 0.0, "confidence": 0.0}
                
            if SCIPY_AVAILABLE:
                freqs, psd = scipy.signal.welch(
                    respiratory_signal,
                    fs=self.sampling_rate,
                    nperseg=min(len(respiratory_signal)//4, 1024)
                )
                
                resp_mask = (freqs >= 0.1) & (freqs <= 0.8)
                if np.any(resp_mask):
                    resp_freqs = freqs[resp_mask]
                    resp_psd = psd[resp_mask]
                    
                    peak_idx = np.argmax(resp_psd)
                    peak_freq = resp_freqs[peak_idx]
                    
                    respiratory_rate = peak_freq * 60
                    
                    peak_power = resp_psd[peak_idx]
                    total_power = np.sum(resp_psd)
                    confidence = peak_power / total_power if total_power > 0 else 0.0
                    
                else:
                    respiratory_rate = 0.0
                    confidence = 0.0
                    
            else:
                zero_crossings = np.where(np.diff(np.signbit(respiratory_signal)))[0]
                if len(zero_crossings) > 2:
                    duration = len(respiratory_signal) / self.sampling_rate
                    respiratory_rate = (len(zero_crossings) / 2) / duration * 60
                    confidence = 0.5
                else:
                    respiratory_rate = 0.0
                    confidence = 0.0
                    
            return {
                "respiratory_rate": float(respiratory_rate),
                "confidence": float(confidence),
                "classification": self._classify_respiratory_rate(respiratory_rate)
            }
            
        except Exception as e:
            logger.error(f"Respiratory rate analysis failed: {e}")
            return {"respiratory_rate": 0.0, "confidence": 0.0, "classification": "unknown"}
            
    def _extract_edr_signal(self, ecg_signal: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Extract EDR signal using R-wave amplitude variations"""
        try:
            r_peaks = self._detect_r_peaks(ecg_signal)
            
            if len(r_peaks) < 10:
                return np.zeros_like(ecg_signal)
                
            r_amplitudes = ecg_signal[r_peaks]
            time_peaks = r_peaks / self.sampling_rate
            time_original = np.arange(len(ecg_signal)) / self.sampling_rate
            
            edr_signal = np.interp(time_original, time_peaks, r_amplitudes)
            
            if len(edr_signal) > 100 and SCIPY_AVAILABLE:
                edr_signal = self._bandpass_filter(edr_signal, 0.1, 0.5)
                
            return edr_signal
            
        except Exception as e:
            logger.error(f"EDR extraction failed: {e}")
            return np.zeros_like(ecg_signal)
            
    def _extract_rsa_signal(self, ecg_signal: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Extract respiratory sinus arrhythmia (RSA) signal"""
        try:
            r_peaks = self._detect_r_peaks(ecg_signal)
            
            if len(r_peaks) < 10:
                return np.zeros_like(ecg_signal)
                
            rr_intervals = np.diff(r_peaks) / self.sampling_rate
            
            if len(rr_intervals) < 5:
                return np.zeros_like(ecg_signal)
                
            time_rr = r_peaks[1:] / self.sampling_rate
            time_original = np.arange(len(ecg_signal)) / self.sampling_rate
            
            rsa_signal = np.interp(time_original, time_rr, rr_intervals)
            
            if len(rsa_signal) > 100 and SCIPY_AVAILABLE:
                rsa_signal = self._bandpass_filter(rsa_signal, 0.15, 0.4)
                
            return rsa_signal
            
        except Exception as e:
            logger.error(f"RSA extraction failed: {e}")
            return np.zeros_like(ecg_signal)
            
    def _extract_amplitude_modulation(self, ecg_signal: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Extract respiratory signal using amplitude modulation"""
        try:
            if SCIPY_AVAILABLE:
                analytic_signal = scipy.signal.hilbert(ecg_signal)
                envelope = np.abs(analytic_signal)
            else:
                window_size = int(0.1 * self.sampling_rate)
                envelope = np.convolve(np.abs(ecg_signal), 
                                     np.ones(window_size)/window_size, 
                                     mode='same')
                
            if len(envelope) > 100 and SCIPY_AVAILABLE:
                respiratory_signal = self._bandpass_filter(envelope, 0.1, 0.5)
            else:
                respiratory_signal = envelope
                
            return respiratory_signal
            
        except Exception as e:
            logger.error(f"Amplitude modulation extraction failed: {e}")
            return np.zeros_like(ecg_signal)
            
    def _detect_r_peaks(self, ecg_signal: npt.NDArray[np.float64]) -> npt.NDArray[np.int64]:
        """Detect R-peaks in ECG signal"""
        try:
            if NEUROKIT_AVAILABLE:
                _, info = nk.ecg_peaks(ecg_signal, sampling_rate=self.sampling_rate)
                return np.array(info["ECG_R_Peaks"], dtype=np.int64)
            elif SCIPY_AVAILABLE:
                min_height = np.std(ecg_signal) * 0.5
                min_distance = int(0.6 * self.sampling_rate)
                
                peaks, _ = scipy.signal.find_peaks(
                    ecg_signal,
                    height=min_height,
                    distance=min_distance
                )
                return peaks.astype(np.int64)
            else:
                diff_signal = np.diff(ecg_signal)
                peaks = []
                
                for i in range(1, len(diff_signal)):
                    if diff_signal[i-1] > 0 and diff_signal[i] <= 0:
                        if ecg_signal[i] > np.mean(ecg_signal) + np.std(ecg_signal):
                            peaks.append(i)
                            
                return np.array(peaks, dtype=np.int64)
                
        except Exception as e:
            logger.error(f"R-peak detection failed: {e}")
            return np.array([], dtype=np.int64)
            
    def _bandpass_filter(
        self, 
        signal: npt.NDArray[np.float64], 
        low_freq: float, 
        high_freq: float
    ) -> npt.NDArray[np.float64]:
        """Apply bandpass filter to signal"""
        if not SCIPY_AVAILABLE:
            return signal
            
        try:
            nyquist = self.sampling_rate / 2
            low = low_freq / nyquist
            high = high_freq / nyquist
            
            if high >= 1.0:
                high = 0.99
            if low <= 0.0:
                low = 0.01
                
            b, a = scipy.signal.butter(4, [low, high], btype='band')
            filtered_signal = scipy.signal.filtfilt(b, a, signal)
            
            return filtered_signal
            
        except Exception as e:
            logger.error(f"Bandpass filtering failed: {e}")
            return signal
            
    def _classify_respiratory_rate(self, rate: float) -> str:
        """Classify respiratory rate"""
        if rate < 9:
            return "bradypnea"
        elif rate > 24:
            return "tachypnea"
        else:
            return "normal"
            


class SleepApneaDetector:
    """Sleep apnea detection from ECG signals"""
    
    def __init__(self, sampling_rate: int = 500):
        self.sampling_rate = sampling_rate
        self.apnea_thresholds = {
            "mild": 5,      # 5-14 events per hour
            "moderate": 15, # 15-29 events per hour
            "severe": 30    # 30+ events per hour
        }
        
    async def detect_sleep_apnea(
        self,
        ecg_signal: npt.NDArray[np.float64],
        duration_hours: float = 1.0
    ) -> Dict[str, Any]:
        """Detect sleep apnea events from ECG signal"""
        try:
            start_time = time.time()
            
            respiratory_analyzer = RespiratoryPatternAnalyzer(self.sampling_rate)
            respiratory_signal = respiratory_analyzer.extract_respiratory_signal(ecg_signal, method="edr")
            
            apnea_events = self._detect_apnea_events(respiratory_signal)
            hypopnea_events = self._detect_hypopnea_events(respiratory_signal)
            
            total_events = len(apnea_events) + len(hypopnea_events)
            ahi = total_events / duration_hours
            
            severity = self._classify_sleep_apnea_severity(ahi)
            
            processing_time = time.time() - start_time
            
            return {
                "ahi": float(ahi),
                "severity": severity,
                "apnea_events": apnea_events,
                "hypopnea_events": hypopnea_events,
                "total_events": total_events,
                "recommendations": self._generate_apnea_recommendations(severity, ahi),
                "processing_time": processing_time,
                "confidence": self._calculate_detection_confidence(len(apnea_events), len(hypopnea_events))
            }
            
        except Exception as e:
            logger.error(f"Sleep apnea detection failed: {e}")
            return self._empty_apnea_result()
            
    def _detect_apnea_events(self, respiratory_signal: npt.NDArray[np.float64]) -> List[Dict[str, Any]]:
        """Detect apnea events (cessation of breathing)"""
        try:
            if len(respiratory_signal) < 100:
                return []
                
            if SCIPY_AVAILABLE:
                envelope = np.abs(scipy.signal.hilbert(respiratory_signal))
            else:
                window_size = int(0.5 * self.sampling_rate)
                envelope = np.convolve(np.abs(respiratory_signal), 
                                     np.ones(window_size)/window_size, 
                                     mode='same')
                
            baseline_amplitude = np.percentile(envelope, 75)
            apnea_threshold = baseline_amplitude * 0.1
            
            below_threshold = envelope < apnea_threshold
            min_duration = int(10 * self.sampling_rate)
            
            apnea_events = []
            in_apnea = False
            start_idx = 0
            
            for i, is_below in enumerate(below_threshold):
                if is_below and not in_apnea:
                    in_apnea = True
                    start_idx = i
                elif not is_below and in_apnea:
                    in_apnea = False
                    duration = i - start_idx
                    
                    if duration >= min_duration:
                        apnea_events.append({
                            "start_time": float(start_idx / self.sampling_rate),
                            "end_time": float(i / self.sampling_rate),
                            "duration": float(duration / self.sampling_rate),
                            "type": "apnea",
                            "severity": "severe" if duration > 30 * self.sampling_rate else "moderate"
                        })
                        
            return apnea_events
            
        except Exception as e:
            logger.error(f"Apnea event detection failed: {e}")
            return []
            
    def _detect_hypopnea_events(self, respiratory_signal: npt.NDArray[np.float64]) -> List[Dict[str, Any]]:
        """Detect hypopnea events (reduced breathing)"""
        try:
            if len(respiratory_signal) < 100:
                return []
                
            if SCIPY_AVAILABLE:
                envelope = np.abs(scipy.signal.hilbert(respiratory_signal))
            else:
                window_size = int(0.5 * self.sampling_rate)
                envelope = np.convolve(np.abs(respiratory_signal), 
                                     np.ones(window_size)/window_size, 
                                     mode='same')
                
            baseline_amplitude = np.percentile(envelope, 75)
            hypopnea_threshold_low = baseline_amplitude * 0.1
            hypopnea_threshold_high = baseline_amplitude * 0.7
            
            in_hypopnea_range = (envelope >= hypopnea_threshold_low) & (envelope <= hypopnea_threshold_high)
            min_duration = int(10 * self.sampling_rate)
            
            hypopnea_events = []
            in_hypopnea = False
            start_idx = 0
            
            for i, is_in_range in enumerate(in_hypopnea_range):
                if is_in_range and not in_hypopnea:
                    in_hypopnea = True
                    start_idx = i
                elif not is_in_range and in_hypopnea:
                    in_hypopnea = False
                    duration = i - start_idx
                    
                    if duration >= min_duration:
                        segment_amplitude = np.mean(envelope[start_idx:i])
                        reduction_percent = (1 - segment_amplitude / baseline_amplitude) * 100
                        
                        hypopnea_events.append({
                            "start_time": float(start_idx / self.sampling_rate),
                            "end_time": float(i / self.sampling_rate),
                            "duration": float(duration / self.sampling_rate),
                            "type": "hypopnea",
                            "reduction_percent": float(reduction_percent),
                            "severity": "severe" if reduction_percent > 70 else "moderate"
                        })
                        
            return hypopnea_events
            
        except Exception as e:
            logger.error(f"Hypopnea event detection failed: {e}")
            return []
            
    def _classify_sleep_apnea_severity(self, ahi: float) -> str:
        """Classify sleep apnea severity based on AHI"""
        if ahi < 5:
            return "normal"
        elif ahi < 15:
            return "mild"
        elif ahi < 30:
            return "moderate"
        else:
            return "severe"
            
    def _calculate_detection_confidence(self, apnea_count: int, hypopnea_count: int) -> float:
        """Calculate confidence in sleep apnea detection"""
        try:
            confidence = 0.5
            
            total_events = apnea_count + hypopnea_count
            if total_events > 10:
                confidence += 0.2
            elif total_events > 5:
                confidence += 0.1
                
            if total_events < 2:
                confidence -= 0.2
                
            return float(max(0.1, min(1.0, confidence)))
            
        except Exception as e:
            logger.error(f"Confidence calculation failed: {e}")
            return 0.5
            
    def _generate_apnea_recommendations(self, severity: str, ahi: float) -> List[str]:
        """Generate sleep apnea recommendations"""
        recommendations = []
        
        if severity == "severe":
            recommendations.extend([
                "Urgent sleep medicine consultation required",
                "Polysomnography (sleep study) recommended",
                "Consider CPAP therapy evaluation",
                "Cardiovascular risk assessment recommended"
            ])
        elif severity == "moderate":
            recommendations.extend([
                "Sleep medicine consultation recommended",
                "Sleep study (polysomnography) advised",
                "Lifestyle modifications (weight loss, sleep position)",
                "Consider oral appliance therapy"
            ])
        elif severity == "mild":
            recommendations.extend([
                "Sleep hygiene counseling",
                "Weight management if applicable",
                "Consider sleep study if symptoms persist",
                "Positional therapy evaluation"
            ])
        else:
            recommendations.extend([
                "Maintain good sleep hygiene",
                "Regular follow-up if symptoms develop"
            ])
            
        return recommendations
        
    def _empty_apnea_result(self) -> Dict[str, Any]:
        """Return empty apnea result structure"""
        return {
            "ahi": 0.0,
            "severity": "unknown",
            "apnea_events": [],
            "hypopnea_events": [],
            "total_events": 0,
            "recommendations": [],
            "processing_time": 0.0,
            "confidence": 0.0
        }
