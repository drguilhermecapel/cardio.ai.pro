"""
Signal quality analysis utilities.
"""

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class SignalQualityAnalyzer:
    """Analyzer for ECG signal quality assessment."""

    async def analyze_quality(self, ecg_data: np.ndarray) -> dict[str, Any]:
        """Analyze ECG signal quality."""
        try:
            quality_metrics = {
                "overall_score": 0.0,
                "noise_level": 0.0,
                "baseline_wander": 0.0,
                "signal_to_noise_ratio": 0.0,
                "artifacts_detected": [],
                "quality_issues": [],
            }

            lead_scores = []

            for i in range(ecg_data.shape[1]):
                lead_data = ecg_data[:, i]
                lead_quality = await self._analyze_lead_quality(lead_data)
                lead_scores.append(lead_quality["score"])

                quality_metrics["artifacts_detected"].extend(lead_quality.get("artifacts", []))
                quality_metrics["quality_issues"].extend(lead_quality.get("issues", []))

            quality_metrics["overall_score"] = np.mean(lead_scores)
            quality_metrics["noise_level"] = await self._calculate_noise_level(ecg_data)
            quality_metrics["baseline_wander"] = await self._calculate_baseline_wander(ecg_data)
            quality_metrics["signal_to_noise_ratio"] = await self._calculate_snr(ecg_data)

            if quality_metrics["overall_score"] < 0.5:
                quality_metrics["quality_issues"].append("Poor overall signal quality")

            if quality_metrics["noise_level"] > 0.3:
                quality_metrics["quality_issues"].append("High noise level detected")

            if quality_metrics["baseline_wander"] > 0.2:
                quality_metrics["quality_issues"].append("Significant baseline wander")

            if quality_metrics["signal_to_noise_ratio"] < 10:
                quality_metrics["quality_issues"].append("Low signal-to-noise ratio")

            return quality_metrics

        except Exception as e:
            logger.error(f"Signal quality analysis failed: {str(e)}")
            return {
                "overall_score": 0.5,
                "noise_level": 0.0,
                "baseline_wander": 0.0,
                "signal_to_noise_ratio": 0.0,
                "artifacts_detected": [],
                "quality_issues": ["Quality analysis failed"],
            }

    async def _analyze_lead_quality(self, lead_data: np.ndarray) -> dict[str, Any]:
        """Analyze quality of a single ECG lead."""
        try:
            quality = {
                "score": 0.0,
                "artifacts": [],
                "issues": [],
            }

            if np.std(lead_data) < 0.01:
                quality["score"] = 0.0
                quality["artifacts"].append("flat_line")
                quality["issues"].append("Possible electrode disconnection")
                return quality

            max_val = np.max(np.abs(lead_data))
            if max_val > 10:  # Assuming mV units
                quality["artifacts"].append("saturation")
                quality["issues"].append("Signal saturation detected")

            signal_variance = np.var(lead_data)
            if signal_variance > 5:
                quality["artifacts"].append("high_noise")
                quality["issues"].append("High noise level")

            score = 1.0

            if signal_variance > 2:
                score -= 0.3

            if max_val > 5:
                score -= 0.2

            if max_val < 0.1:
                score -= 0.2

            quality["score"] = max(0.0, score)

            return quality

        except Exception as e:
            logger.error(f"Lead quality analysis failed: {str(e)}")
            return {"score": 0.5, "artifacts": [], "issues": []}

    async def _calculate_noise_level(self, ecg_data: np.ndarray) -> float:
        """Calculate noise level in ECG signal."""
        try:
            noise_levels = []

            for i in range(ecg_data.shape[1]):
                lead_data = ecg_data[:, i]

                from scipy import signal
                frequencies, power = signal.welch(lead_data, fs=500, nperseg=1024)

                high_freq_mask = frequencies > 50
                noise_power = np.sum(power[high_freq_mask])
                total_power = np.sum(power)

                noise_ratio = noise_power / (total_power + 1e-8)
                noise_levels.append(noise_ratio)

            return float(np.mean(noise_levels))

        except Exception as e:
            logger.error(f"Noise level calculation failed: {str(e)}")
            return 0.1

    async def _calculate_baseline_wander(self, ecg_data: np.ndarray) -> float:
        """Calculate baseline wander in ECG signal."""
        try:
            wander_levels = []

            for i in range(ecg_data.shape[1]):
                lead_data = ecg_data[:, i]

                from scipy import signal
                frequencies, power = signal.welch(lead_data, fs=500, nperseg=1024)

                low_freq_mask = frequencies < 1
                wander_power = np.sum(power[low_freq_mask])
                total_power = np.sum(power)

                wander_ratio = wander_power / (total_power + 1e-8)
                wander_levels.append(wander_ratio)

            return float(np.mean(wander_levels))

        except Exception as e:
            logger.error(f"Baseline wander calculation failed: {str(e)}")
            return 0.1

    async def _calculate_snr(self, ecg_data: np.ndarray) -> float:
        """Calculate signal-to-noise ratio."""
        try:
            snr_values = []

            for i in range(ecg_data.shape[1]):
                lead_data = ecg_data[:, i]

                signal_power = np.var(lead_data)

                from scipy import signal
                b, a = signal.butter(4, 0.1, btype='high', fs=500)
                noise_estimate = signal.filtfilt(b, a, lead_data)
                noise_power = np.var(noise_estimate)

                snr = signal_power / (noise_power + 1e-8)
                snr_db = 10 * np.log10(snr + 1e-8)
                snr_values.append(snr_db)

            return float(np.mean(snr_values))

        except Exception as e:
            logger.error(f"SNR calculation failed: {str(e)}")
            return 20.0  # Default reasonable SNR
