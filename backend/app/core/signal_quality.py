import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, sosfiltfilt

logger = logging.getLogger(__name__)


class SignalQualityAssessment:
    """Sistema completo de avaliação de qualidade de ECG"""

    def __init__(self, sampling_rate: int = 500):
        self.fs = sampling_rate
        self.quality_thresholds = {
            'snr': 10,          # dB
            'baseline_power': 0.1,  # Potência relativa
            'saturation': 0.05,     # 5% de amostras saturadas
            'flatline': 0.1,        # 10% de sinal flat
            'noise_level': 0.2      # mV
        }

    def assess_comprehensive(self, ecg_leads: dict[str, NDArray[np.floating[Any]]]) -> dict[str, Any]:
        """
        Avaliação completa de qualidade multi-derivação

        Args:
            ecg_leads: Dicionário com sinais de cada derivação

        Returns:
            Relatório detalhado de qualidade
        """
        report = {
            'timestamp': np.datetime64('now'),
            'overall_quality': 0.0,
            'lead_quality': {},
            'issues': [],
            'recommendations': []
        }

        for lead_name, signal_data in ecg_leads.items():
            lead_report = self._assess_single_lead(lead_name, signal_data)
            lead_quality = report.get('lead_quality')
            if isinstance(lead_quality, dict):
                lead_quality[lead_name] = lead_report

            if lead_report['quality_score'] < 0.7:
                if isinstance(report['issues'], list) and isinstance(lead_report['issues'], list):
                    report['issues'].extend(lead_report['issues'])

        if isinstance(report['lead_quality'], dict):
            quality_scores = [lead_data['quality_score'] for lead_data in report['lead_quality'].values()]
        else:
            quality_scores = [0.0]
        report['overall_quality'] = np.mean(quality_scores)

        report['recommendations'] = self._generate_recommendations(report)

        if isinstance(report['overall_quality'], int | float):
            report['acceptable_for_diagnosis'] = report['overall_quality'] >= 0.7
        else:
            report['acceptable_for_diagnosis'] = False

        return report

    def _assess_single_lead(self, lead_name: str, signal_data: NDArray[np.floating[Any]]) -> dict[str, Any]:
        """Avaliar qualidade de uma derivação"""
        metrics = {}
        issues = []

        snr = self._calculate_snr(signal_data)
        metrics['snr'] = snr
        if snr < self.quality_thresholds['snr']:
            issues.append(f"SNR baixo em {lead_name}: {snr:.1f} dB")

        saturation_ratio = self._detect_saturation(signal_data)
        metrics['saturation'] = saturation_ratio
        if saturation_ratio > self.quality_thresholds['saturation']:
            issues.append(f"Saturação detectada em {lead_name}: {saturation_ratio*100:.1f}%")

        is_flatline = self._detect_flatline(signal_data)
        metrics['flatline'] = is_flatline
        if is_flatline:
            issues.append(f"Possível eletrodo desconectado em {lead_name}")

        noise_level = self._estimate_noise_level(signal_data)
        metrics['noise_level'] = noise_level
        if noise_level > self.quality_thresholds['noise_level']:
            issues.append(f"Ruído excessivo em {lead_name}: {noise_level:.3f} mV")

        baseline_power = self._assess_baseline_wander(signal_data)
        metrics['baseline_wander'] = baseline_power
        if baseline_power > self.quality_thresholds['baseline_power']:
            issues.append(f"Oscilação de linha de base em {lead_name}")

        quality_score = self._calculate_quality_score(metrics)

        return {
            'lead_name': lead_name,
            'quality_score': quality_score,
            'metrics': metrics,
            'issues': issues
        }

    def _calculate_snr(self, signal_data: NDArray[np.floating[Any]]) -> float:
        """Calcular SNR usando método de banda QRS"""
        try:
            sos = butter(3, [5, 40], 'bandpass', fs=self.fs, output='sos')
            qrs_band = sosfiltfilt(sos, signal_data)

            noise = signal_data - qrs_band

            signal_power = np.mean(qrs_band**2)
            noise_power = np.mean(noise**2)

            if noise_power > 0:
                snr_db = 10 * np.log10(signal_power / noise_power)
            else:
                snr_db = 100  # Sinal perfeito

            return float(snr_db)
        except Exception as e:
            logger.warning(f"SNR calculation failed: {e}")
            return 0.0

    def _detect_saturation(self, signal_data: NDArray[np.floating[Any]]) -> float:
        """Detectar saturação do sinal"""
        try:
            max_val = np.max(np.abs(signal_data))
            saturation_threshold = max_val * 0.95

            saturated_samples = np.sum(np.abs(signal_data) >= saturation_threshold)
            saturation_ratio = saturated_samples / len(signal_data)

            return float(saturation_ratio)
        except Exception as e:
            logger.warning(f"Saturation detection failed: {e}")
            return 0.0

    def _detect_flatline(self, signal_data: NDArray[np.floating[Any]]) -> bool:
        """Detectar eletrodo desconectado (sinal flat)"""
        try:
            window_size = min(self.fs, len(signal_data) // 10)  # 1s ou 10% do sinal
            variances = []

            for i in range(0, len(signal_data) - window_size, window_size // 2):
                window = signal_data[i:i + window_size]
                variances.append(np.var(window))

            low_variance_threshold = np.var(signal_data) * 0.01
            low_variance_windows = np.sum(np.array(variances) < low_variance_threshold)

            return bool(low_variance_windows > len(variances) * 0.5)
        except Exception as e:
            logger.warning(f"Flatline detection failed: {e}")
            return False

    def _estimate_noise_level(self, signal_data: NDArray[np.floating[Any]]) -> float:
        """Estimar nível de ruído usando diferenças de alta frequência"""
        try:
            diff2 = np.diff(signal_data, n=2)
            noise_estimate = np.std(diff2) / np.sqrt(6)  # Normalização teórica

            return float(noise_estimate)
        except Exception as e:
            logger.warning(f"Noise estimation failed: {e}")
            return 0.0

    def _assess_baseline_wander(self, signal_data: NDArray[np.floating[Any]]) -> float:
        """Avaliar oscilação de linha de base"""
        try:
            sos = butter(2, 0.5, 'lp', fs=self.fs, output='sos')
            baseline = sosfiltfilt(sos, signal_data)

            baseline_power = np.var(baseline)
            signal_power = np.var(signal_data)

            relative_power = baseline_power / (signal_power + 1e-10)

            return float(relative_power)
        except Exception as e:
            logger.warning(f"Baseline wander assessment failed: {e}")
            return 0.0

    def _calculate_quality_score(self, metrics: dict[str, Any]) -> float:
        """Calcular score de qualidade final"""
        try:
            scores = []

            snr_score = min(metrics['snr'] / 20.0, 1.0)  # 20 dB = score 1.0
            scores.append(snr_score)

            saturation_score = 1.0 - min(metrics['saturation'] / 0.1, 1.0)
            scores.append(saturation_score)

            flatline_score = 0.0 if metrics['flatline'] else 1.0
            scores.append(flatline_score)

            noise_score = 1.0 - min(metrics['noise_level'] / 0.5, 1.0)
            scores.append(noise_score)

            baseline_score = 1.0 - min(metrics['baseline_wander'] / 0.2, 1.0)
            scores.append(baseline_score)

            weights = [0.3, 0.2, 0.2, 0.15, 0.15]
            quality_score = sum(w * s for w, s in zip(weights, scores, strict=False))

            return float(max(0.0, min(1.0, quality_score)))
        except Exception as e:
            logger.warning(f"Quality score calculation failed: {e}")
            return 0.5

    def _generate_recommendations(self, report: dict[str, Any]) -> list[str]:
        """Gerar recomendações baseadas nos problemas encontrados"""
        recommendations = []

        if report['overall_quality'] < 0.7:
            recommendations.append("Qualidade geral inadequada para diagnóstico")

        all_issues = report['issues']

        if any('SNR baixo' in issue for issue in all_issues):
            recommendations.append("Verificar conexões dos eletrodos e reduzir interferências")

        if any('Saturação' in issue for issue in all_issues):
            recommendations.append("Ajustar ganho do amplificador ou verificar eletrodos")

        if any('eletrodo desconectado' in issue for issue in all_issues):
            recommendations.append("Verificar conexão e aderência dos eletrodos")

        if any('Ruído excessivo' in issue for issue in all_issues):
            recommendations.append("Reduzir fontes de interferência e verificar aterramento")

        if any('linha de base' in issue for issue in all_issues):
            recommendations.append("Verificar respiração do paciente e posicionamento dos eletrodos")

        return recommendations
