import logging
from typing import Any

import numpy as np
import pywt  # type: ignore[import-untyped]
from numpy.typing import NDArray
from scipy.signal import butter, iirnotch, sosfiltfilt

logger = logging.getLogger(__name__)


class ECGSignalProcessor:
    """Processador de sinais ECG com padrões médicos corretos"""

    DIAGNOSTIC_HIGHPASS = 0.05  # Hz - Padrão AHA/ACC (NÃO usar 0.5!)
    DIAGNOSTIC_LOWPASS = 150    # Hz - Preserva morfologia QRS
    MONITORING_HIGHPASS = 0.5   # Hz - Apenas para monitoramento
    MIN_SAMPLING_RATE = 500     # Hz - Mínimo absoluto
    RECOMMENDED_SAMPLING_RATE = 1000  # Hz - Recomendado

    def __init__(self, sampling_rate: int = 500, mode: str = 'diagnostic'):
        """
        Args:
            sampling_rate: Taxa de amostragem em Hz (mínimo 500)
            mode: 'diagnostic' ou 'monitoring'
        """
        if sampling_rate < self.MIN_SAMPLING_RATE:
            raise ValueError(
                f"Taxa de amostragem {sampling_rate} Hz é inadequada! "
                f"Mínimo: {self.MIN_SAMPLING_RATE} Hz, "
                f"Recomendado: {self.RECOMMENDED_SAMPLING_RATE} Hz"
            )

        self.fs = sampling_rate
        self.mode = mode
        self.nyquist = self.fs / 2

        self._setup_filters()

    def _setup_filters(self) -> None:
        """Configurar filtros conforme padrões médicos"""
        if self.mode == 'diagnostic':
            hp_freq = self.DIAGNOSTIC_HIGHPASS
        else:
            hp_freq = self.MONITORING_HIGHPASS

        self.sos_highpass = butter(
            2, hp_freq, 'hp', fs=self.fs, output='sos'
        )

        self.sos_lowpass = butter(
            4, self.DIAGNOSTIC_LOWPASS, 'lp', fs=self.fs, output='sos'
        )

        b_60, a_60 = iirnotch(60, 30, self.fs)
        self.notch_60 = butter(2, [59, 61], 'bandstop', fs=self.fs, output='sos')

        b_50, a_50 = iirnotch(50, 30, self.fs)
        self.notch_50 = butter(2, [49, 51], 'bandstop', fs=self.fs, output='sos')

    def process_diagnostic(self, ecg_signal: NDArray[np.floating[Any]],
                         power_line_freq: int = 60) -> NDArray[np.floating[Any]]:
        """
        Processamento para análise diagnóstica (máxima fidelidade)

        Args:
            ecg_signal: Sinal ECG bruto
            power_line_freq: Frequência da rede elétrica (50 ou 60 Hz)

        Returns:
            Sinal filtrado para diagnóstico
        """
        signal_centered = ecg_signal - np.mean(ecg_signal)

        filtered = sosfiltfilt(self.sos_highpass, signal_centered)

        filtered = sosfiltfilt(self.sos_lowpass, filtered)

        if power_line_freq == 60:
            filtered = sosfiltfilt(self.notch_60, filtered)
        else:
            filtered = sosfiltfilt(self.notch_50, filtered)

        return filtered

    def remove_baseline_wander(self, ecg_signal: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        """Remover oscilação de linha de base usando Wavelet"""
        try:
            coeffs = pywt.wavedec(ecg_signal, 'db4', level=9)

            coeffs[0] = np.zeros_like(coeffs[0])
            coeffs[1] = np.zeros_like(coeffs[1])

            result = pywt.waverec(coeffs, 'db4')
            return np.asarray(result, dtype=np.float64)
        except Exception as e:
            logger.warning(f"Wavelet baseline removal failed: {e}")
            return ecg_signal
