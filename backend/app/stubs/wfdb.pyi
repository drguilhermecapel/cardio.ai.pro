from typing import Any

import numpy as np
import numpy.typing as npt

def rdsamp(record_name: str, sampfrom: int = 0, sampto: int | None = None,
          channels: list[int] | None = None, physical: bool = True,
          return_res: int = 64, smooth_frames: bool = True) -> tuple[npt.NDArray[np.float64], dict[str, Any]]: ...

def rdheader(record_name: str) -> Any: ...

def rdann(record_name: str, extension: str, sampfrom: int = 0,
         sampto: int | None = None, shift_samps: bool = False,
         return_label_elements: list[str] | None = None) -> Any: ...

class Record:
    sig_name: list[str]
    fs: int
    sig_len: int
    n_sig: int
    p_signal: npt.NDArray[np.float64]
    d_signal: npt.NDArray[np.int16]

    def __init__(self) -> None: ...

def wrsamp(record_name: str, fs: int, units: list[str], sig_name: list[str],
          p_signal: npt.NDArray[np.float64], fmt: list[str] | None = None,
          adc_gain: list[float] | None = None, baseline: list[int] | None = None,
          comments: list[str] | None = None, base_time: str | None = None,
          base_date: str | None = None) -> None: ...
