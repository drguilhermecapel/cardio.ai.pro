from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import numpy.typing as npt

def rdsamp(record_name: str, sampfrom: int = 0, sampto: Optional[int] = None, 
          channels: Optional[List[int]] = None, physical: bool = True,
          return_res: int = 64, smooth_frames: bool = True) -> Tuple[npt.NDArray[np.float64], Dict[str, Any]]: ...

def rdheader(record_name: str) -> Any: ...

def rdann(record_name: str, extension: str, sampfrom: int = 0, 
         sampto: Optional[int] = None, shift_samps: bool = False,
         return_label_elements: List[str] = ['symbol']) -> Any: ...

class Record:
    sig_name: List[str]
    fs: int
    sig_len: int
    n_sig: int
    p_signal: npt.NDArray[np.float64]
    d_signal: npt.NDArray[np.int16]
    
    def __init__(self) -> None: ...

def wrsamp(record_name: str, fs: int, units: List[str], sig_name: List[str],
          p_signal: npt.NDArray[np.float64], fmt: Optional[List[str]] = None,
          adc_gain: Optional[List[float]] = None, baseline: Optional[List[int]] = None,
          comments: Optional[List[str]] = None, base_time: Optional[str] = None,
          base_date: Optional[str] = None) -> None: ...
