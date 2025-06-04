from typing import Any, List, Optional, Tuple, Union
import numpy as np
import numpy.typing as npt

def wavedec(data: npt.NDArray[np.float64], wavelet: str, level: Optional[int] = None,
           mode: str = 'symmetric') -> List[npt.NDArray[np.float64]]: ...

def waverec(coeffs: List[npt.NDArray[np.float64]], wavelet: str, 
           mode: str = 'symmetric') -> npt.NDArray[np.float64]: ...

def dwt(data: npt.NDArray[np.float64], wavelet: str, 
       mode: str = 'symmetric') -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...

def idwt(cA: Optional[npt.NDArray[np.float64]], cD: Optional[npt.NDArray[np.float64]], 
        wavelet: str, mode: str = 'symmetric') -> npt.NDArray[np.float64]: ...

def threshold(data: npt.NDArray[np.float64], value: float, mode: str = 'soft',
             substitute: float = 0) -> npt.NDArray[np.float64]: ...

def cwt(data: npt.NDArray[np.float64], scales: npt.NDArray[np.float64], 
       wavelet: str, sampling_period: float = 1.0) -> npt.NDArray[np.complex128]: ...

class Wavelet:
    name: str
    def __init__(self, name: str) -> None: ...

def wavelist(kind: Optional[str] = None) -> List[str]: ...
