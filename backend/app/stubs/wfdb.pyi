"""Type stubs for wfdb library (PhysioNet WFDB)."""

from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np
from numpy.typing import NDArray

def rdsamp(
    record_name: str,
    pn_dir: Optional[str] = None,
    sampfrom: int = 0,
    sampto: Optional[int] = None,
    channels: Optional[List[int]] = None,
    physical: bool = True,
    return_res: int = 64,
    channel_names: Optional[List[str]] = None
) -> Tuple[NDArray[np.float64], Dict[str, Any]]: ...

def rdrecord(
    record_name: str,
    pn_dir: Optional[str] = None,
    sampfrom: int = 0,
    sampto: Optional[int] = None,
    channels: Optional[List[int]] = None,
    physical: bool = True,
    channel_names: Optional[List[str]] = None,
    return_res: int = 64,
    warn_empty: bool = False
) -> Any: ...

class Record:
    """WFDB Record class."""
    sig_name: List[str]
    units: List[str]
    fs: Union[int, float]
    sig_len: int
    p_signal: Optional[NDArray[np.float64]]
    d_signal: Optional[NDArray[np.int_]]
    
    def __init__(self, **kwargs: Any) -> None: ...
