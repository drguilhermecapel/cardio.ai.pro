"""Type stubs for wfdb library (PhysioNet WFDB)."""

from typing import Any

import numpy as np
from numpy.typing import NDArray

def rdsamp(
    record_name: str,
    pn_dir: str | None = None,
    sampfrom: int = 0,
    sampto: int | None = None,
    channels: list[int] | None = None,
    physical: bool = True,
    return_res: int = 64,
    channel_names: list[str] | None = None
) -> tuple[NDArray[np.float64], dict[str, Any]]: ...

def rdrecord(
    record_name: str,
    pn_dir: str | None = None,
    sampfrom: int = 0,
    sampto: int | None = None,
    channels: list[int] | None = None,
    physical: bool = True,
    channel_names: list[str] | None = None,
    return_res: int = 64,
    warn_empty: bool = False
) -> Any: ...

class Record:
    """WFDB Record class."""
    sig_name: list[str]
    units: list[str]
    fs: int | float
    sig_len: int
    p_signal: NDArray[np.float64] | None
    d_signal: NDArray[np.int_] | None

    def __init__(self, **kwargs: Any) -> None: ...
