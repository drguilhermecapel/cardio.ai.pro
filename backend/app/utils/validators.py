from typing import Any, Dict
import os

def validate_ecg_file(file_path: str) -> bool:
    return os.path.exists(file_path)

def validate_ecg_data(data: Dict[str, Any]) -> bool:
    required = ["signal", "sampling_rate"]
    return all(field in data for field in required)

def validate_patient_data(data: Dict[str, Any]) -> bool:
    required = ["name", "date_of_birth"]
    return all(field in data for field in required)

def validate_ecg_signal(signal) -> bool:
    return signal is not None and len(signal) > 0

def validate_analysis_request(request) -> bool:
    return request is not None

def validate_patient_info(info) -> bool:
    return info is not None
