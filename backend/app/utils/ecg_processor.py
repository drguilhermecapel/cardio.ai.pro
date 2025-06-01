"""
ECG signal processing utilities.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import neurokit2 as nk
import numpy as np

from app.core.exceptions import ECGProcessingException

logger = logging.getLogger(__name__)


class ECGProcessor:
    """ECG signal processing and file handling."""

    async def load_ecg_file(self, file_path: str) -> np.ndarray:
        """Load ECG data from file."""
        try:
            path = Path(file_path)
            if not path.exists():
                raise ECGProcessingException(f"File not found: {file_path}")

            if path.suffix.lower() == '.csv':
                return await self._load_csv(file_path)
            elif path.suffix.lower() in ['.txt', '.dat']:
                return await self._load_text(file_path)
            elif path.suffix.lower() == '.xml':
                return await self._load_xml(file_path)
            else:
                raise ECGProcessingException(f"Unsupported file format: {path.suffix}")

        except Exception as e:
            logger.error(f"Failed to load ECG file {file_path}: {str(e)}")
            raise ECGProcessingException(f"Failed to load ECG file: {str(e)}") from e

    async def _load_csv(self, file_path: str) -> np.ndarray:
        """Load ECG data from CSV file."""
        try:
            data = np.loadtxt(file_path, delimiter=',', skiprows=1)
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            return data
        except Exception as e:
            raise ECGProcessingException(f"Failed to load CSV file: {str(e)}") from e

    async def _load_text(self, file_path: str) -> np.ndarray:
        """Load ECG data from text file."""
        try:
            data = np.loadtxt(file_path)
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            return data
        except Exception as e:
            raise ECGProcessingException(f"Failed to load text file: {str(e)}") from e

    async def _load_xml(self, file_path: str) -> np.ndarray:
        """Load ECG data from XML file (simplified implementation)."""
        try:
            import xml.etree.ElementTree as ET

            tree = ET.parse(file_path)
            root = tree.getroot()

            data_elements = root.findall('.//waveform/data')
            if not data_elements:
                raise ECGProcessingException("No waveform data found in XML")

            data_text = data_elements[0].text
            if data_text:
                values = [float(x) for x in data_text.split()]
                data = np.array(values).reshape(-1, 1)
                return data
            else:
                raise ECGProcessingException("Empty waveform data in XML")

        except Exception as e:
            raise ECGProcessingException(f"Failed to load XML file: {str(e)}") from e

    async def extract_metadata(self, file_path: str) -> dict[str, Any]:
        """Extract metadata from ECG file."""
        try:
            path = Path(file_path)
            metadata = {
                "acquisition_date": datetime.utcnow(),
                "sample_rate": 500,  # Default
                "duration_seconds": 10.0,  # Default
                "leads_count": 12,  # Default
                "leads_names": ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
            }

            if path.suffix.lower() == '.xml':
                xml_metadata = await self._extract_xml_metadata(file_path)
                metadata.update(xml_metadata)

            data = await self.load_ecg_file(file_path)
            metadata["leads_count"] = data.shape[1]
            metadata["duration_seconds"] = data.shape[0] / metadata["sample_rate"]

            return metadata

        except Exception as e:
            logger.error(f"Failed to extract metadata from {file_path}: {str(e)}")
            return {
                "acquisition_date": datetime.utcnow(),
                "sample_rate": 500,
                "duration_seconds": 10.0,
                "leads_count": 12,
                "leads_names": ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
            }

    async def _extract_xml_metadata(self, file_path: str) -> dict[str, Any]:
        """Extract metadata from XML file."""
        try:
            import xml.etree.ElementTree as ET

            tree = ET.parse(file_path)
            root = tree.getroot()

            metadata = {}

            sample_rate_elem = root.find('.//sampleRate')
            if sample_rate_elem is not None:
                metadata["sample_rate"] = int(sample_rate_elem.text)

            date_elem = root.find('.//acquisitionDate')
            if date_elem is not None:
                try:
                    metadata["acquisition_date"] = datetime.fromisoformat(date_elem.text)
                except Exception:
                    pass

            device_elem = root.find('.//device')
            if device_elem is not None:
                manufacturer = device_elem.find('manufacturer')
                if manufacturer is not None:
                    metadata["device_manufacturer"] = manufacturer.text

                model = device_elem.find('model')
                if model is not None:
                    metadata["device_model"] = model.text

                serial = device_elem.find('serialNumber')
                if serial is not None:
                    metadata["device_serial"] = serial.text

            return metadata

        except Exception as e:
            logger.error(f"Failed to extract XML metadata: {str(e)}")
            return {}

    async def preprocess_signal(self, ecg_data: np.ndarray) -> np.ndarray:
        """Preprocess ECG signal for analysis."""
        try:
            processed_data = ecg_data.copy()

            for i in range(processed_data.shape[1]):
                lead_data = processed_data[:, i]

                cleaned_signal = nk.ecg_clean(lead_data, sampling_rate=500)
                processed_data[:, i] = cleaned_signal

            return processed_data

        except Exception as e:
            logger.error(f"Signal preprocessing failed: {str(e)}")
            return ecg_data
