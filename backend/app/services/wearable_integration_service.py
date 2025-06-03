"""
Wearable Device Integration Service
Universal data ingestion pipeline for ECG data from various wearable devices
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union, Tuple
from enum import Enum
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

try:
    import xml.etree.ElementTree as ET
    XML_AVAILABLE = True
except ImportError:
    XML_AVAILABLE = False
    logger.warning("XML parsing not available. Some wearable data formats may not be supported.")


class WearableDeviceType(Enum):
    """Supported wearable device types"""
    APPLE_WATCH = "apple_watch"
    FITBIT = "fitbit"
    SAMSUNG_GALAXY_WATCH = "samsung_galaxy_watch"
    GARMIN = "garmin"
    POLAR = "polar"
    UNKNOWN = "unknown"


@dataclass
class WearableECGData:
    """Standardized ECG data structure from wearable devices"""
    device_type: WearableDeviceType
    device_id: str
    timestamp: datetime
    sampling_rate: int
    duration_seconds: float
    ecg_signal: npt.NDArray[np.float32]
    leads: List[str]
    metadata: Dict[str, Any]
    quality_score: Optional[float] = None
    heart_rate: Optional[float] = None
    rhythm_classification: Optional[str] = None


class AppleWatchDataParser:
    """Parser for Apple Watch ECG data (HealthKit format)"""
    
    def __init__(self):
        self.supported_formats = ["xml", "json"]
        
    def parse_healthkit_xml(self, xml_data: str) -> List[WearableECGData]:
        """Parse Apple HealthKit XML export data"""
        if not XML_AVAILABLE:
            raise ValueError("XML parsing not available")
            
        try:
            root = ET.fromstring(xml_data)
            ecg_records = []
            
            for record in root.findall(".//Record[@type='HKQuantityTypeIdentifierElectrocardiogram']"):
                device_name = record.get('sourceName', 'Apple Watch')
                creation_date = record.get('creationDate')
                start_date = record.get('startDate')
                end_date = record.get('endDate')
                
                metadata = {
                    'source_name': device_name,
                    'creation_date': creation_date,
                    'start_date': start_date,
                    'end_date': end_date
                }
                
                voltage_samples = []
                for sample in record.findall(".//InstantaneousBeatsPerMinute"):
                    value = sample.get('value')
                    if value:
                        voltage_samples.append(float(value))
                        
                if voltage_samples:
                    ecg_signal = np.array(voltage_samples, dtype=np.float32)
                    
                    sampling_rate = 512
                    duration = len(ecg_signal) / sampling_rate
                    
                    ecg_data = WearableECGData(
                        device_type=WearableDeviceType.APPLE_WATCH,
                        device_id=device_name,
                        timestamp=self._parse_datetime(start_date),
                        sampling_rate=sampling_rate,
                        duration_seconds=duration,
                        ecg_signal=ecg_signal,
                        leads=["Lead I"],  # Apple Watch uses Lead I
                        metadata=metadata
                    )
                    ecg_records.append(ecg_data)
                    
            return ecg_records
            
        except Exception as e:
            logger.error(f"Apple HealthKit XML parsing failed: {e}")
            return []
            
    def parse_apple_watch_json(self, json_data: Dict[str, Any]) -> WearableECGData:
        """Parse Apple Watch ECG data from JSON format"""
        try:
            ecg_samples = json_data.get('ecg_samples', [])
            if not ecg_samples:
                raise ValueError("No ECG samples found in Apple Watch data")
                
            ecg_signal = np.array(ecg_samples, dtype=np.float32)
            
            metadata = {
                'device_model': json_data.get('device_model', 'Apple Watch'),
                'os_version': json_data.get('os_version'),
                'app_version': json_data.get('app_version'),
                'classification': json_data.get('classification'),
                'average_heart_rate': json_data.get('average_heart_rate')
            }
            
            return WearableECGData(
                device_type=WearableDeviceType.APPLE_WATCH,
                device_id=json_data.get('device_id', 'apple_watch_unknown'),
                timestamp=self._parse_datetime(json_data.get('timestamp')),
                sampling_rate=json_data.get('sampling_rate', 512),
                duration_seconds=json_data.get('duration_seconds', 30.0),
                ecg_signal=ecg_signal,
                leads=["Lead I"],
                metadata=metadata,
                heart_rate=json_data.get('average_heart_rate'),
                rhythm_classification=json_data.get('classification')
            )
            
        except Exception as e:
            logger.error(f"Apple Watch JSON parsing failed: {e}")
            raise ValueError(f"Invalid Apple Watch data format: {e}")
            
    def _parse_datetime(self, date_str: Optional[str]) -> datetime:
        """Parse datetime string from Apple HealthKit"""
        if not date_str:
            return datetime.now(timezone.utc)
            
        try:
            return datetime.fromisoformat(date_str.replace(' +0000', '+00:00'))
        except Exception:
            return datetime.now(timezone.utc)


class FitbitDataParser:
    """Parser for Fitbit ECG data"""
    
    def __init__(self):
        self.supported_formats = ["json", "csv"]
        
    def parse_fitbit_json(self, json_data: Dict[str, Any]) -> WearableECGData:
        """Parse Fitbit ECG data from JSON format"""
        try:
            ecg_readings = json_data.get('ecg_readings', [])
            if not ecg_readings:
                raise ValueError("No ECG readings found in Fitbit data")
                
            ecg_signal = np.array(ecg_readings, dtype=np.float32)
            
            metadata = {
                'device_model': json_data.get('device', 'Fitbit'),
                'firmware_version': json_data.get('firmware_version'),
                'battery_level': json_data.get('battery_level'),
                'user_id': json_data.get('user_id'),
                'measurement_quality': json_data.get('quality')
            }
            
            return WearableECGData(
                device_type=WearableDeviceType.FITBIT,
                device_id=json_data.get('device_id', 'fitbit_unknown'),
                timestamp=self._parse_datetime(json_data.get('timestamp')),
                sampling_rate=json_data.get('sampling_rate', 300),  # Fitbit typically 300 Hz
                duration_seconds=json_data.get('duration', 30.0),
                ecg_signal=ecg_signal,
                leads=["Lead I"],  # Fitbit uses single lead
                metadata=metadata,
                quality_score=json_data.get('quality_score'),
                heart_rate=json_data.get('heart_rate')
            )
            
        except Exception as e:
            logger.error(f"Fitbit JSON parsing failed: {e}")
            raise ValueError(f"Invalid Fitbit data format: {e}")
            
    def _parse_datetime(self, date_str: Optional[str]) -> datetime:
        """Parse datetime string from Fitbit"""
        if not date_str:
            return datetime.now(timezone.utc)
            
        try:
            return datetime.fromisoformat(date_str)
        except Exception:
            return datetime.now(timezone.utc)


class SamsungGalaxyWatchParser:
    """Parser for Samsung Galaxy Watch ECG data"""
    
    def __init__(self):
        self.supported_formats = ["json", "samsung_health"]
        
    def parse_samsung_health_json(self, json_data: Dict[str, Any]) -> WearableECGData:
        """Parse Samsung Health ECG data from JSON format"""
        try:
            ecg_data = json_data.get('ecg_data', [])
            if not ecg_data:
                raise ValueError("No ECG data found in Samsung Health export")
                
            ecg_signal = np.array(ecg_data, dtype=np.float32)
            
            metadata = {
                'device_model': json_data.get('device_model', 'Galaxy Watch'),
                'software_version': json_data.get('software_version'),
                'sensor_type': json_data.get('sensor_type'),
                'measurement_environment': json_data.get('environment'),
                'user_profile': json_data.get('user_profile', {})
            }
            
            return WearableECGData(
                device_type=WearableDeviceType.SAMSUNG_GALAXY_WATCH,
                device_id=json_data.get('device_id', 'galaxy_watch_unknown'),
                timestamp=self._parse_datetime(json_data.get('start_time')),
                sampling_rate=json_data.get('sampling_rate', 500),  # Galaxy Watch typically 500 Hz
                duration_seconds=json_data.get('duration_seconds', 30.0),
                ecg_signal=ecg_signal,
                leads=["Lead I"],
                metadata=metadata,
                quality_score=json_data.get('signal_quality'),
                heart_rate=json_data.get('average_bpm'),
                rhythm_classification=json_data.get('rhythm_result')
            )
            
        except Exception as e:
            logger.error(f"Samsung Galaxy Watch JSON parsing failed: {e}")
            raise ValueError(f"Invalid Samsung Galaxy Watch data format: {e}")
            
    def _parse_datetime(self, timestamp: Optional[Union[str, int]]) -> datetime:
        """Parse datetime from Samsung Health"""
        if not timestamp:
            return datetime.now(timezone.utc)
            
        try:
            if isinstance(timestamp, int):
                return datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc)
            else:
                return datetime.fromisoformat(timestamp)
        except Exception:
            return datetime.now(timezone.utc)


class UniversalWearableDataIngestion:
    """Universal data ingestion pipeline for wearable ECG data"""
    
    def __init__(self):
        self.parsers = {
            WearableDeviceType.APPLE_WATCH: AppleWatchDataParser(),
            WearableDeviceType.FITBIT: FitbitDataParser(),
            WearableDeviceType.SAMSUNG_GALAXY_WATCH: SamsungGalaxyWatchParser()
        }
        
    async def ingest_wearable_data(
        self,
        data: Union[str, Dict[str, Any]],
        device_type: Optional[WearableDeviceType] = None,
        data_format: Optional[str] = None
    ) -> List[WearableECGData]:
        """Ingest ECG data from various wearable devices"""
        try:
            if device_type is None:
                device_type = self._detect_device_type(data)
                
            parser = self.parsers.get(device_type)
            if not parser:
                raise ValueError(f"Unsupported device type: {device_type}")
                
            if device_type == WearableDeviceType.APPLE_WATCH:
                if isinstance(data, str) and data_format == "xml":
                    return parser.parse_healthkit_xml(data)
                elif isinstance(data, dict):
                    return [parser.parse_apple_watch_json(data)]
                    
            elif device_type == WearableDeviceType.FITBIT:
                if isinstance(data, dict):
                    return [parser.parse_fitbit_json(data)]
                    
            elif device_type == WearableDeviceType.SAMSUNG_GALAXY_WATCH:
                if isinstance(data, dict):
                    return [parser.parse_samsung_health_json(data)]
                    
            raise ValueError(f"Unsupported data format for {device_type}")
            
        except Exception as e:
            logger.error(f"Wearable data ingestion failed: {e}")
            return []
            
    def _detect_device_type(self, data: Union[str, Dict[str, Any]]) -> WearableDeviceType:
        """Auto-detect wearable device type from data"""
        try:
            if isinstance(data, str):
                if "HealthKit" in data or "HKQuantityTypeIdentifierElectrocardiogram" in data:
                    return WearableDeviceType.APPLE_WATCH
                    
            elif isinstance(data, dict):
                if "device_model" in data and "apple" in data["device_model"].lower():
                    return WearableDeviceType.APPLE_WATCH
                elif "device" in data and "fitbit" in data["device"].lower():
                    return WearableDeviceType.FITBIT
                elif "device_model" in data and "galaxy" in data["device_model"].lower():
                    return WearableDeviceType.SAMSUNG_GALAXY_WATCH
                elif "ecg_samples" in data:
                    return WearableDeviceType.APPLE_WATCH
                elif "ecg_readings" in data:
                    return WearableDeviceType.FITBIT
                elif "ecg_data" in data:
                    return WearableDeviceType.SAMSUNG_GALAXY_WATCH
                    
            return WearableDeviceType.UNKNOWN
            
        except Exception:
            return WearableDeviceType.UNKNOWN
            
    def standardize_ecg_data(self, wearable_data: WearableECGData) -> npt.NDArray[np.float32]:
        """Standardize ECG data for analysis pipeline"""
        try:
            ecg_signal = wearable_data.ecg_signal
            
            if wearable_data.device_type == WearableDeviceType.APPLE_WATCH:
                ecg_signal = ecg_signal / 1000.0
            elif wearable_data.device_type == WearableDeviceType.FITBIT:
                ecg_signal = ecg_signal * 0.001
            elif wearable_data.device_type == WearableDeviceType.SAMSUNG_GALAXY_WATCH:
                pass
                
            ecg_signal = np.clip(ecg_signal, -5.0, 5.0)
            
            if wearable_data.sampling_rate != 500:
                ecg_signal = self._resample_signal(
                    ecg_signal, 
                    wearable_data.sampling_rate, 
                    target_rate=500
                )
                
            return ecg_signal.astype(np.float32)
            
        except Exception as e:
            logger.error(f"ECG data standardization failed: {e}")
            return wearable_data.ecg_signal
            
    def _resample_signal(
        self, 
        signal: npt.NDArray[np.float32], 
        original_rate: int, 
        target_rate: int
    ) -> npt.NDArray[np.float32]:
        """Resample ECG signal to target sampling rate"""
        try:
            original_length = len(signal)
            target_length = int(original_length * target_rate / original_rate)
            
            original_indices = np.linspace(0, original_length - 1, original_length)
            target_indices = np.linspace(0, original_length - 1, target_length)
            
            resampled_signal = np.interp(target_indices, original_indices, signal)
            
            return resampled_signal.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Signal resampling failed: {e}")
            return signal


class ContinuousMonitoringService:
    """Continuous monitoring service for wearable devices"""
    
    def __init__(self):
        self.ingestion_pipeline = UniversalWearableDataIngestion()
        self.active_monitors: Dict[str, asyncio.Task] = {}
        self.data_buffer: Dict[str, List[WearableECGData]] = {}
        
    async def start_continuous_monitoring(
        self,
        device_id: str,
        device_type: WearableDeviceType,
        monitoring_interval: int = 60,  # seconds
        callback: Optional[callable] = None
    ) -> str:
        """Start continuous monitoring for a wearable device"""
        try:
            monitor_id = f"{device_type.value}_{device_id}_{datetime.now().timestamp()}"
            
            monitor_task = asyncio.create_task(
                self._monitor_device(
                    monitor_id, device_id, device_type, 
                    monitoring_interval, callback
                )
            )
            
            self.active_monitors[monitor_id] = monitor_task
            self.data_buffer[monitor_id] = []
            
            logger.info(f"Started continuous monitoring for {device_id} ({device_type.value})")
            return monitor_id
            
        except Exception as e:
            logger.error(f"Failed to start continuous monitoring: {e}")
            raise
            
    async def stop_continuous_monitoring(self, monitor_id: str) -> bool:
        """Stop continuous monitoring for a device"""
        try:
            if monitor_id in self.active_monitors:
                self.active_monitors[monitor_id].cancel()
                del self.active_monitors[monitor_id]
                
                if monitor_id in self.data_buffer:
                    del self.data_buffer[monitor_id]
                    
                logger.info(f"Stopped continuous monitoring for {monitor_id}")
                return True
            else:
                logger.warning(f"Monitor {monitor_id} not found")
                return False
                
        except Exception as e:
            logger.error(f"Failed to stop continuous monitoring: {e}")
            return False
            
    async def _monitor_device(
        self,
        monitor_id: str,
        device_id: str,
        device_type: WearableDeviceType,
        interval: int,
        callback: Optional[callable]
    ) -> None:
        """Monitor device for new ECG data"""
        try:
            while True:
                await asyncio.sleep(interval)
                
                simulated_data = self._generate_simulated_data(device_type, device_id)
                
                if simulated_data:
                    self.data_buffer[monitor_id].append(simulated_data)
                    
                    if callback:
                        try:
                            await callback(simulated_data)
                        except Exception as e:
                            logger.error(f"Monitoring callback failed: {e}")
                            
                if len(self.data_buffer[monitor_id]) > 100:
                    self.data_buffer[monitor_id] = self.data_buffer[monitor_id][-50:]
                    
        except asyncio.CancelledError:
            logger.info(f"Monitoring cancelled for {monitor_id}")
        except Exception as e:
            logger.error(f"Monitoring error for {monitor_id}: {e}")
            
    def _generate_simulated_data(
        self, 
        device_type: WearableDeviceType, 
        device_id: str
    ) -> WearableECGData:
        """Generate simulated ECG data for testing"""
        try:
            duration = 30.0
            sampling_rate = 500
            samples = int(duration * sampling_rate)
            
            t = np.linspace(0, duration, samples)
            heart_rate = 70 + np.random.normal(0, 5)  # BPM with variation
            
            ecg_signal = np.zeros(samples)
            beat_interval = 60.0 / heart_rate  # seconds per beat
            
            for beat_time in np.arange(0, duration, beat_interval):
                beat_idx = int(beat_time * sampling_rate)
                if beat_idx < samples - 50:
                    ecg_signal[beat_idx:beat_idx+50] += np.random.normal(1.0, 0.1, 50)
                    
            ecg_signal += np.random.normal(0, 0.05, samples)
            
            return WearableECGData(
                device_type=device_type,
                device_id=device_id,
                timestamp=datetime.now(timezone.utc),
                sampling_rate=sampling_rate,
                duration_seconds=duration,
                ecg_signal=ecg_signal.astype(np.float32),
                leads=["Lead I"],
                metadata={"simulated": True},
                heart_rate=heart_rate
            )
            
        except Exception as e:
            logger.error(f"Simulated data generation failed: {e}")
            return None
            
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get status of all active monitors"""
        return {
            "active_monitors": len(self.active_monitors),
            "monitor_ids": list(self.active_monitors.keys()),
            "buffer_sizes": {
                monitor_id: len(buffer) 
                for monitor_id, buffer in self.data_buffer.items()
            }
        }


class WearableIntegrationService:
    """Main service for wearable device integration"""
    
    def __init__(self):
        self.ingestion_pipeline = UniversalWearableDataIngestion()
        self.continuous_monitoring = ContinuousMonitoringService()
        
    async def process_wearable_ecg(
        self,
        data: Union[str, Dict[str, Any]],
        device_type: Optional[WearableDeviceType] = None,
        data_format: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Process ECG data from wearable devices"""
        try:
            wearable_data_list = await self.ingestion_pipeline.ingest_wearable_data(
                data=data,
                device_type=device_type,
                data_format=data_format
            )
            
            results = []
            for wearable_data in wearable_data_list:
                standardized_ecg = self.ingestion_pipeline.standardize_ecg_data(wearable_data)
                
                result = {
                    "device_type": wearable_data.device_type.value,
                    "device_id": wearable_data.device_id,
                    "timestamp": wearable_data.timestamp.isoformat(),
                    "sampling_rate": wearable_data.sampling_rate,
                    "duration_seconds": wearable_data.duration_seconds,
                    "ecg_signal": standardized_ecg.tolist(),
                    "leads": wearable_data.leads,
                    "metadata": wearable_data.metadata,
                    "quality_score": wearable_data.quality_score,
                    "heart_rate": wearable_data.heart_rate,
                    "rhythm_classification": wearable_data.rhythm_classification,
                    "standardized": True
                }
                results.append(result)
                
            return results
            
        except Exception as e:
            logger.error(f"Wearable ECG processing failed: {e}")
            return []
            
    async def start_device_monitoring(
        self,
        device_id: str,
        device_type: WearableDeviceType,
        monitoring_interval: int = 60
    ) -> str:
        """Start continuous monitoring for a wearable device"""
        return await self.continuous_monitoring.start_continuous_monitoring(
            device_id=device_id,
            device_type=device_type,
            monitoring_interval=monitoring_interval
        )
        
    async def stop_device_monitoring(self, monitor_id: str) -> bool:
        """Stop continuous monitoring for a device"""
        return await self.continuous_monitoring.stop_continuous_monitoring(monitor_id)
        
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get status of all active device monitors"""
        return self.continuous_monitoring.get_monitoring_status()
        
    def get_supported_devices(self) -> List[Dict[str, Any]]:
        """Get list of supported wearable devices"""
        return [
            {
                "device_type": WearableDeviceType.APPLE_WATCH.value,
                "name": "Apple Watch",
                "supported_formats": ["json", "xml"],
                "sampling_rates": [512],
                "leads": ["Lead I"],
                "typical_duration": 30
            },
            {
                "device_type": WearableDeviceType.FITBIT.value,
                "name": "Fitbit",
                "supported_formats": ["json"],
                "sampling_rates": [300],
                "leads": ["Lead I"],
                "typical_duration": 30
            },
            {
                "device_type": WearableDeviceType.SAMSUNG_GALAXY_WATCH.value,
                "name": "Samsung Galaxy Watch",
                "supported_formats": ["json"],
                "sampling_rates": [500],
                "leads": ["Lead I"],
                "typical_duration": 30
            }
        ]
