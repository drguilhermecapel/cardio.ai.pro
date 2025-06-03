"""
Tests for Wearable Device Integration Service
"""

import asyncio
import json
import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, patch, AsyncMock

import numpy as np

from app.services.wearable_integration_service import (
    WearableDeviceType,
    WearableECGData,
    AppleWatchDataParser,
    FitbitDataParser,
    SamsungGalaxyWatchParser,
    UniversalWearableDataIngestion,
    ContinuousMonitoringService,
    WearableIntegrationService
)


class TestWearableECGData:
    """Test WearableECGData dataclass"""
    
    def test_wearable_ecg_data_creation(self):
        """Test creating WearableECGData instance"""
        ecg_signal = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        timestamp = datetime.now(timezone.utc)
        
        data = WearableECGData(
            device_type=WearableDeviceType.APPLE_WATCH,
            device_id="test_device",
            timestamp=timestamp,
            sampling_rate=512,
            duration_seconds=30.0,
            ecg_signal=ecg_signal,
            leads=["Lead I"],
            metadata={"test": "data"}
        )
        
        assert data.device_type == WearableDeviceType.APPLE_WATCH
        assert data.device_id == "test_device"
        assert data.timestamp == timestamp
        assert data.sampling_rate == 512
        assert data.duration_seconds == 30.0
        assert np.array_equal(data.ecg_signal, ecg_signal)
        assert data.leads == ["Lead I"]
        assert data.metadata == {"test": "data"}


class TestAppleWatchDataParser:
    """Test Apple Watch data parser"""
    
    def setup_method(self):
        self.parser = AppleWatchDataParser()
        
    def test_parse_apple_watch_json_valid(self):
        """Test parsing valid Apple Watch JSON data"""
        json_data = {
            "ecg_samples": [0.1, 0.2, 0.3, 0.4, 0.5],
            "device_id": "apple_watch_123",
            "timestamp": "2023-12-01T10:30:45+00:00",
            "sampling_rate": 512,
            "duration_seconds": 30.0,
            "device_model": "Apple Watch Series 8",
            "classification": "Sinus Rhythm",
            "average_heart_rate": 72
        }
        
        result = self.parser.parse_apple_watch_json(json_data)
        
        assert result.device_type == WearableDeviceType.APPLE_WATCH
        assert result.device_id == "apple_watch_123"
        assert result.sampling_rate == 512
        assert result.duration_seconds == 30.0
        assert len(result.ecg_signal) == 5
        assert result.leads == ["Lead I"]
        assert result.heart_rate == 72
        assert result.rhythm_classification == "Sinus Rhythm"
        
    def test_parse_apple_watch_json_missing_samples(self):
        """Test parsing Apple Watch JSON with missing ECG samples"""
        json_data = {
            "device_id": "apple_watch_123",
            "timestamp": "2023-12-01T10:30:45+00:00"
        }
        
        with pytest.raises(ValueError, match="No ECG samples found"):
            self.parser.parse_apple_watch_json(json_data)
            
    def test_parse_apple_watch_json_invalid_data(self):
        """Test parsing invalid Apple Watch JSON data"""
        json_data = {
            "ecg_samples": "invalid_data"
        }
        
        with pytest.raises(ValueError, match="Invalid Apple Watch data format"):
            self.parser.parse_apple_watch_json(json_data)
            
    @patch('app.services.wearable_integration_service.XML_AVAILABLE', True)
    def test_parse_healthkit_xml_no_records(self):
        """Test parsing HealthKit XML with no ECG records"""
        xml_data = """<?xml version="1.0" encoding="UTF-8"?>
        <HealthData>
            <Record type="HKQuantityTypeIdentifierHeartRate"/>
        </HealthData>"""
        
        result = self.parser.parse_healthkit_xml(xml_data)
        assert result == []
        
    @patch('app.services.wearable_integration_service.XML_AVAILABLE', False)
    def test_parse_healthkit_xml_unavailable(self):
        """Test parsing HealthKit XML when XML is unavailable"""
        xml_data = "<xml>test</xml>"
        
        with pytest.raises(ValueError, match="XML parsing not available"):
            self.parser.parse_healthkit_xml(xml_data)


class TestFitbitDataParser:
    """Test Fitbit data parser"""
    
    def setup_method(self):
        self.parser = FitbitDataParser()
        
    def test_parse_fitbit_json_valid(self):
        """Test parsing valid Fitbit JSON data"""
        json_data = {
            "ecg_readings": [0.1, 0.2, 0.3, 0.4, 0.5],
            "device_id": "fitbit_456",
            "timestamp": "2023-12-01T10:30:45+00:00",
            "sampling_rate": 300,
            "duration": 30.0,
            "device": "Fitbit Sense",
            "quality_score": 0.85,
            "heart_rate": 68
        }
        
        result = self.parser.parse_fitbit_json(json_data)
        
        assert result.device_type == WearableDeviceType.FITBIT
        assert result.device_id == "fitbit_456"
        assert result.sampling_rate == 300
        assert result.duration_seconds == 30.0
        assert len(result.ecg_signal) == 5
        assert result.leads == ["Lead I"]
        assert result.quality_score == 0.85
        assert result.heart_rate == 68
        
    def test_parse_fitbit_json_missing_readings(self):
        """Test parsing Fitbit JSON with missing ECG readings"""
        json_data = {
            "device_id": "fitbit_456",
            "timestamp": "2023-12-01T10:30:45+00:00"
        }
        
        with pytest.raises(ValueError, match="No ECG readings found"):
            self.parser.parse_fitbit_json(json_data)


class TestSamsungGalaxyWatchParser:
    """Test Samsung Galaxy Watch data parser"""
    
    def setup_method(self):
        self.parser = SamsungGalaxyWatchParser()
        
    def test_parse_samsung_health_json_valid(self):
        """Test parsing valid Samsung Health JSON data"""
        json_data = {
            "ecg_data": [0.1, 0.2, 0.3, 0.4, 0.5],
            "device_id": "galaxy_watch_789",
            "start_time": "2023-12-01T10:30:45+00:00",
            "sampling_rate": 500,
            "duration_seconds": 30.0,
            "device_model": "Galaxy Watch 4",
            "signal_quality": 0.9,
            "average_bpm": 75,
            "rhythm_result": "Normal"
        }
        
        result = self.parser.parse_samsung_health_json(json_data)
        
        assert result.device_type == WearableDeviceType.SAMSUNG_GALAXY_WATCH
        assert result.device_id == "galaxy_watch_789"
        assert result.sampling_rate == 500
        assert result.duration_seconds == 30.0
        assert len(result.ecg_signal) == 5
        assert result.leads == ["Lead I"]
        assert result.quality_score == 0.9
        assert result.heart_rate == 75
        assert result.rhythm_classification == "Normal"
        
    def test_parse_samsung_health_json_timestamp_int(self):
        """Test parsing Samsung Health JSON with integer timestamp"""
        json_data = {
            "ecg_data": [0.1, 0.2, 0.3],
            "device_id": "galaxy_watch_789",
            "start_time": 1701425445000,  # milliseconds
            "sampling_rate": 500,
            "duration_seconds": 30.0
        }
        
        result = self.parser.parse_samsung_health_json(json_data)
        assert result.device_id == "galaxy_watch_789"
        assert isinstance(result.timestamp, datetime)
        
    def test_parse_samsung_health_json_missing_data(self):
        """Test parsing Samsung Health JSON with missing ECG data"""
        json_data = {
            "device_id": "galaxy_watch_789",
            "start_time": "2023-12-01T10:30:45+00:00"
        }
        
        with pytest.raises(ValueError, match="No ECG data found"):
            self.parser.parse_samsung_health_json(json_data)


class TestUniversalWearableDataIngestion:
    """Test universal wearable data ingestion pipeline"""
    
    def setup_method(self):
        self.ingestion = UniversalWearableDataIngestion()
        
    @pytest.mark.asyncio
    async def test_ingest_apple_watch_json(self):
        """Test ingesting Apple Watch JSON data"""
        data = {
            "ecg_samples": [0.1, 0.2, 0.3],
            "device_id": "apple_watch_123",
            "timestamp": "2023-12-01T10:30:45+00:00",
            "sampling_rate": 512,
            "duration_seconds": 30.0
        }
        
        result = await self.ingestion.ingest_wearable_data(
            data=data,
            device_type=WearableDeviceType.APPLE_WATCH
        )
        
        assert len(result) == 1
        assert result[0].device_type == WearableDeviceType.APPLE_WATCH
        
    @pytest.mark.asyncio
    async def test_ingest_fitbit_json(self):
        """Test ingesting Fitbit JSON data"""
        data = {
            "ecg_readings": [0.1, 0.2, 0.3],
            "device_id": "fitbit_456",
            "timestamp": "2023-12-01T10:30:45+00:00",
            "sampling_rate": 300,
            "duration": 30.0
        }
        
        result = await self.ingestion.ingest_wearable_data(
            data=data,
            device_type=WearableDeviceType.FITBIT
        )
        
        assert len(result) == 1
        assert result[0].device_type == WearableDeviceType.FITBIT
        
    @pytest.mark.asyncio
    async def test_ingest_samsung_json(self):
        """Test ingesting Samsung Galaxy Watch JSON data"""
        data = {
            "ecg_data": [0.1, 0.2, 0.3],
            "device_id": "galaxy_watch_789",
            "start_time": "2023-12-01T10:30:45+00:00",
            "sampling_rate": 500,
            "duration_seconds": 30.0
        }
        
        result = await self.ingestion.ingest_wearable_data(
            data=data,
            device_type=WearableDeviceType.SAMSUNG_GALAXY_WATCH
        )
        
        assert len(result) == 1
        assert result[0].device_type == WearableDeviceType.SAMSUNG_GALAXY_WATCH
        
    @pytest.mark.asyncio
    async def test_ingest_unsupported_device(self):
        """Test ingesting data from unsupported device type"""
        data = {"test": "data"}
        
        result = await self.ingestion.ingest_wearable_data(
            data=data,
            device_type=WearableDeviceType.UNKNOWN
        )
        
        assert result == []
        
    def test_detect_device_type_apple_watch(self):
        """Test auto-detection of Apple Watch device type"""
        data = {"ecg_samples": [0.1, 0.2, 0.3]}
        
        device_type = self.ingestion._detect_device_type(data)
        assert device_type == WearableDeviceType.APPLE_WATCH
        
    def test_detect_device_type_fitbit(self):
        """Test auto-detection of Fitbit device type"""
        data = {"ecg_readings": [0.1, 0.2, 0.3]}
        
        device_type = self.ingestion._detect_device_type(data)
        assert device_type == WearableDeviceType.FITBIT
        
    def test_detect_device_type_samsung(self):
        """Test auto-detection of Samsung Galaxy Watch device type"""
        data = {"ecg_data": [0.1, 0.2, 0.3]}
        
        device_type = self.ingestion._detect_device_type(data)
        assert device_type == WearableDeviceType.SAMSUNG_GALAXY_WATCH
        
    def test_detect_device_type_unknown(self):
        """Test auto-detection with unknown device type"""
        data = {"unknown_field": [0.1, 0.2, 0.3]}
        
        device_type = self.ingestion._detect_device_type(data)
        assert device_type == WearableDeviceType.UNKNOWN
        
    def test_standardize_ecg_data_apple_watch(self):
        """Test standardizing Apple Watch ECG data"""
        ecg_signal = np.array([100, 200, 300], dtype=np.float32)  # microvolts
        
        wearable_data = WearableECGData(
            device_type=WearableDeviceType.APPLE_WATCH,
            device_id="test",
            timestamp=datetime.now(timezone.utc),
            sampling_rate=512,
            duration_seconds=30.0,
            ecg_signal=ecg_signal,
            leads=["Lead I"],
            metadata={}
        )
        
        standardized = self.ingestion.standardize_ecg_data(wearable_data)
        
        expected = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        np.testing.assert_array_almost_equal(standardized, expected)
        
    def test_standardize_ecg_data_fitbit(self):
        """Test standardizing Fitbit ECG data"""
        ecg_signal = np.array([1000, 2000, 3000], dtype=np.float32)
        
        wearable_data = WearableECGData(
            device_type=WearableDeviceType.FITBIT,
            device_id="test",
            timestamp=datetime.now(timezone.utc),
            sampling_rate=300,
            duration_seconds=30.0,
            ecg_signal=ecg_signal,
            leads=["Lead I"],
            metadata={}
        )
        
        standardized = self.ingestion.standardize_ecg_data(wearable_data)
        
        assert len(standardized) > len(ecg_signal)  # Upsampled
        assert standardized.dtype == np.float32
        
    def test_resample_signal(self):
        """Test signal resampling"""
        signal = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        
        resampled = self.ingestion._resample_signal(signal, 100, 200)
        
        assert len(resampled) == 10
        assert resampled.dtype == np.float32


class TestContinuousMonitoringService:
    """Test continuous monitoring service"""
    
    def setup_method(self):
        self.monitoring = ContinuousMonitoringService()
        
    @pytest.mark.asyncio
    async def test_start_continuous_monitoring(self):
        """Test starting continuous monitoring"""
        monitor_id = await self.monitoring.start_continuous_monitoring(
            device_id="test_device",
            device_type=WearableDeviceType.APPLE_WATCH,
            monitoring_interval=1
        )
        
        assert monitor_id in self.monitoring.active_monitors
        assert monitor_id in self.monitoring.data_buffer
        
        await self.monitoring.stop_continuous_monitoring(monitor_id)
        
    @pytest.mark.asyncio
    async def test_stop_continuous_monitoring(self):
        """Test stopping continuous monitoring"""
        monitor_id = await self.monitoring.start_continuous_monitoring(
            device_id="test_device",
            device_type=WearableDeviceType.APPLE_WATCH,
            monitoring_interval=1
        )
        
        result = await self.monitoring.stop_continuous_monitoring(monitor_id)
        
        assert result is True
        assert monitor_id not in self.monitoring.active_monitors
        assert monitor_id not in self.monitoring.data_buffer
        
    @pytest.mark.asyncio
    async def test_stop_nonexistent_monitoring(self):
        """Test stopping non-existent monitoring"""
        result = await self.monitoring.stop_continuous_monitoring("nonexistent")
        assert result is False
        
    def test_get_monitoring_status(self):
        """Test getting monitoring status"""
        status = self.monitoring.get_monitoring_status()
        
        assert "active_monitors" in status
        assert "monitor_ids" in status
        assert "buffer_sizes" in status
        assert isinstance(status["active_monitors"], int)
        assert isinstance(status["monitor_ids"], list)
        assert isinstance(status["buffer_sizes"], dict)
        
    def test_generate_simulated_data(self):
        """Test generating simulated ECG data"""
        data = self.monitoring._generate_simulated_data(
            WearableDeviceType.APPLE_WATCH,
            "test_device"
        )
        
        assert data is not None
        assert data.device_type == WearableDeviceType.APPLE_WATCH
        assert data.device_id == "test_device"
        assert len(data.ecg_signal) > 0
        assert data.sampling_rate == 500
        assert data.duration_seconds == 30.0
        assert data.metadata["simulated"] is True


class TestWearableIntegrationService:
    """Test main wearable integration service"""
    
    def setup_method(self):
        self.service = WearableIntegrationService()
        
    @pytest.mark.asyncio
    async def test_process_wearable_ecg_apple_watch(self):
        """Test processing Apple Watch ECG data"""
        data = {
            "ecg_samples": [0.1, 0.2, 0.3, 0.4, 0.5],
            "device_id": "apple_watch_123",
            "timestamp": "2023-12-01T10:30:45+00:00",
            "sampling_rate": 512,
            "duration_seconds": 30.0
        }
        
        results = await self.service.process_wearable_ecg(
            data=data,
            device_type=WearableDeviceType.APPLE_WATCH
        )
        
        assert len(results) == 1
        result = results[0]
        assert result["device_type"] == "apple_watch"
        assert result["device_id"] == "apple_watch_123"
        assert result["standardized"] is True
        assert len(result["ecg_signal"]) == 5
        
    @pytest.mark.asyncio
    async def test_process_wearable_ecg_auto_detect(self):
        """Test processing ECG data with auto-detection"""
        data = {
            "ecg_readings": [0.1, 0.2, 0.3],
            "device_id": "fitbit_456",
            "timestamp": "2023-12-01T10:30:45+00:00"
        }
        
        results = await self.service.process_wearable_ecg(data=data)
        
        assert len(results) == 1
        assert results[0]["device_type"] == "fitbit"
        
    @pytest.mark.asyncio
    async def test_process_wearable_ecg_invalid_data(self):
        """Test processing invalid ECG data"""
        data = {"invalid": "data"}
        
        results = await self.service.process_wearable_ecg(data=data)
        assert results == []
        
    @pytest.mark.asyncio
    async def test_start_device_monitoring(self):
        """Test starting device monitoring"""
        monitor_id = await self.service.start_device_monitoring(
            device_id="test_device",
            device_type=WearableDeviceType.APPLE_WATCH,
            monitoring_interval=1
        )
        
        assert isinstance(monitor_id, str)
        
        await self.service.stop_device_monitoring(monitor_id)
        
    @pytest.mark.asyncio
    async def test_stop_device_monitoring(self):
        """Test stopping device monitoring"""
        monitor_id = await self.service.start_device_monitoring(
            device_id="test_device",
            device_type=WearableDeviceType.APPLE_WATCH,
            monitoring_interval=1
        )
        
        result = await self.service.stop_device_monitoring(monitor_id)
        assert result is True
        
    def test_get_monitoring_status(self):
        """Test getting monitoring status"""
        status = self.service.get_monitoring_status()
        
        assert "active_monitors" in status
        assert "monitor_ids" in status
        assert "buffer_sizes" in status
        
    def test_get_supported_devices(self):
        """Test getting supported devices list"""
        devices = self.service.get_supported_devices()
        
        assert len(devices) == 3
        device_types = [device["device_type"] for device in devices]
        assert "apple_watch" in device_types
        assert "fitbit" in device_types
        assert "samsung_galaxy_watch" in device_types
        
        apple_watch = next(d for d in devices if d["device_type"] == "apple_watch")
        assert "name" in apple_watch
        assert "supported_formats" in apple_watch
        assert "sampling_rates" in apple_watch
        assert "leads" in apple_watch
        assert "typical_duration" in apple_watch


@pytest.mark.asyncio
async def test_integration_workflow():
    """Test complete integration workflow"""
    service = WearableIntegrationService()
    
    apple_data = {
        "ecg_samples": np.random.normal(0, 0.1, 1000).tolist(),
        "device_id": "apple_watch_integration_test",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sampling_rate": 512,
        "duration_seconds": 30.0,
        "classification": "Sinus Rhythm",
        "average_heart_rate": 72
    }
    
    results = await service.process_wearable_ecg(
        data=apple_data,
        device_type=WearableDeviceType.APPLE_WATCH
    )
    
    assert len(results) == 1
    result = results[0]
    assert result["device_type"] == "apple_watch"
    assert result["standardized"] is True
    assert len(result["ecg_signal"]) == 1000
    
    monitor_id = await service.start_device_monitoring(
        device_id="integration_test_device",
        device_type=WearableDeviceType.APPLE_WATCH,
        monitoring_interval=1
    )
    
    await asyncio.sleep(0.1)
    
    status = service.get_monitoring_status()
    assert status["active_monitors"] >= 1
    assert monitor_id in status["monitor_ids"]
    
    stopped = await service.stop_device_monitoring(monitor_id)
    assert stopped is True
    
    final_status = service.get_monitoring_status()
    assert monitor_id not in final_status["monitor_ids"]
