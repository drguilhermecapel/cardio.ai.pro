import asyncio
import json
import logging
import time
from typing import Any, Dict, List

import numpy as np
import numpy.typing as npt
import redis
from celery import current_task

from app.core.celery import celery_app
from app.core.config import settings
from app.db.session import get_session_factory

# from app.repositories.ecg_repository import ECGRepository  # Reserved for future use
from app.services.ecg_service import ECGAnalysisService
from app.services.ml_model_service import MLModelService
from app.services.validation_service import ValidationService

logger = logging.getLogger(__name__)

redis_client = redis.Redis.from_url(settings.REDIS_URL)


class StreamingECGProcessor:
    """Real-time ECG streaming processor with <10ms latency"""
    
    def __init__(self):
        self.processing_buffer: Dict[str, List[float]] = {}
        self.last_analysis_time: Dict[str, float] = {}
        
    async def process_streaming_sample(
        self, 
        session_id: str, 
        sample_data: List[float],
        timestamp: float,
        lead_name: str = "II"
    ) -> Dict[str, Any]:
        """Process single ECG sample with <10ms latency"""
        start_time = time.perf_counter()
        
        try:
            if session_id not in self.processing_buffer:
                self.processing_buffer[session_id] = []
                self.last_analysis_time[session_id] = timestamp
                
            self.processing_buffer[session_id].extend(sample_data)
            
            max_buffer_size = 5000
            if len(self.processing_buffer[session_id]) > max_buffer_size:
                self.processing_buffer[session_id] = self.processing_buffer[session_id][-max_buffer_size:]
                
            result = await self._quick_analysis(
                session_id, 
                sample_data, 
                timestamp,
                lead_name
            )
            
            processing_time_ms = (time.perf_counter() - start_time) * 1000
            result["processing_latency_ms"] = processing_time_ms
            
            await self._publish_streaming_result(session_id, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Streaming sample processing failed: {e}")
            return {
                "error": str(e),
                "processing_latency_ms": (time.perf_counter() - start_time) * 1000
            }
            
    async def _quick_analysis(
        self, 
        session_id: str, 
        sample_data: List[float],
        timestamp: float,
        lead_name: str
    ) -> Dict[str, Any]:
        """Perform quick analysis optimized for <10ms latency"""
        import numpy as np
        
        buffer = self.processing_buffer[session_id]
        
        if len(buffer) >= 100:  # Need at least 100 samples for analysis
            recent_samples = np.array(buffer[-100:])
            
            signal_quality = self._assess_signal_quality(recent_samples)
            
            rhythm_info = self._detect_basic_rhythm(recent_samples)
            
            artifacts = self._detect_artifacts(recent_samples)
            
            return {
                "session_id": session_id,
                "timestamp": timestamp,
                "lead": lead_name,
                "signal_quality": signal_quality,
                "rhythm": rhythm_info,
                "artifacts": artifacts,
                "buffer_size": len(buffer),
                "analysis_type": "real_time_quick"
            }
        else:
            return {
                "session_id": session_id,
                "timestamp": timestamp,
                "lead": lead_name,
                "status": "buffering",
                "buffer_size": len(buffer),
                "samples_needed": 100 - len(buffer)
            }
            
    def _assess_signal_quality(self, samples: npt.NDArray[np.float64]) -> Dict[str, Any]:
        """Quick signal quality assessment"""
        import numpy as np
        
        signal_power = np.var(samples)
        noise_estimate = np.var(np.diff(samples))  # High-frequency noise
        snr = signal_power / noise_estimate if noise_estimate > 0 else float('inf')
        
        baseline_drift = np.abs(np.mean(samples[:50]) - np.mean(samples[-50:]))
        
        quality_score = min(1.0, snr / 100.0) * (1.0 - min(1.0, baseline_drift / 100.0))
        
        return {
            "score": float(quality_score),
            "snr": float(snr),
            "baseline_drift": float(baseline_drift),
            "quality_level": "good" if quality_score > 0.8 else "fair" if quality_score > 0.5 else "poor"
        }
        
    def _detect_basic_rhythm(self, samples: npt.NDArray[np.float64]) -> Dict[str, Any]:
        """Basic rhythm detection for real-time feedback"""
        import numpy as np
        from scipy import signal
        
        peaks, _ = signal.find_peaks(samples, height=np.std(samples), distance=50)
        
        if len(peaks) >= 2:
            intervals = np.diff(peaks) / 500.0  # Assuming 500Hz sampling
            heart_rate = 60.0 / np.mean(intervals) if len(intervals) > 0 else 0
            
            rr_variability = np.std(intervals) / np.mean(intervals) if len(intervals) > 0 else 0
            
            return {
                "heart_rate": float(heart_rate),
                "rr_variability": float(rr_variability),
                "rhythm_regularity": "regular" if rr_variability < 0.1 else "irregular",
                "peaks_detected": len(peaks)
            }
        else:
            return {
                "heart_rate": 0,
                "status": "insufficient_data",
                "peaks_detected": len(peaks)
            }
            
    def _detect_artifacts(self, samples: npt.NDArray[np.float64]) -> Dict[str, Any]:
        """Quick artifact detection"""
        import numpy as np
        
        amplitude_threshold = 5 * np.std(samples)
        amplitude_artifacts = np.sum(np.abs(samples) > amplitude_threshold)
        
        diff_threshold = 3 * np.std(np.diff(samples))
        sudden_changes = np.sum(np.abs(np.diff(samples)) > diff_threshold)
        
        return {
            "amplitude_artifacts": int(amplitude_artifacts),
            "sudden_changes": int(sudden_changes),
            "artifact_level": "high" if amplitude_artifacts > 5 or sudden_changes > 3 else "low"
        }
        
    async def _publish_streaming_result(self, session_id: str, result: Dict[str, Any]) -> None:
        """Publish result to WebSocket channel via Redis"""
        try:
            channel = f"ecg_stream:{session_id}"
            message = json.dumps(result)
            redis_client.publish(channel, message)
        except Exception as e:
            logger.error(f"Failed to publish streaming result: {e}")


streaming_processor = StreamingECGProcessor()

@celery_app.task(bind=True)
def process_ecg_analysis(self: Any, analysis_id: int) -> dict[str, Any]:
    """Process ECG analysis in background"""
    try:
        current_task.update_state(
            state="PROGRESS",
            meta={"current": 0, "total": 100, "status": "Starting analysis..."}
        )

        async def _process() -> dict[str, Any]:
            session_factory = get_session_factory()
            async with session_factory() as db:
                # ecg_repo = ECGRepository(db)  # Reserved for future use
                ml_service = MLModelService()
                from app.services.notification_service import NotificationService
                notification_service = NotificationService(db)
                validation_service = ValidationService(db, notification_service)
                service = ECGAnalysisService(db, ml_service, validation_service)
                await service._process_analysis_async(analysis_id)
                return {"status": "completed", "analysis_id": analysis_id}

        result = asyncio.run(_process())

        current_task.update_state(
            state="SUCCESS",
            meta={"current": 100, "total": 100, "status": "Analysis complete", "result": result}
        )

        return result

    except Exception as exc:
        logger.error("ECG analysis task failed: %s", exc)
        current_task.update_state(
            state="FAILURE",
            meta={"current": 0, "total": 100, "status": str(exc)}
        )
        raise


@celery_app.task(bind=True)
def process_streaming_ecg_sample(
    self: Any, 
    session_id: str, 
    sample_data: List[float],
    timestamp: float,
    lead_name: str = "II"
) -> Dict[str, Any]:
    """
    Process single ECG sample for real-time streaming
    
    Args:
        session_id: Unique session identifier
        sample_data: ECG sample values
        timestamp: Sample timestamp
        lead_name: ECG lead name
        
    Returns:
        Real-time analysis results
    """
    try:
        result = asyncio.run(
            streaming_processor.process_streaming_sample(
                session_id, sample_data, timestamp, lead_name
            )
        )
        
        return result
        
    except Exception as exc:
        logger.error(f"Streaming ECG sample processing failed: {str(exc)}")
        return {
            "error": str(exc),
            "session_id": session_id,
            "timestamp": timestamp
        }


@celery_app.task(bind=True)
def process_batch_streaming_samples(
    self: Any,
    session_id: str,
    samples_batch: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Process batch of ECG samples for optimized streaming
    
    Args:
        session_id: Unique session identifier
        samples_batch: List of sample dictionaries with data, timestamp, lead
        
    Returns:
        Batch processing results
    """
    try:
        results = []
        total_latency = 0
        
        for sample in samples_batch:
            result = asyncio.run(
                streaming_processor.process_streaming_sample(
                    session_id,
                    sample["data"],
                    sample["timestamp"],
                    sample.get("lead", "II")
                )
            )
            results.append(result)
            total_latency += result.get("processing_latency_ms", 0)
            
        return {
            "session_id": session_id,
            "batch_size": len(samples_batch),
            "results": results,
            "total_latency_ms": total_latency,
            "avg_latency_ms": total_latency / len(samples_batch) if samples_batch else 0
        }
        
    except Exception as exc:
        logger.error(f"Batch streaming processing failed: {str(exc)}")
        return {
            "error": str(exc),
            "session_id": session_id,
            "batch_size": len(samples_batch) if samples_batch else 0
        }


@celery_app.task(bind=True)
def cleanup_streaming_session(self: Any, session_id: str) -> Dict[str, Any]:
    """
    Clean up streaming session resources
    
    Args:
        session_id: Session to clean up
        
    Returns:
        Cleanup status
    """
    try:
        if session_id in streaming_processor.processing_buffer:
            del streaming_processor.processing_buffer[session_id]
            
        if session_id in streaming_processor.last_analysis_time:
            del streaming_processor.last_analysis_time[session_id]
            
        channel = f"ecg_stream:{session_id}"
        redis_client.delete(channel)
        
        return {
            "status": "SUCCESS",
            "session_id": session_id,
            "message": "Session cleaned up successfully"
        }
        
    except Exception as exc:
        logger.error(f"Session cleanup failed: {str(exc)}")
        return {
            "status": "ERROR",
            "session_id": session_id,
            "error": str(exc)
        }
