"""
Batch ECG Processing Service

This module provides advanced batch processing capabilities for ECG document scanning,
including parallel processing, failure detection, retry logic, and quality reporting.
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from app.core.config import settings
from app.core.exceptions import ECGProcessingException
from app.services.ecg_document_scanner import ECGDocumentScanner


@dataclass
class BatchProcessingResult:
    """Result container for batch processing operations."""
    
    input_path: str
    success: bool
    confidence: float = 0.0
    processing_time: float = 0.0
    error_message: Optional[str] = None
    retry_count: int = 0
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class BatchQualityReport:
    """Quality report for batch processing operations."""
    
    total_processed: int
    successful: int
    failed: int
    success_rate: float
    average_confidence: float
    average_processing_time: float
    total_processing_time: float
    high_quality_count: int  # confidence > 0.8
    medium_quality_count: int  # 0.5 < confidence <= 0.8
    low_quality_count: int  # confidence <= 0.5
    retry_statistics: Dict[str, int]
    error_categories: Dict[str, int]


class BatchECGProcessor:
    """
    Advanced batch processing service for ECG document scanning.
    
    Features:
    - Parallel processing with configurable worker count
    - Exponential backoff retry logic
    - Comprehensive quality reporting
    - Progress tracking and monitoring
    - Error categorization and analysis
    """

    def __init__(
        self,
        max_workers: Optional[int] = None,
        max_retries: int = 3,
        base_retry_delay: float = 1.0,
        timeout_seconds: Optional[int] = None
    ):
        """
        Initialize batch processor.
        
        Args:
            max_workers: Maximum number of parallel workers
            max_retries: Maximum retry attempts per image
            base_retry_delay: Base delay for exponential backoff
            timeout_seconds: Timeout for individual image processing
        """
        self.max_workers = max_workers or settings.ECG_SCANNER_MAX_WORKERS
        self.max_retries = max_retries
        self.base_retry_delay = base_retry_delay
        self.timeout_seconds = timeout_seconds or settings.ECG_SCANNER_TIMEOUT_SECONDS
        
        self.scanner = ECGDocumentScanner()
        self.logger = logging.getLogger(__name__)
        
        self._reset_statistics()

    def _reset_statistics(self) -> None:
        """Reset internal statistics."""
        self.total_processed = 0
        self.successful_processed = 0
        self.failed_processed = 0
        self.retry_counts = {str(i): 0 for i in range(self.max_retries + 1)}
        self.error_categories = {}
        self.processing_times = []

    async def process_batch(
        self,
        image_paths: List[str],
        output_directory: Optional[str] = None,
        progress_callback: Optional[callable] = None
    ) -> List[BatchProcessingResult]:
        """
        Process a batch of ECG images with parallel processing and retry logic.
        
        Args:
            image_paths: List of image file paths to process
            output_directory: Optional directory for processed images
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of BatchProcessingResult objects
        """
        self.logger.info(f"Starting batch processing of {len(image_paths)} images")
        self._reset_statistics()
        
        start_time = time.time()
        results = []
        
        if output_directory:
            Path(output_directory).mkdir(parents=True, exist_ok=True)
        
        batch_size = settings.ECG_SCANNER_BATCH_SIZE
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_results = await self._process_batch_chunk(
                batch_paths, output_directory, progress_callback, i
            )
            results.extend(batch_results)
            
            if progress_callback:
                progress = min(100, (i + len(batch_paths)) / len(image_paths) * 100)
                await progress_callback(progress, len(results))
        
        total_time = time.time() - start_time
        self.logger.info(
            f"Batch processing completed in {total_time:.2f}s. "
            f"Success rate: {self.successful_processed}/{len(image_paths)}"
        )
        
        return results

    async def _process_batch_chunk(
        self,
        image_paths: List[str],
        output_directory: Optional[str],
        progress_callback: Optional[callable],
        batch_offset: int
    ) -> List[BatchProcessingResult]:
        """Process a chunk of images in parallel."""
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_path = {
                executor.submit(
                    self._process_single_image_sync,
                    path,
                    output_directory
                ): path
                for path in image_paths
            }
            
            results = []
            completed = 0
            
            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    result = future.result(timeout=self.timeout_seconds)
                    results.append(result)
                    
                    if result.success:
                        self.successful_processed += 1
                    else:
                        self.failed_processed += 1
                    
                    self.total_processed += 1
                    self.processing_times.append(result.processing_time)
                    
                except Exception as e:
                    self.logger.error(f"Unexpected error processing {path}: {str(e)}")
                    result = BatchProcessingResult(
                        input_path=path,
                        success=False,
                        error_message=f"Unexpected error: {str(e)}"
                    )
                    results.append(result)
                    self.failed_processed += 1
                    self.total_processed += 1
                
                completed += 1
                
                if progress_callback:
                    chunk_progress = completed / len(image_paths) * 100
                    await progress_callback(
                        chunk_progress, 
                        batch_offset + completed
                    )
        
        return results

    def _process_single_image_sync(
        self,
        image_path: str,
        output_directory: Optional[str]
    ) -> BatchProcessingResult:
        """
        Process a single image with retry logic (synchronous version for ThreadPoolExecutor).
        """
        start_time = time.time()
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                output_path = None
                if output_directory:
                    filename = Path(image_path).name
                    output_path = str(Path(output_directory) / f"processed_{filename}")
                
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(
                        self.scanner.process_ecg_image(image_path, output_path)
                    )
                finally:
                    loop.close()
                
                processing_time = time.time() - start_time
                confidence = result.get("confidence", 0.0)
                
                self.retry_counts[str(attempt)] += 1
                
                return BatchProcessingResult(
                    input_path=image_path,
                    success=True,
                    confidence=confidence,
                    processing_time=processing_time,
                    retry_count=attempt,
                    metadata=result.get("metadata", {})
                )
                
            except Exception as e:
                last_error = e
                error_category = self._categorize_error(e)
                self.error_categories[error_category] = (
                    self.error_categories.get(error_category, 0) + 1
                )
                
                if attempt < self.max_retries:
                    delay = self.base_retry_delay * (2 ** attempt)
                    self.logger.warning(
                        f"Attempt {attempt + 1} failed for {image_path}: {str(e)}. "
                        f"Retrying in {delay:.1f}s"
                    )
                    time.sleep(delay)
                else:
                    self.logger.error(
                        f"All {self.max_retries + 1} attempts failed for {image_path}: {str(e)}"
                    )
        
        processing_time = time.time() - start_time
        self.retry_counts[str(self.max_retries)] += 1
        
        return BatchProcessingResult(
            input_path=image_path,
            success=False,
            processing_time=processing_time,
            error_message=str(last_error),
            retry_count=self.max_retries
        )

    def _categorize_error(self, error: Exception) -> str:
        """Categorize errors for reporting."""
        error_str = str(error).lower()
        
        if "file not found" in error_str or "no such file" in error_str:
            return "file_not_found"
        elif "permission" in error_str:
            return "permission_error"
        elif "memory" in error_str or "out of memory" in error_str:
            return "memory_error"
        elif "timeout" in error_str:
            return "timeout_error"
        elif "invalid image" in error_str or "corrupt" in error_str:
            return "invalid_image"
        elif isinstance(error, ECGProcessingException):
            return "ecg_processing_error"
        else:
            return "unknown_error"

    async def generate_quality_report(
        self,
        results: List[BatchProcessingResult]
    ) -> BatchQualityReport:
        """
        Generate comprehensive quality report from batch processing results.
        
        Args:
            results: List of batch processing results
            
        Returns:
            BatchQualityReport with detailed statistics
        """
        if not results:
            return BatchQualityReport(
                total_processed=0,
                successful=0,
                failed=0,
                success_rate=0.0,
                average_confidence=0.0,
                average_processing_time=0.0,
                total_processing_time=0.0,
                high_quality_count=0,
                medium_quality_count=0,
                low_quality_count=0,
                retry_statistics={},
                error_categories={}
            )
        
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]
        
        confidences = [r.confidence for r in successful_results]
        average_confidence = np.mean(confidences) if confidences else 0.0
        
        high_quality = len([c for c in confidences if c > 0.8])
        medium_quality = len([c for c in confidences if 0.5 < c <= 0.8])
        low_quality = len([c for c in confidences if c <= 0.5])
        
        processing_times = [r.processing_time for r in results]
        average_processing_time = np.mean(processing_times) if processing_times else 0.0
        total_processing_time = sum(processing_times)
        
        retry_stats = {}
        for result in results:
            retry_count = str(result.retry_count)
            retry_stats[retry_count] = retry_stats.get(retry_count, 0) + 1
        
        error_categories = {}
        for result in failed_results:
            if result.error_message:
                category = self._categorize_error(Exception(result.error_message))
                error_categories[category] = error_categories.get(category, 0) + 1
        
        return BatchQualityReport(
            total_processed=len(results),
            successful=len(successful_results),
            failed=len(failed_results),
            success_rate=len(successful_results) / len(results) if results else 0.0,
            average_confidence=float(average_confidence),
            average_processing_time=float(average_processing_time),
            total_processing_time=float(total_processing_time),
            high_quality_count=high_quality,
            medium_quality_count=medium_quality,
            low_quality_count=low_quality,
            retry_statistics=retry_stats,
            error_categories=error_categories
        )

    async def process_with_monitoring(
        self,
        image_paths: List[str],
        output_directory: Optional[str] = None,
        report_interval: int = 10
    ) -> Tuple[List[BatchProcessingResult], BatchQualityReport]:
        """
        Process batch with real-time monitoring and reporting.
        
        Args:
            image_paths: List of image paths to process
            output_directory: Optional output directory
            report_interval: Interval for progress reports (in seconds)
            
        Returns:
            Tuple of (results, quality_report)
        """
        
        async def progress_callback(progress: float, completed: int) -> None:
            """Progress callback for monitoring."""
            self.logger.info(
                f"Batch processing progress: {progress:.1f}% "
                f"({completed}/{len(image_paths)} completed)"
            )
        
        results = await self.process_batch(
            image_paths, output_directory, progress_callback
        )
        
        quality_report = await self.generate_quality_report(results)
        
        self.logger.info(
            f"Batch processing completed:\n"
            f"  Total: {quality_report.total_processed}\n"
            f"  Successful: {quality_report.successful}\n"
            f"  Failed: {quality_report.failed}\n"
            f"  Success Rate: {quality_report.success_rate:.2%}\n"
            f"  Average Confidence: {quality_report.average_confidence:.3f}\n"
            f"  Average Processing Time: {quality_report.average_processing_time:.2f}s\n"
            f"  High Quality (>0.8): {quality_report.high_quality_count}\n"
            f"  Medium Quality (0.5-0.8): {quality_report.medium_quality_count}\n"
            f"  Low Quality (≤0.5): {quality_report.low_quality_count}"
        )
        
        return results, quality_report

    async def validate_batch_inputs(self, image_paths: List[str]) -> List[str]:
        """
        Validate batch input files and return list of valid paths.
        
        Args:
            image_paths: List of image paths to validate
            
        Returns:
            List of valid image paths
        """
        valid_paths = []
        invalid_paths = []
        
        for path in image_paths:
            path_obj = Path(path)
            
            if not path_obj.exists():
                invalid_paths.append((path, "File does not exist"))
                continue
            
            valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
            if path_obj.suffix.lower() not in valid_extensions:
                invalid_paths.append((path, "Unsupported file format"))
                continue
            
            try:
                file_size = path_obj.stat().st_size
                if file_size > settings.MAX_ECG_FILE_SIZE:
                    invalid_paths.append((path, "File too large"))
                    continue
                if file_size == 0:
                    invalid_paths.append((path, "Empty file"))
                    continue
            except Exception as e:
                invalid_paths.append((path, f"Cannot access file: {str(e)}"))
                continue
            
            valid_paths.append(path)
        
        if invalid_paths:
            self.logger.warning(
                f"Found {len(invalid_paths)} invalid files out of {len(image_paths)} total:\n" +
                "\n".join([f"  {path}: {reason}" for path, reason in invalid_paths])
            )
        
        self.logger.info(f"Validated {len(valid_paths)} files for batch processing")
        return valid_paths

    async def estimate_processing_time(self, image_paths: List[str]) -> Dict[str, float]:
        """
        Estimate processing time for a batch of images.
        
        Args:
            image_paths: List of image paths
            
        Returns:
            Dictionary with time estimates
        """
        base_time_per_image = 2.0
        
        total_size = 0
        for path in image_paths:
            try:
                total_size += Path(path).stat().st_size
            except Exception:
                continue
        
        avg_size_mb = (total_size / len(image_paths)) / (1024 * 1024) if image_paths else 0
        size_factor = 1.0 + (avg_size_mb / 10.0)  # +10% per 10MB
        
        parallel_factor = min(len(image_paths), self.max_workers) / len(image_paths) if image_paths else 1
        
        estimated_time = (
            len(image_paths) * base_time_per_image * size_factor * parallel_factor
        )
        
        return {
            "estimated_total_time": estimated_time,
            "estimated_time_per_image": base_time_per_image * size_factor,
            "parallelization_factor": parallel_factor,
            "size_factor": size_factor,
            "total_images": len(image_paths)
        }



async def process_directory_batch(
    directory_path: str,
    output_directory: Optional[str] = None,
    file_pattern: str = "*.jpg",
    max_workers: Optional[int] = None
) -> Tuple[List[BatchProcessingResult], BatchQualityReport]:
    """
    Process all images in a directory.
    
    Args:
        directory_path: Directory containing images
        output_directory: Optional output directory
        file_pattern: File pattern to match (e.g., "*.jpg", "*.png")
        max_workers: Maximum parallel workers
        
    Returns:
        Tuple of (results, quality_report)
    """
    directory = Path(directory_path)
    image_paths = [str(p) for p in directory.glob(file_pattern)]
    
    processor = BatchECGProcessor(max_workers=max_workers)
    return await processor.process_with_monitoring(image_paths, output_directory)


async def process_file_list_batch(
    file_list_path: str,
    output_directory: Optional[str] = None,
    max_workers: Optional[int] = None
) -> Tuple[List[BatchProcessingResult], BatchQualityReport]:
    """
    Process images from a file list.
    
    Args:
        file_list_path: Path to text file containing image paths (one per line)
        output_directory: Optional output directory
        max_workers: Maximum parallel workers
        
    Returns:
        Tuple of (results, quality_report)
    """
    with open(file_list_path, 'r') as f:
        image_paths = [line.strip() for line in f if line.strip()]
    
    processor = BatchECGProcessor(max_workers=max_workers)
    return await processor.process_with_monitoring(image_paths, output_directory)
