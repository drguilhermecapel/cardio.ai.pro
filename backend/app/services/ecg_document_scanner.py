"""
ECG Document Scanner Service

This module provides comprehensive ECG document scanning capabilities including:
- Edge detection and document boundary identification
- Perspective correction and geometric transformation
- Image enhancement and quality optimization
- ECG-specific validation and confidence scoring
- Integration with existing ECG analysis pipeline
"""

import asyncio
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import numpy.typing as npt
from scipy import ndimage
from skimage import filters, measure, morphology, restoration
from skimage.feature import canny
from skimage.transform import hough_line, hough_line_peaks

from app.core.config import settings
from app.core.exceptions import ECGProcessingException, NonECGImageException
from app.services.non_ecg_image_classifier import NonECGImageClassifier
from app.services.contextual_response_generator import ContextualResponseGenerator

logger = logging.getLogger(__name__)


class ECGDocumentScanner:
    """
    Advanced ECG document scanner with computer vision algorithms
    for automatic document detection, perspective correction, and enhancement.
    """

    def __init__(self) -> None:
        """Initialize the ECG document scanner with performance optimizations."""
        self.min_document_area = settings.ECG_SCANNER_MIN_CONTOUR_AREA
        self.max_document_area = 2000000  # Maximum area for valid document
        self.edge_threshold_low = settings.ECG_SCANNER_EDGE_THRESHOLD
        self.edge_threshold_high = settings.ECG_SCANNER_EDGE_THRESHOLD * 2
        self.gaussian_blur_kernel = (5, 5)
        self.morphology_kernel_size = 3
        self.hough_threshold = 100
        self.min_line_length = 100
        self.max_line_gap = 10
        self.perspective_margin = 20  # Pixels to add around detected document
        
        self.max_detection_size = 800  # Max dimension for initial detection
        self.gpu_available = self._check_gpu_availability()
        
        self._transformation_cache: Dict[str, np.ndarray] = {}
        self._cache_max_size = 100
        
        self.non_ecg_classifier = NonECGImageClassifier()
        self.response_generator = ContextualResponseGenerator()
        self.non_ecg_threshold = getattr(settings, 'NON_ECG_CLASSIFICATION_THRESHOLD', 0.7)

    async def process_ecg_image(
        self, 
        image_path: str,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Main pipeline method to process ECG image from photo to clean document.
        
        Args:
            image_path: Path to input image file
            output_path: Optional path to save processed image
            
        Returns:
            Dictionary containing processed image data and metadata
        """
        try:
            image = await self._load_image(image_path)
            if image is None:
                raise ECGProcessingException(f"Failed to load image: {image_path}")

            corners = await self.detect_document_edges(image)
            if corners is None:
                logger.warning("Could not detect document edges, using full image")
                processed_image = image
                confidence = 0.3
            else:
                processed_image = await self.apply_perspective_correction(image, corners)
                confidence = 0.8

            enhanced_image = await self.enhance_image_quality(processed_image)

            validation_result = await self.validate_ecg_document(enhanced_image)
            confidence *= validation_result["confidence"]

            if output_path:
                await self._save_image(enhanced_image, output_path)

            return {
                "processed_image": enhanced_image,
                "original_image": image,
                "corners": corners,
                "confidence": confidence,
                "validation": validation_result,
                "metadata": {
                    "original_size": image.shape[:2],
                    "processed_size": enhanced_image.shape[:2],
                    "processing_method": "automatic_detection" if corners is not None else "full_image",
                }
            }

        except Exception as e:
            logger.error(f"ECG image processing failed: {str(e)}")
            raise ECGProcessingException(f"ECG image processing failed: {str(e)}") from e

    async def detect_document_edges(self, image: npt.NDArray[np.uint8]) -> Optional[npt.NDArray[np.float32]]:
        """
        Detect document edges using advanced computer vision techniques.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Array of corner points [top-left, top-right, bottom-right, bottom-left] or None
        """
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            blurred = cv2.GaussianBlur(gray, self.gaussian_blur_kernel, 0)

            edges = cv2.Canny(blurred, self.edge_threshold_low, self.edge_threshold_high)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.morphology_kernel_size, self.morphology_kernel_size))
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                return None

            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)

            if area < self.min_document_area or area > self.max_document_area:
                logger.warning(f"Document area {area} outside valid range")
                return None

            epsilon = 0.02 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)

            if len(approx) == 4:
                return self._order_corner_points(approx.reshape(4, 2).astype(np.float32))

            hull = cv2.convexHull(largest_contour)
            if len(hull) >= 4:
                corners = self._find_best_corners(hull.reshape(-1, 2).astype(np.float32))
                return self._order_corner_points(corners)

            return None

        except Exception as e:
            logger.error(f"Edge detection failed: {str(e)}")
            return None

    async def find_corner_points(self, image: npt.NDArray[np.uint8]) -> Optional[npt.NDArray[np.float32]]:
        """
        Alternative method using Harris corner detection and Hough line detection.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Array of corner points or None
        """
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            corners = cv2.cornerHarris(gray, 2, 3, 0.04)
            corners = cv2.dilate(corners, None)

            edges = cv2.Canny(gray, self.edge_threshold_low, self.edge_threshold_high)
            lines = cv2.HoughLinesP(
                edges, 
                1, 
                np.pi/180, 
                self.hough_threshold,
                minLineLength=self.min_line_length,
                maxLineGap=self.max_line_gap
            )

            if lines is None or len(lines) < 4:
                return None

            intersections = []
            for i in range(len(lines)):
                for j in range(i + 1, len(lines)):
                    intersection = self._line_intersection(lines[i][0], lines[j][0])
                    if intersection is not None:
                        intersections.append(intersection)

            if len(intersections) < 4:
                return None

            corners = self._cluster_points(intersections, 4)
            if len(corners) == 4:
                return self._order_corner_points(np.array(corners, dtype=np.float32))

            return None

        except Exception as e:
            logger.error(f"Corner detection failed: {str(e)}")
            return None

    async def apply_perspective_correction(
        self, 
        image: npt.NDArray[np.uint8], 
        corners: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.uint8]:
        """
        Apply perspective transformation to correct document orientation.
        
        Args:
            image: Input image
            corners: Corner points [top-left, top-right, bottom-right, bottom-left]
            
        Returns:
            Perspective-corrected image
        """
        try:
            width_top = np.linalg.norm(corners[1] - corners[0])
            width_bottom = np.linalg.norm(corners[2] - corners[3])
            height_left = np.linalg.norm(corners[3] - corners[0])
            height_right = np.linalg.norm(corners[2] - corners[1])

            max_width = int(max(width_top, width_bottom))
            max_height = int(max(height_left, height_right))

            dst_corners = np.array([
                [0, 0],
                [max_width - 1, 0],
                [max_width - 1, max_height - 1],
                [0, max_height - 1]
            ], dtype=np.float32)

            transform_matrix = cv2.getPerspectiveTransform(corners, dst_corners)

            corrected = cv2.warpPerspective(image, transform_matrix, (max_width, max_height))

            return corrected

        except Exception as e:
            logger.error(f"Perspective correction failed: {str(e)}")
            return image

    async def enhance_image_quality(self, image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        """
        Enhance image quality using advanced image processing techniques.
        
        Args:
            image: Input image
            
        Returns:
            Enhanced image
        """
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            shadow_removed = await self._remove_shadows(gray)

            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            contrast_enhanced = clahe.apply(shadow_removed)

            adaptive_thresh = cv2.adaptiveThreshold(
                contrast_enhanced,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11,
                2
            )

            denoised = cv2.bilateralFilter(adaptive_thresh, 9, 75, 75)

            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(denoised, -1, kernel)

            return sharpened

        except Exception as e:
            logger.error(f"Image enhancement failed: {str(e)}")
            return image

    async def validate_ecg_document(
        self, 
        image: npt.NDArray[np.uint8],
        user_session: Optional[Any] = None,
        raise_non_ecg_exception: bool = True
    ) -> Dict[str, Any]:
        """
        Validate that the processed image contains a valid ECG document.
        First checks if the image is likely a non-ECG image for privacy compliance.
        
        Args:
            image: Processed image
            user_session: Optional user session for personalized responses
            raise_non_ecg_exception: Whether to raise NonECGImageException for non-ECG images
            
        Returns:
            Validation results with confidence score
            
        Raises:
            NonECGImageException: If image is classified as non-ECG and raise_non_ecg_exception is True
        """
        try:
            category, non_ecg_confidence, metadata = await self.non_ecg_classifier.classify_image_array(image)
            
            if non_ecg_confidence > self.non_ecg_threshold:
                logger.info(f"Non-ECG image detected: category={category}, confidence={non_ecg_confidence:.2f}")
                
                contextual_response = await self.response_generator.generate_response(
                    category, non_ecg_confidence, user_session
                )
                
                if raise_non_ecg_exception:
                    raise NonECGImageException(
                        message=contextual_response.get("message", "Non-ECG image detected"),
                        category=category,
                        contextual_response=contextual_response,
                        confidence=non_ecg_confidence
                    )
                else:
                    return {
                        "is_valid_ecg": False,
                        "confidence": 0.0,
                        "detected_features": [f"non_ecg_category: {category}"],
                        "grid_detected": False,
                        "leads_detected": 0,
                        "waveform_quality": 0.0,
                        "non_ecg_category": category,
                        "non_ecg_confidence": non_ecg_confidence,
                        "contextual_response": contextual_response
                    }
            
            validation_result = {
                "is_valid_ecg": False,
                "confidence": 0.0,
                "detected_features": [],
                "grid_detected": False,
                "leads_detected": 0,
                "waveform_quality": 0.0,
                "non_ecg_category": None,
                "non_ecg_confidence": non_ecg_confidence
            }

            grid_confidence = await self._detect_grid_pattern(image)
            validation_result["grid_detected"] = grid_confidence > 0.3
            validation_result["detected_features"].append(f"grid_pattern: {grid_confidence:.2f}")

            waveform_confidence = await self._detect_waveform_patterns(image)
            validation_result["waveform_quality"] = waveform_confidence
            validation_result["detected_features"].append(f"waveform_patterns: {waveform_confidence:.2f}")

            leads_detected = await self._detect_lead_labels(image)
            validation_result["leads_detected"] = leads_detected
            validation_result["detected_features"].append(f"leads_detected: {leads_detected}")

            confidence = (
                grid_confidence * 0.3 +
                waveform_confidence * 0.5 +
                min(leads_detected / 12.0, 1.0) * 0.2
            )

            validation_result["confidence"] = confidence
            validation_result["is_valid_ecg"] = confidence > 0.5

            return validation_result

        except Exception as e:
            logger.error(f"ECG validation failed: {str(e)}")
            return {
                "is_valid_ecg": False,
                "confidence": 0.0,
                "detected_features": [],
                "grid_detected": False,
                "leads_detected": 0,
                "waveform_quality": 0.0
            }


    async def _load_image(self, image_path: str) -> Optional[npt.NDArray[np.uint8]]:
        """Load image from file path."""
        try:
            path = Path(image_path)
            if not path.exists():
                raise ECGProcessingException(f"Image file not found: {image_path}")

            image = cv2.imread(str(path))
            if image is None:
                raise ECGProcessingException(f"Failed to load image: {image_path}")

            return image

        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {str(e)}")
            return None

    async def _save_image(self, image: npt.NDArray[np.uint8], output_path: str) -> bool:
        """Save processed image to file."""
        try:
            success = cv2.imwrite(output_path, image)
            if success:
                logger.info(f"Processed image saved to: {output_path}")
            return success
        except Exception as e:
            logger.error(f"Failed to save image to {output_path}: {str(e)}")
            return False

    def _order_corner_points(self, corners: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Order corner points as [top-left, top-right, bottom-right, bottom-left]."""
        center = np.mean(corners, axis=0)
        
        angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])
        sorted_indices = np.argsort(angles)
        
        ordered = corners[sorted_indices]
        
        sums = np.sum(ordered, axis=1)
        top_left_idx = np.argmin(sums)
        
        ordered = np.roll(ordered, -top_left_idx, axis=0)
        
        return ordered

    def _find_best_corners(self, points: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Find the 4 best corner points from a set of points."""
        if len(points) <= 4:
            return points[:4]

        hull_indices = cv2.convexHull(points, returnPoints=False)
        hull_points = points[hull_indices.flatten()]

        if len(hull_points) <= 4:
            return hull_points

        center = np.mean(hull_points, axis=0)
        distances = np.linalg.norm(hull_points - center, axis=1)
        
        corner_indices = np.argpartition(distances, -4)[-4:]
        corners = hull_points[corner_indices]

        return corners

    def _line_intersection(self, line1: npt.NDArray[np.int32], line2: npt.NDArray[np.int32]) -> Optional[Tuple[float, float]]:
        """Calculate intersection point of two lines."""
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:
            return None

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

        if 0 <= t <= 1 and 0 <= u <= 1:
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            return (x, y)

        return None

    def _cluster_points(self, points: List[Tuple[float, float]], n_clusters: int) -> List[Tuple[float, float]]:
        """Simple clustering to find main corner points."""
        if len(points) <= n_clusters:
            return points

        points_array = np.array(points)
        
        centroids = points_array[np.random.choice(len(points_array), n_clusters, replace=False)]
        
        for _ in range(10):  # Max iterations
            distances = np.linalg.norm(points_array[:, np.newaxis] - centroids, axis=2)
            assignments = np.argmin(distances, axis=1)
            
            new_centroids = []
            for i in range(n_clusters):
                cluster_points = points_array[assignments == i]
                if len(cluster_points) > 0:
                    new_centroids.append(np.mean(cluster_points, axis=0))
                else:
                    new_centroids.append(centroids[i])
            
            centroids = np.array(new_centroids)

        return [(float(c[0]), float(c[1])) for c in centroids]

    async def _remove_shadows(self, image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        """Remove shadows using morphological operations."""
        try:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
            
            background = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
            
            shadow_removed = cv2.subtract(image, background)
            
            shadow_removed = cv2.normalize(shadow_removed, None, 0, 255, cv2.NORM_MINMAX)
            
            return shadow_removed

        except Exception as e:
            logger.error(f"Shadow removal failed: {str(e)}")
            return image

    async def _detect_grid_pattern(self, image: npt.NDArray[np.uint8]) -> float:
        """Detect grid pattern typical in ECG papers."""
        try:
            edges = cv2.Canny(image, 50, 150)
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)

            if lines is None:
                return 0.0

            horizontal_lines = 0
            vertical_lines = 0
            
            for line in lines:
                rho, theta = line[0]
                angle = theta * 180 / np.pi
                
                if abs(angle) < 10 or abs(angle - 180) < 10:  # Horizontal
                    horizontal_lines += 1
                elif abs(angle - 90) < 10:  # Vertical
                    vertical_lines += 1

            grid_confidence = min(horizontal_lines / 10.0, 1.0) * min(vertical_lines / 10.0, 1.0)
            
            return grid_confidence

        except Exception as e:
            logger.error(f"Grid detection failed: {str(e)}")
            return 0.0

    async def _detect_waveform_patterns(self, image: npt.NDArray[np.uint8]) -> float:
        """Detect ECG waveform patterns."""
        try:
            
            horizontal_projection = np.sum(image == 0, axis=1)  # Count black pixels
            
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(horizontal_projection, height=np.max(horizontal_projection) * 0.1)
            
            if len(peaks) >= 3:  # At least 3 signal areas
                return 0.8
            elif len(peaks) >= 1:
                return 0.5
            else:
                return 0.1

        except Exception as e:
            logger.error(f"Waveform detection failed: {str(e)}")
            return 0.0

    async def _detect_lead_labels(self, image: npt.NDArray[np.uint8]) -> int:
        """Detect ECG lead labels (I, II, III, aVR, etc.)."""
        try:
            
            height, width = image.shape[:2]
            
            if height > width:  # Portrait orientation
                estimated_leads = 12
            elif width / height > 2:  # Very wide, likely 3x4 or 4x3 layout
                estimated_leads = 12
            else:
                estimated_leads = 6  # Partial ECG

            return estimated_leads

        except Exception as e:
            logger.error(f"Lead detection failed: {str(e)}")
            return 0

    def _check_gpu_availability(self) -> bool:
        """Check if GPU acceleration is available."""
        try:
            import cv2
            return cv2.cuda.getCudaEnabledDeviceCount() > 0
        except Exception:
            return False


class BatchECGScanner:
    """Batch processing capabilities for multiple ECG images."""

    def __init__(self, max_workers: int = 4) -> None:
        """Initialize batch scanner with concurrency control."""
        self.max_workers = max_workers
        self.scanner = ECGDocumentScanner()

    async def process_batch(
        self, 
        image_paths: List[str],
        output_dir: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Process multiple ECG images in parallel.
        
        Args:
            image_paths: List of input image paths
            output_dir: Optional directory to save processed images
            
        Returns:
            List of processing results
        """
        semaphore = asyncio.Semaphore(self.max_workers)
        
        async def process_single(image_path: str) -> Dict[str, Any]:
            async with semaphore:
                try:
                    output_path = None
                    if output_dir:
                        output_path = str(Path(output_dir) / f"processed_{Path(image_path).name}")
                    
                    result = await self.scanner.process_ecg_image(image_path, output_path)
                    result["input_path"] = image_path
                    result["success"] = True
                    return result
                    
                except Exception as e:
                    logger.error(f"Failed to process {image_path}: {str(e)}")
                    return {
                        "input_path": image_path,
                        "success": False,
                        "error": str(e),
                        "confidence": 0.0
                    }

        tasks = [process_single(path) for path in image_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        valid_results = []
        for result in results:
            if isinstance(result, dict):
                valid_results.append(result)
            else:
                logger.error(f"Batch processing exception: {result}")
                
        return valid_results

    async def get_processing_stats(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate statistics from batch processing results."""
        total_processed = len(results)
        successful = sum(1 for r in results if r.get("success", False))
        failed = total_processed - successful
        
        if successful > 0:
            avg_confidence = np.mean([r.get("confidence", 0.0) for r in results if r.get("success", False)])
            valid_ecgs = sum(1 for r in results if r.get("validation", {}).get("is_valid_ecg", False))
        else:
            avg_confidence = 0.0
            valid_ecgs = 0

        return {
            "total_processed": total_processed,
            "successful": successful,
            "failed": failed,
            "success_rate": successful / total_processed if total_processed > 0 else 0.0,
            "average_confidence": avg_confidence,
            "valid_ecgs_detected": valid_ecgs,
            "ecg_detection_rate": valid_ecgs / successful if successful > 0 else 0.0
        }
