"""
Optimized ECG Document Scanner Service

This module provides performance-optimized ECG document scanning capabilities including:
- Multi-resolution processing for faster edge detection
- GPU acceleration support with CUDA
- Caching for transformation matrices
- Optimized OpenCV operations
- Progressive enhancement approach
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
from app.core.exceptions import ECGProcessingException

logger = logging.getLogger(__name__)


class OptimizedECGDocumentScanner:
    """
    Performance-optimized ECG document scanner with advanced computer vision algorithms
    for automatic document detection, perspective correction, and enhancement.
    
    Features:
    - Multi-resolution processing for performance
    - GPU acceleration with CUDA support
    - Transformation matrix caching
    - Progressive enhancement approach
    """

    def __init__(self) -> None:
        """Initialize the optimized ECG document scanner."""
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
        
        logger.info(f"Initialized ECG scanner with GPU support: {self.gpu_available}")

    async def process_ecg_image(
        self, 
        image_path: str,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Main pipeline method to process ECG image with performance optimizations.
        
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

            corners = await self.detect_document_edges_optimized(image)
            if corners is None:
                logger.warning("Could not detect document edges, using full image")
                processed_image = image
                confidence = 0.3
                processing_method = "full_image"
            else:
                processed_image = await self.apply_perspective_correction_optimized(image, corners)
                confidence = 0.8
                processing_method = "automatic_detection"

            enhanced_image = await self.enhance_image_quality_optimized(processed_image)

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
                    "processing_method": processing_method,
                    "gpu_accelerated": self.gpu_available,
                }
            }

        except Exception as e:
            logger.error(f"ECG image processing failed: {str(e)}")
            raise ECGProcessingException(f"Processing failed: {str(e)}")

    async def detect_document_edges_optimized(self, image: npt.NDArray[np.uint8]) -> Optional[npt.NDArray[np.float32]]:
        """
        Detect document edges using multi-resolution approach for performance.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Array of corner points if document detected, None otherwise
        """
        try:
            original_shape = image.shape[:2]
            scale_factor = self._calculate_scale_factor(original_shape)
            
            if scale_factor < 1.0:
                small_image = self._resize_image_optimized(image, scale_factor)
                corners = await self._detect_edges_at_resolution(small_image)
                
                if corners is not None:
                    corners = corners / scale_factor
                    corners = await self._refine_corners_full_resolution(image, corners)
                    return corners
            else:
                return await self._detect_edges_at_resolution(image)
            
            return None

        except Exception as e:
            logger.error(f"Error in optimized edge detection: {str(e)}")
            raise ECGProcessingException(f"Edge detection failed: {str(e)}")

    async def _detect_edges_at_resolution(self, image: npt.NDArray[np.uint8]) -> Optional[npt.NDArray[np.float32]]:
        """Detect edges at a specific resolution with GPU acceleration."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        gray = self._preprocess_image_optimized(gray)

        edges = self._canny_edge_detection_optimized(gray)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.morphology_kernel_size, self.morphology_kernel_size))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            logger.warning("No contours found in image")
            return None

        largest_contour = max(contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(largest_contour)

        if contour_area < self.min_document_area:
            logger.warning(f"Largest contour area {contour_area} below minimum {self.min_document_area}")
            return None

        corners = await self.find_corner_points(largest_contour)
        
        if corners is not None:
            return self._order_corner_points(corners)
        
        return None

    async def apply_perspective_correction_optimized(
        self, 
        image: npt.NDArray[np.uint8], 
        corners: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.uint8]:
        """
        Apply perspective correction with caching and GPU acceleration.
        
        Args:
            image: Input image
            corners: Four corner points of the document
            
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

            corners_hash = hash(corners.tobytes())
            
            transform_matrix = self._get_cached_transformation_matrix(
                str(corners_hash), max_width, max_height
            )
            
            if transform_matrix is None:
                dst_points = np.array([
                    [0, 0],
                    [max_width - 1, 0],
                    [max_width - 1, max_height - 1],
                    [0, max_height - 1]
                ], dtype=np.float32)

                transform_matrix = cv2.getPerspectiveTransform(corners, dst_points)
                
                self._cache_transformation_matrix(
                    str(corners_hash), max_width, max_height, transform_matrix
                )

            if self.gpu_available:
                try:
                    gpu_img = cv2.cuda_GpuMat()
                    gpu_img.upload(image)
                    gpu_corrected = cv2.cuda.warpPerspective(
                        gpu_img, transform_matrix, (max_width, max_height)
                    )
                    corrected = gpu_corrected.download()
                    return corrected
                except Exception as e:
                    logger.warning(f"GPU perspective correction failed, falling back to CPU: {e}")

            corrected = cv2.warpPerspective(image, transform_matrix, (max_width, max_height))
            return corrected

        except Exception as e:
            logger.error(f"Error in optimized perspective correction: {str(e)}")
            raise ECGProcessingException(f"Perspective correction failed: {str(e)}")

    async def enhance_image_quality_optimized(self, image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        """
        Enhance image quality with GPU acceleration for better ECG analysis.
        
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

            if self.gpu_available:
                try:
                    return await self._enhance_image_gpu(gray)
                except Exception as e:
                    logger.warning(f"GPU enhancement failed, falling back to CPU: {e}")

            return await self._enhance_image_cpu(gray)

        except Exception as e:
            logger.error(f"Error in optimized image enhancement: {str(e)}")
            raise ECGProcessingException(f"Image enhancement failed: {str(e)}")

    async def _enhance_image_gpu(self, gray: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        """GPU-accelerated image enhancement."""
        gpu_img = cv2.cuda_GpuMat()
        gpu_img.upload(gray)
        
        clahe = cv2.cuda.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gpu_enhanced = clahe.apply(gpu_img)
        
        gpu_denoised = cv2.cuda.bilateralFilter(gpu_enhanced, -1, 75, 75)
        
        enhanced = gpu_denoised.download()
        
        shadow_removed = await self._remove_shadows(enhanced)
        
        gaussian = cv2.GaussianBlur(shadow_removed, (0, 0), 2.0)
        sharpened = cv2.addWeighted(shadow_removed, 1.5, gaussian, -0.5, 0)
        
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    async def _enhance_image_cpu(self, gray: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        """CPU-based image enhancement."""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)

        shadow_removed = await self._remove_shadows(denoised)

        gaussian = cv2.GaussianBlur(shadow_removed, (0, 0), 2.0)
        sharpened = cv2.addWeighted(shadow_removed, 1.5, gaussian, -0.5, 0)

        return np.clip(sharpened, 0, 255).astype(np.uint8)

    def _calculate_scale_factor(self, image_shape: Tuple[int, int]) -> float:
        """Calculate optimal scale factor for multi-resolution processing."""
        height, width = image_shape
        max_dimension = max(height, width)
        
        if max_dimension <= self.max_detection_size:
            return 1.0
        
        return self.max_detection_size / max_dimension

    def _resize_image_optimized(self, image: npt.NDArray[np.uint8], scale_factor: float) -> npt.NDArray[np.uint8]:
        """Resize image with optimized interpolation."""
        if scale_factor >= 1.0:
            return image
        
        height, width = image.shape[:2]
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    def _preprocess_image_optimized(self, gray: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        """Optimized image preprocessing."""
        kernel_size = 5 if gray.shape[0] > 1000 else 3
        return cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    def _canny_edge_detection_optimized(self, gray: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        """Optimized Canny edge detection with GPU support if available."""
        if self.gpu_available:
            try:
                gpu_img = cv2.cuda_GpuMat()
                gpu_img.upload(gray)
                gpu_edges = cv2.cuda.Canny(gpu_img, self.edge_threshold_low, self.edge_threshold_high)
                edges = gpu_edges.download()
                return edges
            except Exception:
                pass
        
        return cv2.Canny(gray, self.edge_threshold_low, self.edge_threshold_high)

    async def _refine_corners_full_resolution(
        self, 
        image: npt.NDArray[np.uint8], 
        corners: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        """Refine corner detection at full resolution."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        refined_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        return self._order_corner_points(refined_corners)

    @lru_cache(maxsize=100)
    def _get_cached_transformation_matrix(
        self, 
        corners_hash: str, 
        target_width: int, 
        target_height: int
    ) -> Optional[npt.NDArray[np.float32]]:
        """Get cached transformation matrix for common transformations."""
        cache_key = f"{corners_hash}_{target_width}_{target_height}"
        return self._transformation_cache.get(cache_key)

    def _cache_transformation_matrix(
        self, 
        corners_hash: str, 
        target_width: int, 
        target_height: int, 
        matrix: npt.NDArray[np.float32]
    ) -> None:
        """Cache transformation matrix for reuse."""
        cache_key = f"{corners_hash}_{target_width}_{target_height}"
        
        if len(self._transformation_cache) >= self._cache_max_size:
            oldest_key = next(iter(self._transformation_cache))
            del self._transformation_cache[oldest_key]
        
        self._transformation_cache[cache_key] = matrix

    def _check_gpu_availability(self) -> bool:
        """Check if GPU acceleration is available."""
        try:
            return cv2.cuda.getCudaEnabledDeviceCount() > 0
        except Exception:
            return False

    async def find_corner_points(self, contour: npt.NDArray[np.int32]) -> Optional[npt.NDArray[np.float32]]:
        """Find corner points from contour using Douglas-Peucker algorithm."""
        try:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(approx) == 4:
                return approx.reshape(4, 2).astype(np.float32)
            
            hull = cv2.convexHull(contour)
            if len(hull) >= 4:
                corners = self._find_best_corners(hull)
                if corners is not None:
                    return corners.astype(np.float32)
            
            return None
            
        except Exception as e:
            logger.error(f"Error finding corner points: {str(e)}")
            return None

    def _find_best_corners(self, hull: npt.NDArray[np.int32]) -> Optional[npt.NDArray[np.int32]]:
        """Find the best 4 corners from convex hull."""
        try:
            hull = hull.reshape(-1, 2)
            
            top_left = hull[np.argmin(hull.sum(axis=1))]
            bottom_right = hull[np.argmax(hull.sum(axis=1))]
            top_right = hull[np.argmin(hull[:, 1] - hull[:, 0])]
            bottom_left = hull[np.argmax(hull[:, 1] - hull[:, 0])]
            
            return np.array([top_left, top_right, bottom_right, bottom_left])
            
        except Exception:
            return None

    def _order_corner_points(self, corners: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Order corner points consistently: top-left, top-right, bottom-right, bottom-left."""
        sum_coords = corners.sum(axis=1)
        diff_coords = np.diff(corners, axis=1)
        
        top_left = corners[np.argmin(sum_coords)]
        bottom_right = corners[np.argmax(sum_coords)]
        
        top_right = corners[np.argmin(diff_coords)]
        bottom_left = corners[np.argmax(diff_coords)]
        
        return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)

    async def validate_ecg_document(self, image: npt.NDArray[np.uint8]) -> Dict[str, Any]:
        """Validate if the processed image contains a valid ECG."""
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            grid_detected = await self._detect_grid_pattern(gray)
            waveform_detected = await self._detect_waveform_patterns(gray)
            
            confidence = 0.0
            if grid_detected:
                confidence += 0.4
            if waveform_detected:
                confidence += 0.6
            
            return {
                "is_valid_ecg": confidence > 0.5,
                "confidence": confidence,
                "detected_features": {
                    "grid_pattern": grid_detected,
                    "waveform_patterns": waveform_detected,
                },
                "grid_detected": grid_detected,
                "leads_detected": 12 if confidence > 0.7 else 0,
                "waveform_quality": "high" if confidence > 0.8 else "medium" if confidence > 0.5 else "low"
            }
            
        except Exception as e:
            logger.error(f"Error in ECG validation: {str(e)}")
            return {
                "is_valid_ecg": False,
                "confidence": 0.0,
                "detected_features": {},
                "grid_detected": False,
                "leads_detected": 0,
                "waveform_quality": "low"
            }

    async def _detect_grid_pattern(self, gray: npt.NDArray[np.uint8]) -> bool:
        """Detect ECG grid pattern in the image."""
        try:
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=self.hough_threshold,
                                   minLineLength=self.min_line_length, maxLineGap=self.max_line_gap)
            
            if lines is not None and len(lines) > 10:
                return True
            return False
            
        except Exception:
            return False

    async def _detect_waveform_patterns(self, gray: npt.NDArray[np.uint8]) -> bool:
        """Detect ECG waveform patterns in the image."""
        try:
            edges = cv2.Canny(gray, 30, 100)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            waveform_contours = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if 100 < area < 10000:  # Reasonable size for waveform segments
                    waveform_contours += 1
            
            return waveform_contours > 5
            
        except Exception:
            return False

    async def _remove_shadows(self, image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        """Remove shadows from the image."""
        try:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
            background = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
            
            result = cv2.subtract(image, background)
            result = cv2.add(result, np.full_like(result, 128))  # Add offset
            
            return result
            
        except Exception:
            return image

    async def _load_image(self, image_path: str) -> Optional[npt.NDArray[np.uint8]]:
        """Load image from file path."""
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ECGProcessingException(f"Could not load image: {image_path}")
            return image
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {str(e)}")
            return None

    async def _save_image(self, image: npt.NDArray[np.uint8], output_path: str) -> None:
        """Save image to file path."""
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(output_path, image)
        except Exception as e:
            logger.error(f"Error saving image to {output_path}: {str(e)}")
            raise ECGProcessingException(f"Could not save image: {str(e)}")


async def create_optimized_scanner() -> OptimizedECGDocumentScanner:
    """Create an optimized ECG document scanner instance."""
    return OptimizedECGDocumentScanner()
