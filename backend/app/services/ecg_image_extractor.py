"""
ECG Image Extractor Service - Advanced implementation for extracting ECG signals from images.
Supports multiple formats including JPEG, JPG, PNG, PDF and other image formats.
"""

import os
import logging
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image
from pdf2image import convert_from_path
import pytesseract
from scipy import signal as scipy_signal
from skimage import filters, morphology, measure, feature, color, exposure, segmentation

from app.core.exceptions import ECGProcessingException

logger = logging.getLogger(__name__)


class ECGImageExtractor:
    """Advanced ECG image extractor with support for multiple formats."""

    def __init__(self) -> None:
        """Initialize the ECG image extractor."""
        self.supported_formats = {
            ".png": self._process_image,
            ".jpg": self._process_image,
            ".jpeg": self._process_image,
            ".bmp": self._process_image,
            ".tiff": self._process_image,
            ".tif": self._process_image,
            ".pdf": self._process_pdf,
        }
        
        # Default parameters for signal extraction
        self.default_sampling_rate = 500
        self.default_leads = [
            "I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"
        ]
        
        # Grid detection parameters
        self.grid_detection_params = {
            "min_line_length": 50,
            "max_line_gap": 10,
            "threshold": 50,
            "grid_spacing_mm": 5,  # Standard ECG grid spacing in mm
            "pixels_per_mm": None,  # Will be calculated based on image
        }
        
        # Signal extraction parameters
        self.signal_extraction_params = {
            "signal_color": "black",  # ECG signal is typically black or dark
            "background_threshold": 200,  # Threshold for background (white) pixels
            "min_signal_length": 100,  # Minimum length of a valid signal in pixels
            "max_signal_width": 5,  # Maximum width of the signal line in pixels
        }

    def extract_from_file(self, file_path: str) -> Dict[str, Any]:
        """
        Extract ECG signal from an image file.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            Dictionary containing the extracted signal data and metadata
        """
        path = Path(file_path)
        ext = path.suffix.lower()
        
        if ext not in self.supported_formats:
            raise ECGProcessingException(f"Unsupported file format: {ext}")
        
        try:
            return self.supported_formats[ext](file_path)
        except Exception as e:
            logger.error(f"Failed to extract ECG from image: {e}")
            raise ECGProcessingException(f"Error extracting ECG from image: {str(e)}")

    def _process_pdf(self, file_path: str) -> Dict[str, Any]:
        """
        Process a PDF file by converting it to images and then processing each image.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Dictionary containing the extracted signal data and metadata
        """
        try:
            # Convert PDF to images
            with tempfile.TemporaryDirectory() as temp_dir:
                images = convert_from_path(file_path, output_folder=temp_dir)
                
                # Process the first page by default
                # In the future, this could be extended to process multiple pages
                if images:
                    # Save the first image to a temporary file
                    temp_image_path = os.path.join(temp_dir, "temp_image.png")
                    images[0].save(temp_image_path, "PNG")
                    
                    # Process the image
                    result = self._process_image(temp_image_path)
                    
                    # Add metadata about PDF source
                    result["metadata"]["source_type"] = "pdf"
                    result["metadata"]["page_count"] = len(images)
                    result["metadata"]["processed_page"] = 1
                    
                    return result
                else:
                    raise ECGProcessingException("No images extracted from PDF")
                
        except Exception as e:
            logger.error(f"PDF processing failed: {e}")
            raise ECGProcessingException(f"PDF processing failed: {str(e)}")

    def _process_image(self, file_path: str) -> Dict[str, Any]:
        """
        Process an image file to extract ECG signals.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            Dictionary containing the extracted signal data and metadata
        """
        try:
            # Load and preprocess the image
            image = cv2.imread(file_path)
            if image is None:
                raise ECGProcessingException(f"Could not load image from {file_path}")
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply preprocessing to enhance the ECG signal
            preprocessed = self._preprocess_image(gray)
            
            # Detect the ECG grid
            grid_info = self._detect_grid(preprocessed)
            
            # Detect lead regions
            lead_regions = self._detect_lead_regions(preprocessed, grid_info)
            
            # Extract signals from each region
            signals = []
            confidence_scores = []
            
            for i, region in enumerate(lead_regions):
                signal, confidence = self._extract_signal_from_region(
                    region, self.default_sampling_rate
                )
                signals.append(signal)
                confidence_scores.append(confidence)
            
            # Ensure we have 12 leads (standard ECG)
            while len(signals) < 12:
                signals.append(np.zeros(int(self.default_sampling_rate * 10)))  # 10 seconds of zeros
                confidence_scores.append(0.0)
            
            # Create signal matrix
            signal_matrix = np.column_stack(signals[:12])
            
            # Calculate average confidence
            avg_confidence = np.mean(confidence_scores[:12])
            
            # Return the result
            return {
                "signal": signal_matrix,
                "sampling_rate": self.default_sampling_rate,
                "labels": self.default_leads,
                "metadata": {
                    "source": "digitized_image",
                    "source_type": "image",
                    "scanner_confidence": float(avg_confidence),
                    "document_detected": grid_info["grid_detected"],
                    "processing_method": "advanced_image_processing",
                    "grid_detected": grid_info["grid_detected"],
                    "leads_detected": len([c for c in confidence_scores if c > 0.3]),
                    "image_resolution": f"{image.shape[1]}x{image.shape[0]}",
                    "file_path": file_path,
                },
            }
            
        except Exception as e:
            logger.error(f"Image processing failed: {e}")
            raise ECGProcessingException(f"Image processing failed: {str(e)}")

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess the image to enhance the ECG signal.
        
        Args:
            image: Grayscale image
            
        Returns:
            Preprocessed image
        """
        # Apply contrast enhancement
        enhanced = exposure.equalize_adapthist(image)
        enhanced = (enhanced * 255).astype(np.uint8)
        
        # Apply thresholding to separate signal from background
        _, binary = cv2.threshold(enhanced, 200, 255, cv2.THRESH_BINARY_INV)
        
        # Remove small noise
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Thin the lines to make signal extraction more accurate
        thinned = morphology.thin(cleaned).astype(np.uint8) * 255
        
        return thinned

    def _detect_grid(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Detect the ECG grid in the image.
        
        Args:
            image: Preprocessed image
            
        Returns:
            Dictionary containing grid information
        """
        height, width = image.shape
        
        # Create kernels for horizontal and vertical lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
        
        # Detect horizontal and vertical lines
        horizontal_lines = cv2.morphologyEx(image, cv2.MORPH_OPEN, horizontal_kernel)
        vertical_lines = cv2.morphologyEx(image, cv2.MORPH_OPEN, vertical_kernel)
        
        # Combine horizontal and vertical lines to get the grid
        grid = cv2.add(horizontal_lines, vertical_lines)
        
        # Check if grid is detected
        grid_detected = np.sum(grid) > 1000
        
        # Estimate grid spacing
        if grid_detected:
            # Use Hough transform to detect lines
            lines = cv2.HoughLinesP(
                grid, 
                1, 
                np.pi/180, 
                threshold=self.grid_detection_params["threshold"],
                minLineLength=self.grid_detection_params["min_line_length"],
                maxLineGap=self.grid_detection_params["max_line_gap"]
            )
            
            if lines is not None:
                # Separate horizontal and vertical lines
                horizontal_distances = []
                vertical_distances = []
                
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    if abs(x2 - x1) > abs(y2 - y1):  # Horizontal line
                        horizontal_distances.append((x1, y1))
                    else:  # Vertical line
                        vertical_distances.append((x1, y1))
                
                # Sort by position
                horizontal_distances.sort(key=lambda x: x[1])
                vertical_distances.sort(key=lambda x: x[0])
                
                # Calculate average spacing
                h_spacing = self._calculate_average_spacing([p[1] for p in horizontal_distances])
                v_spacing = self._calculate_average_spacing([p[0] for p in vertical_distances])
                
                # Estimate pixels per mm based on standard ECG grid (5mm spacing)
                if h_spacing > 0 and v_spacing > 0:
                    h_pixels_per_mm = h_spacing / self.grid_detection_params["grid_spacing_mm"]
                    v_pixels_per_mm = v_spacing / self.grid_detection_params["grid_spacing_mm"]
                    pixels_per_mm = (h_pixels_per_mm + v_pixels_per_mm) / 2
                else:
                    pixels_per_mm = None
            else:
                h_spacing = v_spacing = pixels_per_mm = None
        else:
            h_spacing = v_spacing = pixels_per_mm = None
        
        return {
            "grid_detected": grid_detected,
            "horizontal_lines": horizontal_lines,
            "vertical_lines": vertical_lines,
            "grid": grid,
            "h_spacing": h_spacing,
            "v_spacing": v_spacing,
            "pixels_per_mm": pixels_per_mm,
        }

    def _calculate_average_spacing(self, positions: List[int]) -> float:
        """
        Calculate the average spacing between grid lines.
        
        Args:
            positions: List of line positions
            
        Returns:
            Average spacing between lines
        """
        if len(positions) < 2:
            return 0
        
        # Calculate differences between consecutive positions
        differences = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
        
        # Filter out outliers
        if differences:
            mean_diff = np.mean(differences)
            std_diff = np.std(differences)
            filtered_diffs = [d for d in differences if abs(d - mean_diff) < 2 * std_diff]
            
            if filtered_diffs:
                return np.mean(filtered_diffs)
        
        return 0

    def _detect_lead_regions(self, image: np.ndarray, grid_info: Dict[str, Any]) -> List[np.ndarray]:
        """
        Detect regions containing individual ECG leads.
        
        Args:
            image: Preprocessed image
            grid_info: Grid information from _detect_grid
            
        Returns:
            List of image regions containing individual leads
        """
        height, width = image.shape
        
        # If grid is detected, try to use it for more accurate lead detection
        if grid_info["grid_detected"] and grid_info["h_spacing"] and grid_info["v_spacing"]:
            # Try to detect lead labels using OCR
            try:
                # Convert back to regular image for OCR
                regular_image = 255 - image
                text_regions = pytesseract.image_to_data(regular_image, output_type=pytesseract.Output.DICT)
                
                # Look for lead labels (I, II, III, aVR, aVL, aVF, V1-V6)
                lead_positions = []
                for i, text in enumerate(text_regions["text"]):
                    if text.strip() in self.default_leads:
                        x = text_regions["left"][i]
                        y = text_regions["top"][i]
                        w = text_regions["width"][i]
                        h = text_regions["height"][i]
                        lead_positions.append((text, x, y, w, h))
                
                # If we found lead labels, use them to define regions
                if len(lead_positions) >= 3:  # At least a few leads detected
                    regions = []
                    for lead, x, y, w, h in lead_positions:
                        # Define a region to the right of the label
                        region_x = x + w
                        region_y = y - h  # Slightly above the label
                        region_w = width // 4  # Approximate width
                        region_h = height // 6  # Approximate height
                        
                        # Ensure region is within image bounds
                        region_x = max(0, region_x)
                        region_y = max(0, region_y)
                        region_w = min(width - region_x, region_w)
                        region_h = min(height - region_y, region_h)
                        
                        if region_w > 0 and region_h > 0:
                            region = image[region_y:region_y+region_h, region_x:region_x+region_w]
                            regions.append(region)
                    
                    # If we found regions, return them
                    if regions:
                        return regions
            except Exception as e:
                logger.warning(f"OCR-based lead detection failed: {e}")
        
        # Fallback: divide image into a grid of regions
        regions = []
        rows, cols = 3, 4  # Standard 12-lead ECG layout
        
        for row in range(rows):
            for col in range(cols):
                y1 = int(row * height / rows)
                y2 = int((row + 1) * height / rows)
                x1 = int(col * width / cols)
                x2 = int((col + 1) * width / cols)
                
                region = image[y1:y2, x1:x2]
                regions.append(region)
        
        return regions

    def _extract_signal_from_region(self, region: np.ndarray, sampling_rate: int) -> Tuple[np.ndarray, float]:
        """
        Extract ECG signal from a region using advanced image processing.
        
        Args:
            region: Image region containing an ECG lead
            sampling_rate: Target sampling rate for the extracted signal
            
        Returns:
            Tuple of (signal, confidence)
        """
        try:
            height, width = region.shape
            
            # Find contours in the region
            contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return np.zeros(int(sampling_rate * 10)), 0.0
            
            # Filter contours by size to find the ECG signal line
            valid_contours = []
            for contour in contours:
                # Calculate contour properties
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                
                # Calculate contour thinness (perimeter^2 / area)
                # ECG signal lines are thin and long
                if area > 0:
                    thinness = perimeter**2 / area
                    if thinness > 100:  # Threshold for thin lines
                        valid_contours.append(contour)
            
            if not valid_contours:
                # Try alternative approach: use the longest contour
                longest_contour = max(contours, key=cv2.contourArea)
                valid_contours = [longest_contour]
            
            # Create a mask for all valid contours
            mask = np.zeros_like(region)
            cv2.drawContours(mask, valid_contours, -1, 255, 1)
            
            # Extract signal points
            signal_points = []
            
            for x in range(width):
                # Find all y-coordinates of signal at this x-position
                y_coords = np.where(mask[:, x] > 0)[0]
                
                if len(y_coords) > 0:
                    # Use the average y-coordinate if multiple points
                    avg_y = np.mean(y_coords)
                    # Normalize: invert and scale to [-0.5, 0.5]
                    signal_value = (height - avg_y) / height - 0.5
                    signal_points.append(signal_value)
                else:
                    # If no signal at this x-position, use previous value or 0
                    if signal_points:
                        signal_points.append(signal_points[-1])
                    else:
                        signal_points.append(0.0)
            
            # Resample to target sampling rate
            if signal_points:
                # Target 10 seconds of signal
                target_length = int(sampling_rate * 10)
                
                # Interpolate to get uniform sampling
                signal = np.interp(
                    np.linspace(0, len(signal_points) - 1, target_length),
                    np.arange(len(signal_points)),
                    signal_points
                )
                
                # Calculate confidence based on signal variance and continuity
                variance = np.var(signal)
                continuity = self._calculate_signal_continuity(signal)
                
                # Combined confidence score
                confidence = min(1.0, variance * 20) * continuity
                
                return signal, float(confidence)
            else:
                return np.zeros(int(sampling_rate * 10)), 0.0
            
        except Exception as e:
            logger.warning(f"Signal extraction failed: {e}")
            return np.zeros(int(sampling_rate * 10)), 0.0

    def _calculate_signal_continuity(self, signal: np.ndarray) -> float:
        """
        Calculate the continuity of the signal (how well-connected it is).
        
        Args:
            signal: Extracted signal
            
        Returns:
            Continuity score between 0 and 1
        """
        # Calculate the derivative
        derivative = np.diff(signal)
        
        # Count large jumps (discontinuities)
        threshold = 3 * np.std(derivative)
        jumps = np.sum(np.abs(derivative) > threshold)
        
        # Calculate continuity score
        if len(derivative) > 0:
            continuity = 1.0 - min(1.0, jumps / len(derivative) * 10)
            return float(continuity)
        else:
            return 0.0

    def extract_text_from_image(self, image_path: str) -> str:
        """
        Extract text from an image using OCR.
        
        Args:
            image_path: Path to the image
            
        Returns:
            Extracted text
        """
        try:
            # Load image
            image = Image.open(image_path)
            
            # Extract text
            text = pytesseract.image_to_string(image)
            
            return text
        except Exception as e:
            logger.error(f"OCR text extraction failed: {e}")
            return ""