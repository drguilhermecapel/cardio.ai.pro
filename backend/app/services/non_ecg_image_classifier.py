"""
Non-ECG Image Classification Service
Intelligent classification and contextual response system for non-ECG images.
"""

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
from PIL import Image

logger = logging.getLogger(__name__)


class NonECGImageClassifier:
    """Multi-class classifier for non-ECG images with contextual responses."""

    def __init__(self) -> None:
        """Initialize the non-ECG image classifier."""
        self.categories = {
            'medical_other': 'Outro exame médico',
            'document': 'Documento geral',
            'xray': 'Radiografia',
            'prescription': 'Receita médica',
            'lab_results': 'Resultado laboratorial',
            'photo_person': 'Foto de pessoa',
            'screenshot': 'Captura de tela',
            'nature': 'Foto de natureza/paisagem',
            'object': 'Objeto comum',
            'text_document': 'Documento de texto',
            'other_medical_device': 'Outro dispositivo médico',
            'monitor_screen': 'Tela de monitor médico',
            'handwritten': 'Anotação manuscrita',
            'food': 'Comida',
            'unknown': 'Não identificado'
        }

        self.confidence_thresholds = {
            'medical_other': 0.7,
            'document': 0.6,
            'xray': 0.8,
            'prescription': 0.7,
            'lab_results': 0.7,
            'photo_person': 0.6,
            'screenshot': 0.5,
            'nature': 0.6,
            'object': 0.5,
            'text_document': 0.6,
            'other_medical_device': 0.7,
            'monitor_screen': 0.6,
            'handwritten': 0.5,
            'food': 0.6,
            'unknown': 0.3
        }

    async def classify_image(self, image_path: str) -> tuple[str, float, dict[str, Any]]:
        """
        Classify non-ECG image and return category, confidence, and metadata.

        Args:
            image_path: Path to the image file

        Returns:
            Tuple of (category, confidence, metadata)
        """
        try:
            image = await self._load_image(image_path)
            if image is None:
                return 'unknown', 0.0, {'error': 'Failed to load image'}

            features = await self._extract_features(image)
            category, confidence = await self._classify_features(features)

            metadata = {
                'image_size': image.shape[:2],
                'features': features,
                'classification_method': 'computer_vision_heuristics',
                'processing_time': 0.0  # Could add timing if needed
            }

            return category, confidence, metadata

        except Exception as e:
            logger.error(f"Error classifying image {image_path}: {str(e)}")
            return 'unknown', 0.0, {'error': str(e)}

    async def _load_image(self, image_path: str) -> npt.NDArray[np.uint8] | None:
        """Load image from file path."""
        try:
            if not Path(image_path).exists():
                return None

            image = cv2.imread(image_path)
            if image is None:
                pil_image = Image.open(image_path)  # type: ignore[unreachable]
                image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

            return image.astype(np.uint8)
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {str(e)}")
            raise Exception(f"Failed to load image {image_path}: {str(e)}") from e

    async def _extract_features(self, image: npt.NDArray[np.uint8]) -> dict[str, Any]:
        """Extract features from image for classification."""
        features: dict[str, Any] = {}

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.uint8)
        else:
            gray = image.astype(np.uint8)

        height, width = gray.shape
        features['size_category'] = self._categorize_size(width, height)
        features['aspect_ratio'] = float(width / height)

        color_analysis = await self._analyze_colors(image)
        features['color_analysis'] = color_analysis

        if len(image.shape) == 3:
            features['color_variance'] = float(np.var(image))
            features['brightness'] = float(np.mean(image))
        else:
            features['color_variance'] = float(np.var(image))
            features['brightness'] = float(np.mean(image))

        features['edge_density'] = await self._calculate_edge_density(gray)
        texture_analysis = await self._analyze_texture(gray)
        features['texture_analysis'] = texture_analysis
        features['texture_energy'] = texture_analysis.get('energy', 0.0)

        grid_pattern = await self._detect_grid_pattern(gray)
        features['grid_pattern'] = grid_pattern

        features['text_density'] = await self._estimate_text_density(gray)

        medical_indicators = await self._detect_medical_indicators(image)
        features['medical_indicators'] = medical_indicators

        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        features['contour_count'] = len(contours)

        screen_indicators = await self._detect_screen_indicators(image)
        features['screen_indicators'] = screen_indicators

        return features

    def _categorize_size(self, width: int, height: int) -> str:
        """Categorize image size."""
        total_pixels = width * height

        if total_pixels < 100000:  # < 0.1MP
            return 'small'
        elif total_pixels < 1000000:  # < 1MP
            return 'medium'
        elif total_pixels < 5000000:  # < 5MP
            return 'large'
        else:
            return 'very_large'

    async def _analyze_colors(self, image: npt.NDArray[np.uint8]) -> dict[str, Any]:
        """Analyze color distribution in image."""
        if len(image.shape) == 3:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.uint8)

            hist_h = cv2.calcHist([hsv], [0], None, [180], [0, 180])

            dominant_hue = np.argmax(hist_h)
            saturation_mean = np.mean(hsv[:, :, 1])
            value_mean = np.mean(hsv[:, :, 2])

            medical_colors = self._detect_medical_colors(hsv)

            colors = image.reshape(-1, 3)
            unique_colors = np.unique(colors, axis=0)
            dominant_colors = unique_colors[:5].tolist()  # Top 5 colors

            return {
                'dominant_hue': int(dominant_hue),
                'saturation_mean': float(saturation_mean),
                'value_mean': float(value_mean),
                'medical_colors': medical_colors,
                'dominant_colors': dominant_colors,
                'is_grayscale': saturation_mean < 30,
                'is_high_contrast': np.std(hsv[:, :, 2]) > 50
            }
        else:
            value_mean = np.mean(image)
            medical_colors = {
                'white_ratio': 0.0,
                'blue_ratio': 0.0,
                'green_ratio': 0.0
            }

            return {
                'dominant_hue': 0,
                'saturation_mean': 0.0,
                'value_mean': float(value_mean),
                'medical_colors': medical_colors,
                'is_grayscale': True,
                'is_high_contrast': np.std(image) > 50,
                'dominant_colors': [[int(value_mean)] * 3]  # Grayscale as RGB
            }

    def _detect_medical_colors(self, hsv: npt.NDArray[np.uint8]) -> dict[str, float]:
        """Detect medical-related colors in HSV image."""
        blue_mask = cv2.inRange(hsv, np.array([100, 50, 50]), np.array([130, 255, 255]))
        green_mask = cv2.inRange(hsv, np.array([40, 50, 50]), np.array([80, 255, 255]))
        white_mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 30, 255]))

        total_pixels = hsv.shape[0] * hsv.shape[1]

        return {
            'blue_ratio': float(np.sum(blue_mask > 0) / total_pixels),
            'green_ratio': float(np.sum(green_mask > 0) / total_pixels),
            'white_ratio': float(np.sum(white_mask > 0) / total_pixels)
        }

    async def _calculate_edge_density(self, gray: npt.NDArray[np.uint8]) -> float:
        """Calculate edge density using Canny edge detection."""
        edges = cv2.Canny(gray, 50, 150)
        edge_pixels = np.sum(edges > 0)
        total_pixels = gray.shape[0] * gray.shape[1]
        return float(edge_pixels / total_pixels)

    async def _analyze_texture(self, gray: npt.NDArray[np.uint8]) -> dict[str, float]:
        """Analyze texture characteristics."""
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        return {
            'laplacian_variance': float(laplacian_var),
            'gradient_mean': float(np.mean(gradient_magnitude)),
            'gradient_std': float(np.std(gradient_magnitude))
        }

    async def _detect_grid_pattern(self, gray: npt.NDArray[np.uint8]) -> dict[str, Any]:
        """Detect grid patterns typical in medical documents."""
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))

        horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel)
        vertical_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, vertical_kernel)

        h_line_pixels = np.sum(horizontal_lines > 100)
        v_line_pixels = np.sum(vertical_lines > 100)
        total_pixels = gray.shape[0] * gray.shape[1]

        grid_score = (h_line_pixels + v_line_pixels) / total_pixels

        return {
            'grid_score': float(grid_score),
            'has_horizontal_lines': h_line_pixels > total_pixels * 0.01,
            'has_vertical_lines': v_line_pixels > total_pixels * 0.01,
            'likely_grid': grid_score > 0.02
        }

    async def _estimate_text_density(self, gray: npt.NDArray[np.uint8]) -> float:
        """Estimate text density in image."""
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        text_like_area = 0.0
        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0

            if 10 < area < 1000 and 0.1 < aspect_ratio < 10:
                text_like_area += area

        total_area = gray.shape[0] * gray.shape[1]
        return text_like_area / total_area

    async def _detect_medical_indicators(self, image: npt.NDArray[np.uint8]) -> dict[str, Any]:
        """Detect indicators that suggest medical content."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.uint8)
        else:
            gray = image.astype(np.uint8)

        waveform_score = await self._detect_waveforms(gray)

        scale_score = await self._detect_measurement_scales(gray)

        symbol_score = await self._detect_medical_symbols(gray)

        return {
            'waveform_score': waveform_score,
            'scale_score': scale_score,
            'symbol_score': symbol_score,
            'medical_likelihood': (waveform_score + scale_score + symbol_score) / 3
        }

    async def _detect_waveforms(self, gray: npt.NDArray[np.uint8]) -> float:
        """Detect waveform patterns in image."""
        row_means = np.mean(gray, axis=1)
        row_variance = np.var(row_means)

        normalized_variance = row_variance / (gray.shape[1] ** 2)

        return float(min(normalized_variance / 100, 1.0))

    async def _detect_measurement_scales(self, gray: npt.NDArray[np.uint8]) -> float:
        """Detect measurement scales or rulers."""
        edges = cv2.Canny(gray, 50, 150)

        vertical_projection = np.sum(edges, axis=0)

        if len(vertical_projection) > 10:
            autocorr = np.correlate(vertical_projection, vertical_projection, mode='full')
            autocorr = autocorr[autocorr.size // 2:]

            peaks = []
            for i in range(1, min(len(autocorr) - 1, 50)):
                if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
                    peaks.append(autocorr[i])

            if peaks:
                return float(min(len(peaks) / 10, 1.0))

        return 0.0

    async def _detect_medical_symbols(self, gray: npt.NDArray[np.uint8]) -> float:
        """Detect medical symbols or patterns."""
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=30,
            param1=50,
            param2=30,
            minRadius=5,
            maxRadius=50
        )

        if circles is not None:
            return float(min(len(circles[0]) / 10, 1.0))

        return 0.0  # type: ignore[unreachable]

    async def _detect_screen_indicators(self, image: npt.NDArray[np.uint8]) -> dict[str, Any]:
        """Detect indicators that image is a screen capture."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.uint8)
        else:
            gray = image.astype(np.uint8)

        ui_score = await self._detect_ui_elements(gray)

        screen_resolution = self._check_screen_resolution((image.shape[0], image.shape[1]))

        pixelation_score = await self._detect_pixelation(gray)

        return {
            'ui_score': ui_score,
            'screen_resolution': screen_resolution,
            'pixelation_score': pixelation_score,
            'screen_likelihood': (ui_score + screen_resolution + pixelation_score) / 3
        }

    async def _detect_ui_elements(self, gray: npt.NDArray[np.uint8]) -> float:
        """Detect UI elements like buttons, windows."""
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        ui_elements = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                if len(approx) == 4:  # Rectangular
                    ui_elements += 1

        return float(min(ui_elements / 20, 1.0))

    def _check_screen_resolution(self, shape: tuple[int, int]) -> float:
        """Check if image dimensions match common screen resolutions."""
        height, width = shape

        common_resolutions = [
            (1920, 1080), (1366, 768), (1280, 720), (1024, 768),
            (800, 600), (1440, 900), (1600, 900), (2560, 1440)
        ]

        for res_w, res_h in common_resolutions:
            if abs(width - res_w) < 50 and abs(height - res_h) < 50:
                return 1.0

        return 0.0

    async def _detect_pixelation(self, gray: npt.NDArray[np.uint8]) -> float:
        """Detect pixelation patterns typical in screenshots."""
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])

        return float(min(float(edge_density * 5), 1.0))

    async def _classify_features(self, features: dict[str, Any]) -> tuple[str, float]:
        """Classify image based on extracted features."""
        scores = {}

        scores['medical_other'] = await self._score_medical_other(features)
        scores['xray'] = await self._score_xray(features)
        scores['prescription'] = await self._score_prescription(features)
        scores['lab_results'] = await self._score_lab_results(features)
        scores['other_medical_device'] = await self._score_medical_device(features)
        scores['monitor_screen'] = await self._score_monitor_screen(features)

        scores['document'] = await self._score_document(features)
        scores['text_document'] = await self._score_text_document(features)
        scores['handwritten'] = await self._score_handwritten(features)

        scores['photo_person'] = await self._score_photo_person(features)
        scores['nature'] = await self._score_nature(features)
        scores['object'] = await self._score_object(features)
        scores['food'] = await self._score_food(features)

        scores['screenshot'] = await self._score_screenshot(features)

        best_category = max(scores.keys(), key=lambda k: scores[k])
        best_score = scores[best_category]

        threshold = self.confidence_thresholds.get(best_category, 0.5)
        if best_score < threshold:
            return 'unknown', best_score

        if best_score >= 1.0:
            best_score = 0.95

        return best_category, best_score

    async def _score_medical_other(self, features: dict[str, Any]) -> float:
        """Score likelihood of being another medical document."""
        score = 0.0

        medical_indicators = features.get('medical_indicators', {})
        score += medical_indicators.get('medical_likelihood', 0) * 0.5

        grid_pattern = features.get('grid_pattern', {})
        if grid_pattern.get('likely_grid', False):
            score += 0.4

        color_analysis = features.get('color_analysis', {})
        medical_colors = color_analysis.get('medical_colors', {})
        score += (medical_colors.get('blue_ratio', 0) + medical_colors.get('white_ratio', 0)) * 0.4

        text_density = features.get('text_density', 0)
        if text_density > 0.05:
            score += 0.3

        brightness = features.get('brightness', 0)
        if brightness > 200:
            score += 0.3

        contour_count = features.get('contour_count', 0)
        if contour_count > 10:
            score += 0.2

        return float(min(score, 1.0))

    async def _score_xray(self, features: dict[str, Any]) -> float:
        """Score likelihood of being an X-ray image."""
        score = 0.0

        color_analysis = features.get('color_analysis', {})

        if color_analysis.get('is_grayscale', False):
            score += 0.4

        if color_analysis.get('is_high_contrast', False):
            score += 0.3

        if color_analysis.get('value_mean', 255) < 100:
            score += 0.3

        return float(min(score, 1.0))

    async def _score_prescription(self, features: dict[str, Any]) -> float:
        """Score likelihood of being a prescription."""
        score = 0.0

        text_density = features.get('text_density', 0)
        if text_density > 0.1:
            score += 0.4

        aspect_ratio = features.get('aspect_ratio', 1)
        if 0.7 < aspect_ratio < 1.5:  # Roughly square or portrait
            score += 0.3

        medical_indicators = features.get('medical_indicators', {})
        score += medical_indicators.get('medical_likelihood', 0) * 0.3

        return float(min(score, 1.0))

    async def _score_lab_results(self, features: dict[str, Any]) -> float:
        """Score likelihood of being lab results."""
        score = 0.0

        grid_pattern = features.get('grid_pattern', {})
        if grid_pattern.get('likely_grid', False):
            score += 0.4

        text_density = features.get('text_density', 0)
        if text_density > 0.05:
            score += 0.3

        color_analysis = features.get('color_analysis', {})
        medical_colors = color_analysis.get('medical_colors', {})
        score += medical_colors.get('white_ratio', 0) * 0.3

        return float(min(score, 1.0))

    async def _score_medical_device(self, features: dict[str, Any]) -> float:
        """Score likelihood of being another medical device output."""
        score = 0.0

        medical_indicators = features.get('medical_indicators', {})
        score += medical_indicators.get('medical_likelihood', 0) * 0.5

        score += medical_indicators.get('waveform_score', 0) * 0.3

        score += medical_indicators.get('scale_score', 0) * 0.2

        return float(min(score, 1.0))

    async def _score_monitor_screen(self, features: dict[str, Any]) -> float:
        """Score likelihood of being a monitor screen capture."""
        score = 0.0

        screen_indicators = features.get('screen_indicators', {})
        score += screen_indicators.get('screen_likelihood', 0) * 0.6

        medical_indicators = features.get('medical_indicators', {})
        score += medical_indicators.get('medical_likelihood', 0) * 0.4

        return float(min(score, 1.0))

    async def _score_document(self, features: dict[str, Any]) -> float:
        """Score likelihood of being a general document."""
        score = 0.0

        text_density = features.get('text_density', 0)
        if text_density > 0.05:
            score += 0.4

        color_analysis = features.get('color_analysis', {})
        medical_colors = color_analysis.get('medical_colors', {})
        if medical_colors.get('white_ratio', 0) > 0.3:
            score += 0.3

        aspect_ratio = features.get('aspect_ratio', 1)
        if 0.7 < aspect_ratio < 1.5:
            score += 0.3

        return float(min(score, 1.0))

    async def _score_text_document(self, features: dict[str, Any]) -> float:
        """Score likelihood of being a text document."""
        score = 0.0

        text_density = features.get('text_density', 0)
        if text_density > 0.15:
            score += 0.6

        edge_density = features.get('edge_density', 0)
        if edge_density < 0.1:
            score += 0.2

        color_analysis = features.get('color_analysis', {})
        medical_colors = color_analysis.get('medical_colors', {})
        if medical_colors.get('white_ratio', 0) > 0.4:
            score += 0.2

        return float(min(score, 1.0))

    async def _score_handwritten(self, features: dict[str, Any]) -> float:
        """Score likelihood of being handwritten content."""
        score = 0.0

        text_density = features.get('text_density', 0)
        if 0.02 < text_density < 0.1:
            score += 0.4

        texture_analysis = features.get('texture_analysis', {})
        if texture_analysis.get('gradient_std', 0) > 20:
            score += 0.3

        edge_density = features.get('edge_density', 0)
        if 0.05 < edge_density < 0.15:
            score += 0.3

        return float(min(score, 1.0))

    async def _score_photo_person(self, features: dict[str, Any]) -> float:
        """Score likelihood of being a photo of a person."""
        score = 0.0

        color_analysis = features.get('color_analysis', {})
        dominant_hue = color_analysis.get('dominant_hue', 0)
        if 5 < dominant_hue < 25:
            score += 0.4

        saturation_mean = color_analysis.get('saturation_mean', 0)
        if 50 < saturation_mean < 150:
            score += 0.3

        text_density = features.get('text_density', 0)
        if text_density < 0.02:
            score += 0.3

        return float(min(score, 1.0))

    async def _score_nature(self, features: dict[str, Any]) -> float:
        """Score likelihood of being a nature/landscape photo."""
        score = 0.0

        color_analysis = features.get('color_analysis', {})
        dominant_hue = color_analysis.get('dominant_hue', 0)

        if 40 < dominant_hue < 130:  # Green to blue range
            score += 0.4

        saturation_mean = color_analysis.get('saturation_mean', 0)
        if saturation_mean > 100:
            score += 0.3

        text_density = features.get('text_density', 0)
        if text_density < 0.01:
            score += 0.2  # Reduced weight since food also has low text

        edge_density = features.get('edge_density', 0)
        if edge_density < 0.08:
            score += 0.2

        if dominant_hue < 30 or dominant_hue > 150:  # Red/orange/yellow range
            score -= 0.1

        return float(max(min(score, 1.0), 0.0))

    async def _score_object(self, features: dict[str, Any]) -> float:
        """Score likelihood of being a photo of an object."""
        score = 0.0

        edge_density = features.get('edge_density', 0)
        if 0.05 < edge_density < 0.2:
            score += 0.3

        text_density = features.get('text_density', 0)
        if text_density < 0.02:
            score += 0.3

        color_analysis = features.get('color_analysis', {})
        if not color_analysis.get('is_grayscale', False):
            score += 0.2

        size_category = features.get('size_category', '')
        if size_category in ['medium', 'large']:
            score += 0.2

        return float(min(score, 1.0))

    async def _score_food(self, features: dict[str, Any]) -> float:
        """Score likelihood of being a food photo."""
        score = 0.0

        color_analysis = features.get('color_analysis', {})
        dominant_hue = color_analysis.get('dominant_hue', 0)

        if dominant_hue < 30 or dominant_hue > 150:  # Red/orange/yellow range
            score += 0.6

        saturation_mean = color_analysis.get('saturation_mean', 0)
        if saturation_mean > 40:  # Lower threshold for food
            score += 0.5

        text_density = features.get('text_density', 0)
        if text_density < 0.01:
            score += 0.3

        edge_density = features.get('edge_density', 0)
        if 0.02 < edge_density < 0.15:  # Moderate edge density for food textures
            score += 0.2

        if not (40 < dominant_hue < 130):  # Not in green-blue nature range
            score += 0.2

        return float(min(score, 1.0))

    async def _score_screenshot(self, features: dict[str, Any]) -> float:
        """Score likelihood of being a screenshot."""
        score = 0.0

        screen_indicators = features.get('screen_indicators', {})
        score += screen_indicators.get('screen_likelihood', 0) * 0.7

        edge_density = features.get('edge_density', 0)
        if edge_density > 0.1:
            score += 0.3

        return float(min(score, 1.0))


non_ecg_classifier = NonECGImageClassifier()
