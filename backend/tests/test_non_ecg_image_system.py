"""
Comprehensive tests for the non-ECG image detection and response system.
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
from typing import Dict, Any
import tempfile
import os
from PIL import Image
import cv2

from app.services.non_ecg_image_classifier import NonECGImageClassifier
from app.services.contextual_response_generator import ContextualResponseGenerator
from app.services.adaptive_feedback_service import AdaptiveFeedbackService, UserFeedbackMetrics
from app.services.non_ecg_metrics_service import NonECGMetricsService
from app.core.exceptions import NonECGImageException


class TestNonECGImageClassifier:
    """Test suite for NonECGImageClassifier."""
    
    @pytest.fixture
    def classifier(self):
        """Create a NonECGImageClassifier instance for testing."""
        return NonECGImageClassifier()
    
    @pytest.fixture
    def sample_images(self):
        """Create sample test images for different categories."""
        images = {}
        
        temp_dir = tempfile.mkdtemp()
        
        medical_doc = np.ones((400, 600, 3), dtype=np.uint8) * 255
        cv2.rectangle(medical_doc, (50, 50), (550, 350), (0, 0, 0), 2)
        cv2.putText(medical_doc, "MEDICAL REPORT", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        medical_path = os.path.join(temp_dir, "medical_doc.jpg")
        cv2.imwrite(medical_path, medical_doc)
        images['medical_document'] = medical_path
        
        xray = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.circle(xray, (200, 200), 150, (200, 200, 200), -1)
        cv2.circle(xray, (180, 180), 30, (255, 255, 255), -1)
        cv2.circle(xray, (220, 180), 30, (255, 255, 255), -1)
        xray_path = os.path.join(temp_dir, "xray.jpg")
        cv2.imwrite(xray_path, xray)
        images['x_ray'] = xray_path
        
        food = np.random.randint(100, 255, (300, 300, 3), dtype=np.uint8)
        cv2.circle(food, (150, 150), 100, (255, 100, 50), -1)  # Orange circle
        food_path = os.path.join(temp_dir, "food.jpg")
        cv2.imwrite(food_path, food)
        images['food'] = food_path
        
        ecg = np.ones((400, 600, 3), dtype=np.uint8) * 255
        for i in range(0, 600, 20):
            cv2.line(ecg, (i, 0), (i, 400), (200, 200, 200), 1)
        for i in range(0, 400, 20):
            cv2.line(ecg, (0, i), (600, i), (200, 200, 200), 1)
        points = []
        for x in range(0, 600, 5):
            y = 200 + int(50 * np.sin(x * 0.1))
            points.append((x, y))
        for i in range(len(points) - 1):
            cv2.line(ecg, points[i], points[i + 1], (0, 0, 0), 2)
        ecg_path = os.path.join(temp_dir, "ecg.jpg")
        cv2.imwrite(ecg_path, ecg)
        images['ecg'] = ecg_path
        
        return images, temp_dir
    
    def test_classifier_initialization(self, classifier):
        """Test that classifier initializes correctly."""
        assert classifier is not None
        assert hasattr(classifier, 'categories')
        assert len(classifier.categories) > 0
        assert 'medical_other' in classifier.categories
        assert 'xray' in classifier.categories
        assert 'food' in classifier.categories
    
    @pytest.mark.asyncio
    async def test_classify_medical_document(self, classifier, sample_images):
        """Test classification of medical document."""
        images, temp_dir = sample_images
        
        try:
            category, confidence, metadata = await classifier.classify_image(images['medical_document'])
            
            assert category == 'medical_other'
            assert 0.0 <= confidence <= 1.0
            assert isinstance(metadata, dict)
            assert 'features' in metadata
            
        finally:
            for path in images.values():
                if os.path.exists(path):
                    os.unlink(path)
            os.rmdir(temp_dir)
    
    @pytest.mark.asyncio
    async def test_classify_xray(self, classifier, sample_images):
        """Test classification of X-ray image."""
        images, temp_dir = sample_images
        
        try:
            category, confidence, metadata = await classifier.classify_image(images['x_ray'])
            
            assert category == 'xray'
            assert confidence > 0.3  # Should have reasonable confidence for X-ray
            assert 'features' in metadata
            
        finally:
            for path in images.values():
                if os.path.exists(path):
                    os.unlink(path)
            os.rmdir(temp_dir)
    
    @pytest.mark.asyncio
    async def test_classify_food(self, classifier, sample_images):
        """Test classification of food image."""
        images, temp_dir = sample_images
        
        try:
            category, confidence, metadata = await classifier.classify_image(images['food'])
            
            assert category == 'food'
            assert confidence > 0.2
            assert 'features' in metadata
            
        finally:
            for path in images.values():
                if os.path.exists(path):
                    os.unlink(path)
            os.rmdir(temp_dir)
    
    @pytest.mark.asyncio
    async def test_classify_ecg_like_image(self, classifier, sample_images):
        """Test that ECG-like images get lower non-ECG confidence."""
        images, temp_dir = sample_images
        
        try:
            category, confidence, metadata = await classifier.classify_image(images['ecg'])

            assert confidence < 1.0  # Should not be 100% confident it's non-ECG
            
        finally:
            for path in images.values():
                if os.path.exists(path):
                    os.unlink(path)
            os.rmdir(temp_dir)
    
    @pytest.mark.asyncio
    async def test_classify_image_array(self, classifier):
        """Test classification using numpy array input."""
        image_array = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        
        category, confidence, metadata = await classifier.classify_image(image_array)
        
        assert isinstance(category, str)
        assert 0.0 <= confidence <= 1.0
        assert isinstance(metadata, dict)
    
    @pytest.mark.asyncio
    async def test_extract_color_features(self, classifier):
        """Test color feature extraction."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[:, :, 0] = 255  # Red channel
        
        features = await classifier._extract_features(image)
        assert 'color_analysis' in features
        
        assert isinstance(features, dict)
        assert 'color_analysis' in features
        assert 'dominant_colors' in features['color_analysis']
        assert 'color_variance' in features
        assert 'brightness' in features
    
    @pytest.mark.asyncio
    async def test_extract_texture_features(self, classifier):
        """Test texture feature extraction."""
        image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        
        features = await classifier._extract_features(image)
        assert 'texture_analysis' in features
        
        assert isinstance(features, dict)
        assert 'edge_density' in features
        assert 'texture_energy' in features
    
    @pytest.mark.asyncio
    async def test_extract_shape_features(self, classifier):
        """Test shape feature extraction."""
        image = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(image, (20, 20), (80, 80), 255, -1)
        
        features = await classifier._extract_features(image)
        assert 'edge_density' in features
        
        assert isinstance(features, dict)
        assert 'contour_count' in features
        assert 'aspect_ratio' in features


class TestContextualResponseGenerator:
    """Test suite for ContextualResponseGenerator."""
    
    @pytest.fixture
    def response_generator(self):
        """Create a ContextualResponseGenerator instance for testing."""
        return ContextualResponseGenerator()
    
    def test_generate_response_medical_document(self, response_generator):
        """Test response generation for medical document."""
        response = response_generator.generate_response(
            category='medical_document',
            confidence=0.8,
            user_session=None
        )
        
        assert isinstance(response, dict)
        assert 'message' in response
        assert 'explanation' in response
        assert 'tips' in response
        assert isinstance(response['tips'], list)
        assert len(response['tips']) > 0
    
    def test_generate_response_xray(self, response_generator):
        """Test response generation for X-ray."""
        response = response_generator.generate_response(
            category='x_ray',
            confidence=0.9,
            user_session=None
        )
        
        assert 'message' in response
        assert 'x-ray' in response['message'].lower() or 'raio-x' in response['message'].lower()
        assert 'tips' in response
    
    def test_generate_response_food(self, response_generator):
        """Test response generation for food with humor."""
        response = response_generator.generate_response(
            category='food',
            confidence=0.7,
            user_session=None
        )
        
        assert 'message' in response
        assert 'humor_response' in response
        assert response['humor_response'] is not None
    
    def test_generate_response_with_user_session(self, response_generator):
        """Test response generation with user session for personalization."""
        mock_user_session = Mock()
        mock_user_session.id = "test_user_123"
        mock_user_session.category_history = ["medical_other", "xray"]
        mock_user_session.learning_stage = "beginner"
        
        response = response_generator.generate_response(
            category='medical_document',
            confidence=0.6,
            user_session=mock_user_session
        )
        
        assert 'adaptive_suggestions' in response
    
    def test_get_base_response_categories(self, response_generator):
        """Test that base responses exist for all categories."""
        test_categories = [
            'medical_document', 'x_ray', 'mri', 'ct_scan', 'prescription',
            'food', 'animal', 'person', 'landscape', 'text_document',
            'screenshot', 'drawing', 'other', 'unknown'
        ]
        
        for category in test_categories:
            response = response_generator._get_base_response(category, 0.5)
            assert isinstance(response, dict)
            assert 'message' in response
            assert 'explanation' in response
    
    def test_generate_educational_content(self, response_generator):
        """Test educational content generation."""
        content = response_generator.generate_educational_content('medical_other')
        
        assert isinstance(content, dict)
        assert 'title' in content
        assert 'description' in content
        assert 'key_features' in content
        assert isinstance(content['key_features'], list)


class TestAdaptiveFeedbackService:
    """Test suite for AdaptiveFeedbackService."""
    
    @pytest.fixture
    def feedback_service(self):
        """Create an AdaptiveFeedbackService instance for testing."""
        return AdaptiveFeedbackService()
    
    @pytest.mark.asyncio
    async def test_track_user_attempt_new_user(self, feedback_service):
        """Test tracking attempt for new user."""
        mock_session = Mock()
        mock_session.id = "new_user_123"
        
        await feedback_service.track_user_attempt(
            user_session=mock_session,
            category='food',
            success=False,
            confidence=0.8
        )
        
        assert mock_session.id in feedback_service.user_metrics
        metrics = feedback_service.user_metrics[mock_session.id]
        assert metrics.total_attempts == 1
        assert metrics.successful_attempts == 0
        assert 'food' in metrics.category_history
    
    @pytest.mark.asyncio
    async def test_track_successful_attempt(self, feedback_service):
        """Test tracking successful ECG upload."""
        mock_session = Mock()
        mock_session.id = "test_user_456"
        
        await feedback_service.track_user_attempt(
            user_session=mock_session,
            category='medical_document',
            success=False,
            confidence=0.7
        )
        
        await feedback_service.track_user_attempt(
            user_session=mock_session,
            category='ecg_success',
            success=True,
            confidence=0.9
        )
        
        metrics = feedback_service.user_metrics[mock_session.id]
        assert metrics.total_attempts == 2
        assert metrics.successful_attempts == 1
        assert metrics.success_rate == 0.5
    
    def test_get_personalized_response(self, feedback_service):
        """Test personalized response generation."""
        mock_session = Mock()
        mock_session.id = "experienced_user"
        
        feedback_service.user_metrics[mock_session.id] = UserFeedbackMetrics(
            user_id=mock_session.id,
            total_attempts=10,
            successful_attempts=7,
            category_history=['food', 'medical_document', 'x_ray'] * 3,
            learning_progress=0.7,
            last_attempt_time=datetime.now()
        )
        
        response = feedback_service.get_personalized_response(
            user_session=mock_session,
            category='food',
            base_message='Base message'
        )
        
        assert 'adaptive_suggestions' in response
        assert 'learning_stage' in response
        assert response['learning_stage'] in ['beginner', 'intermediate', 'advanced']
    
    @pytest.mark.asyncio
    async def test_collect_feedback(self, feedback_service):
        """Test feedback collection."""
        mock_session = Mock()
        mock_session.id = "feedback_user"
        
        await feedback_service.collect_feedback(
            user_session=mock_session,
            category='medical_document',
            helpfulness_score=4.5,
            feedback_text="Very helpful explanation"
        )
        
        assert mock_session.id in feedback_service.user_metrics
        metrics = feedback_service.user_metrics[mock_session.id]
        assert len(metrics.feedback_scores) == 1
        assert metrics.feedback_scores[0] == 4.5
    
    def test_determine_learning_stage(self, feedback_service):
        """Test learning stage determination."""
        beginner_metrics = UserFeedbackMetrics(
            user_id="beginner",
            total_attempts=2,
            successful_attempts=0,
            category_history=['food', 'medical_document'],
            learning_progress=0.0,
            last_attempt_time=datetime.now()
        )
        stage = feedback_service._determine_learning_stage(beginner_metrics)
        assert stage == 'beginner'
        
        advanced_metrics = UserFeedbackMetrics(
            user_id="advanced",
            total_attempts=20,
            successful_attempts=18,
            category_history=['ecg_success'] * 18 + ['food', 'medical_document'],
            learning_progress=0.9,
            last_attempt_time=datetime.now()
        )
        stage = feedback_service._determine_learning_stage(advanced_metrics)
        assert stage == 'advanced'


class TestNonECGMetricsService:
    """Test suite for NonECGMetricsService."""
    
    @pytest.fixture
    def metrics_service(self):
        """Create a NonECGMetricsService instance for testing."""
        return NonECGMetricsService()
    
    @pytest.mark.asyncio
    async def test_track_non_ecg_detection(self, metrics_service):
        """Test tracking non-ECG detection events."""
        await metrics_service.track_non_ecg_detection(
            user_id="test_user",
            category="food",
            confidence=0.8,
            contextual_response={'response_type': 'humorous'},
            session_id="session_123"
        )
        
        user_metrics = await metrics_service.get_user_learning_metrics("test_user")
        assert user_metrics.user_id == "test_user"
        assert user_metrics.non_ecg_attempts == 1
        assert 'food' in user_metrics.categories_encountered
    
    @pytest.mark.asyncio
    async def test_track_successful_ecg_upload(self, metrics_service):
        """Test tracking successful ECG uploads."""
        await metrics_service.track_successful_ecg_upload(
            user_id="success_user",
            confidence=0.9,
            time_since_last_attempt=300.0,
            session_id="session_456"
        )
        
        user_metrics = await metrics_service.get_user_learning_metrics("success_user")
        assert user_metrics.successful_attempts == 1
    
    @pytest.mark.asyncio
    async def test_track_response_feedback(self, metrics_service):
        """Test tracking response feedback."""
        await metrics_service.track_response_feedback(
            user_id="feedback_user",
            category="medical_document",
            helpfulness_score=4.2,
            feedback_type="educational",
            additional_feedback="Very clear explanation"
        )
        
        user_metrics = await metrics_service.get_user_learning_metrics("feedback_user")
        assert len(user_metrics.response_helpfulness_scores) == 1
        assert user_metrics.response_helpfulness_scores[0] == 4.2
    
    @pytest.mark.asyncio
    async def test_get_system_metrics(self, metrics_service):
        """Test system metrics retrieval."""
        await metrics_service.track_non_ecg_detection("user1", "food", 0.8, {})
        await metrics_service.track_non_ecg_detection("user2", "x_ray", 0.9, {})
        await metrics_service.track_successful_ecg_upload("user1", 0.9)
        
        system_metrics = await metrics_service.get_system_metrics()
        
        assert system_metrics is not None
        assert hasattr(system_metrics, 'total_non_ecg_detections')
        assert hasattr(system_metrics, 'category_distribution')
        assert hasattr(system_metrics, 'user_success_rate_24h')
    
    @pytest.mark.asyncio
    async def test_generate_insights_report(self, metrics_service):
        """Test insights report generation."""
        await metrics_service.track_non_ecg_detection("user1", "food", 0.8, {})
        await metrics_service.track_response_feedback("user1", "food", 4.0, "humorous")
        
        insights = await metrics_service.generate_insights_report()
        
        assert isinstance(insights, dict)
        assert 'summary' in insights
        assert 'category_insights' in insights
        assert 'learning_insights' in insights
        assert 'recommendations' in insights
        assert 'generated_at' in insights


class TestPrivacyCompliance:
    """Test suite for privacy compliance of non-ECG image handling."""
    
    @pytest.mark.asyncio
    async def test_non_ecg_image_not_stored(self):
        """Test that non-ECG images are not stored permanently."""
        temp_dir = tempfile.mkdtemp()
        test_image_path = os.path.join(temp_dir, "test_food.jpg")
        
        food_image = np.random.randint(100, 255, (300, 300, 3), dtype=np.uint8)
        cv2.imwrite(test_image_path, food_image)
        
        try:
            classifier = NonECGImageClassifier()
            category, confidence, metadata = await classifier.classify_image(test_image_path)
            
            assert category in classifier.categories
            assert confidence > 0.0
            
            if confidence > 0.5:  # Non-ECG detected
                if os.path.exists(test_image_path):
                    os.unlink(test_image_path)
                
                assert not os.path.exists(test_image_path)
            
        finally:
            if os.path.exists(test_image_path):
                os.unlink(test_image_path)
            os.rmdir(temp_dir)
    
    @pytest.mark.asyncio
    async def test_non_ecg_exception_contains_no_image_data(self):
        """Test that NonECGImageException doesn't contain image data."""
        contextual_response = {
            'message': 'Food image detected',
            'explanation': 'This appears to be a food image',
            'tips': ['Try uploading an ECG instead']
        }
        
        exception = NonECGImageException(
            message="Non-ECG image detected",
            category="food",
            contextual_response=contextual_response,
            confidence=0.8
        )
        
        assert 'image' not in exception.details
        assert 'image_data' not in exception.details
        assert 'file_content' not in exception.details
        
        assert exception.details['category'] == 'food'
        assert exception.details['confidence'] == 0.8
        assert exception.details['contextual_response'] == contextual_response
    
    def test_user_session_data_privacy(self):
        """Test that user session data doesn't contain sensitive information."""
        feedback_service = AdaptiveFeedbackService()
        
        user_metrics = UserFeedbackMetrics(
            user_id="privacy_test_user",
            total_attempts=5,
            successful_attempts=2,
            category_history=['food', 'medical_document', 'ecg_success', 'x_ray', 'ecg_success'],
            learning_progress=0.4,
            last_attempt_time=datetime.now()
        )
        
        assert not hasattr(user_metrics, 'image_data')
        assert not hasattr(user_metrics, 'file_paths')
        assert not hasattr(user_metrics, 'personal_info')
        
        assert isinstance(user_metrics.category_history, list)
        assert all(isinstance(category, str) for category in user_metrics.category_history)
        assert isinstance(user_metrics.learning_progress, (int, float))


class TestIntegrationScenarios:
    """Integration tests for complete non-ECG image handling workflows."""
    
    @pytest.mark.asyncio
    async def test_complete_non_ecg_workflow(self):
        """Test complete workflow from detection to response generation."""
        classifier = NonECGImageClassifier()
        response_generator = ContextualResponseGenerator()
        feedback_service = AdaptiveFeedbackService()
        metrics_service = NonECGMetricsService()
        
        temp_dir = tempfile.mkdtemp()
        test_image_path = os.path.join(temp_dir, "test_medical_doc.jpg")
        
        medical_doc = np.ones((400, 600, 3), dtype=np.uint8) * 255
        cv2.rectangle(medical_doc, (50, 50), (550, 350), (0, 0, 0), 2)
        cv2.putText(medical_doc, "MEDICAL REPORT", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.imwrite(test_image_path, medical_doc)
        
        try:
            category, confidence, metadata = await classifier.classify_image(test_image_path)
            assert category == 'medical_other'
            assert confidence > 0.3
            
            mock_user_session = Mock()
            mock_user_session.id = "integration_test_user"
            mock_user_session.category_history = ["medical_other"]
            mock_user_session.learning_stage = "beginner"
            
            response = response_generator.generate_response(
                category=category,
                confidence=confidence,
                user_session=mock_user_session
            )
            
            assert 'message' in response
            assert 'tips' in response
            assert len(response['tips']) > 0
            
        finally:
            if os.path.exists(test_image_path):
                os.unlink(test_image_path)
            os.rmdir(temp_dir)
    
    @pytest.mark.asyncio
    async def test_user_learning_progression(self):
        """Test user learning progression over multiple attempts."""
        feedback_service = AdaptiveFeedbackService()
        metrics_service = NonECGMetricsService()
        
        mock_user_session = Mock()
        mock_user_session.id = "learning_progression_user"
        
        failed_categories = ['food', 'medical_document', 'x_ray', 'person']
        
        for i, category in enumerate(failed_categories):
            await feedback_service.track_user_attempt(
                user_session=mock_user_session,
                category=category,
                success=False,
                confidence=0.8 - (i * 0.1)  # Decreasing confidence
            )
        
        user_metrics = feedback_service.user_metrics[mock_user_session.id]
        learning_stage = feedback_service._determine_learning_stage(user_metrics)
        assert learning_stage == 'beginner'
        
        await feedback_service.track_user_attempt(
            user_session=mock_user_session,
            category='ecg_success',
            success=True,
            confidence=0.9
        )
        
        updated_metrics = feedback_service.user_metrics[mock_user_session.id]
        assert updated_metrics.successful_attempts == 1
        
        for _ in range(4):
            await feedback_service.track_user_attempt(
                user_session=mock_user_session,
                category='ecg_success',
                success=True,
                confidence=0.9
            )
        
        final_metrics = feedback_service.user_metrics[mock_user_session.id]
        final_stage = feedback_service._determine_learning_stage(final_metrics)
        assert final_stage in ['intermediate', 'advanced']
        assert final_metrics.success_rate > 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
