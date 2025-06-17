#!/usr/bin/env python3
"""
Script para criar testes críticos com 100% de cobertura para módulos essenciais do CardioAI Pro.
Foca em módulos de segurança, processamento de ECG e validação médica.
"""

import os
from pathlib import Path

class CriticalTestsCreator:
    def __init__(self):
        self.backend_dir = Path.cwd() / "backend"
        self.tests_dir = self.backend_dir / "tests"
        self.tests_dir.mkdir(exist_ok=True)
        
    def create_all_critical_tests(self):
        """Cria todos os testes críticos necessários."""
        print("🎯 Criando testes críticos para 100% de cobertura\n")
        
        # 1. Testes de segurança (crítico)
        self.create_security_tests()
        
        # 2. Testes de processamento ECG (crítico)
        self.create_ecg_processing_tests()
        
        # 3. Testes de validação médica (crítico)
        self.create_medical_validation_tests()
        
        # 4. Testes de serviços principais
        self.create_core_services_tests()
        
        # 5. Testes de utilidades
        self.create_utils_tests()
        
        print("\n✅ Todos os testes críticos foram criados!")
        
    def create_security_tests(self):
        """Cria testes completos para módulo de segurança."""
        print("🔐 Criando testes de segurança...")
        
        security_test = self.tests_dir / "test_security_complete.py"
        
        content = '''"""Testes completos para módulo de segurança - 100% cobertura."""

import pytest
from datetime import datetime, timedelta
from jose import jwt, JWTError
from app.core.security import (
    create_access_token,
    decode_access_token,
    verify_password,
    get_password_hash,
    create_refresh_token,
    verify_refresh_token,
    generate_password_reset_token,
    verify_password_reset_token,
)
from app.core.config import settings


class TestSecurityComplete:
    """Testes completos para funções de segurança."""
    
    def test_password_hashing_and_verification(self):
        """Testa hash e verificação de senha."""
        password = "TestPassword123!"
        
        # Criar hash
        hashed = get_password_hash(password)
        assert hashed != password
        assert hashed.startswith("$2b$")
        
        # Verificar senha correta
        assert verify_password(password, hashed) is True
        
        # Verificar senha incorreta
        assert verify_password("WrongPassword", hashed) is False
        
        # Verificar com hash inválido
        assert verify_password(password, "invalid_hash") is False
        
    def test_access_token_creation_and_decoding(self):
        """Testa criação e decodificação de token de acesso."""
        user_id = "user123"
        user_role = "physician"
        
        # Criar token
        token = create_access_token(
            data={"sub": user_id, "role": user_role}
        )
        assert token is not None
        assert isinstance(token, str)
        
        # Decodificar token válido
        payload = decode_access_token(token)
        assert payload is not None
        assert payload.get("sub") == user_id
        assert payload.get("role") == user_role
        
        # Token expirado
        expired_token = create_access_token(
            data={"sub": user_id},
            expires_delta=timedelta(seconds=-1)
        )
        payload = decode_access_token(expired_token)
        assert payload is None
        
        # Token inválido
        assert decode_access_token("invalid.token.here") is None
        
    def test_refresh_token_functionality(self):
        """Testa funcionalidade de refresh token."""
        user_id = "user456"
        
        # Criar refresh token
        refresh_token = create_refresh_token(data={"sub": user_id})
        assert refresh_token is not None
        
        # Verificar refresh token válido
        payload = verify_refresh_token(refresh_token)
        assert payload is not None
        assert payload.get("sub") == user_id
        assert payload.get("type") == "refresh"
        
        # Refresh token inválido
        assert verify_refresh_token("invalid.refresh.token") is None
        
    def test_password_reset_token(self):
        """Testa tokens de reset de senha."""
        email = "user@example.com"
        
        # Gerar token de reset
        reset_token = generate_password_reset_token(email)
        assert reset_token is not None
        assert isinstance(reset_token, str)
        
        # Verificar token válido
        verified_email = verify_password_reset_token(reset_token)
        assert verified_email == email
        
        # Token inválido
        assert verify_password_reset_token("invalid.reset.token") is None
        
        # Token expirado (simular)
        expired_token = jwt.encode(
            {"sub": email, "exp": datetime.utcnow() - timedelta(hours=1)},
            settings.SECRET_KEY,
            algorithm=settings.ALGORITHM
        )
        assert verify_password_reset_token(expired_token) is None
        
    def test_token_with_additional_claims(self):
        """Testa tokens com claims adicionais."""
        data = {
            "sub": "user789",
            "role": "admin",
            "permissions": ["read", "write", "delete"],
            "department": "cardiology"
        }
        
        token = create_access_token(data=data)
        payload = decode_access_token(token)
        
        assert payload["sub"] == data["sub"]
        assert payload["role"] == data["role"]
        assert payload["permissions"] == data["permissions"]
        assert payload["department"] == data["department"]
        
    def test_token_expiration_times(self):
        """Testa diferentes tempos de expiração."""
        user_id = "user999"
        
        # Token com expiração personalizada
        custom_expires = timedelta(minutes=5)
        token = create_access_token(
            data={"sub": user_id},
            expires_delta=custom_expires
        )
        
        payload = decode_access_token(token)
        assert payload is not None
        
        # Verificar que exp está presente
        assert "exp" in payload
        
    def test_security_edge_cases(self):
        """Testa casos extremos de segurança."""
        # Hash de senha vazia
        empty_hash = get_password_hash("")
        assert empty_hash is not None
        assert verify_password("", empty_hash) is True
        
        # Token com dados vazios
        empty_token = create_access_token(data={})
        empty_payload = decode_access_token(empty_token)
        assert empty_payload is not None
        
        # Verificar algoritmo incorreto
        wrong_algo_token = jwt.encode(
            {"sub": "test"},
            settings.SECRET_KEY,
            algorithm="HS512"  # Algoritmo diferente
        )
        assert decode_access_token(wrong_algo_token) is None


# Testes de integração com autenticação
class TestAuthenticationFlow:
    """Testa fluxo completo de autenticação."""
    
    def test_complete_auth_flow(self):
        """Testa fluxo completo de autenticação."""
        # 1. Registrar usuário (simulado)
        password = "SecurePass123!"
        hashed_password = get_password_hash(password)
        
        # 2. Login - verificar senha
        assert verify_password(password, hashed_password)
        
        # 3. Gerar tokens
        user_data = {"sub": "user@example.com", "role": "physician"}
        access_token = create_access_token(data=user_data)
        refresh_token = create_refresh_token(data=user_data)
        
        # 4. Usar access token
        payload = decode_access_token(access_token)
        assert payload["sub"] == user_data["sub"]
        
        # 5. Renovar com refresh token
        refresh_payload = verify_refresh_token(refresh_token)
        assert refresh_payload["sub"] == user_data["sub"]
        
        # 6. Gerar novo access token
        new_access_token = create_access_token(data=user_data)
        assert new_access_token != access_token
'''
        
        with open(security_test, 'w', encoding='utf-8') as f:
            f.write(content)
            
        print("✅ Testes de segurança criados!")
        
    def create_ecg_processing_tests(self):
        """Cria testes completos para processamento de ECG."""
        print("💓 Criando testes de processamento ECG...")
        
        ecg_test = self.tests_dir / "test_ecg_processing_complete.py"
        
        content = '''"""Testes completos para processamento de ECG - 100% cobertura."""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from app.utils.ecg_processor import ECGProcessor
from app.utils.signal_quality import SignalQualityAnalyzer
from app.core.exceptions import ECGProcessingException


class TestECGProcessingComplete:
    """Testes completos para processamento de ECG."""
    
    @pytest.fixture
    def ecg_processor(self):
        """Fixture para ECGProcessor."""
        return ECGProcessor()
        
    @pytest.fixture
    def signal_analyzer(self):
        """Fixture para SignalQualityAnalyzer."""
        return SignalQualityAnalyzer()
        
    @pytest.fixture
    def sample_ecg_signal(self):
        """Sinal ECG de amostra."""
        # Simular sinal ECG de 12 derivações, 10 segundos a 500Hz
        duration = 10  # segundos
        sampling_rate = 500  # Hz
        n_samples = duration * sampling_rate
        n_leads = 12
        
        # Gerar sinal com componentes típicos de ECG
        t = np.linspace(0, duration, n_samples)
        signal = np.zeros((n_leads, n_samples))
        
        for lead in range(n_leads):
            # Componente base (ritmo sinusal ~1Hz)
            signal[lead] = 0.5 * np.sin(2 * np.pi * 1.0 * t)
            # Adicionar complexo QRS simulado
            qrs_locations = np.arange(0, n_samples, sampling_rate)
            for qrs_loc in qrs_locations:
                if qrs_loc < n_samples - 50:
                    signal[lead, qrs_loc:qrs_loc+50] += np.random.normal(1.5, 0.2)
            # Adicionar ruído realista
            signal[lead] += np.random.normal(0, 0.05, n_samples)
            
        return signal
        
    def test_ecg_preprocessing(self, ecg_processor, sample_ecg_signal):
        """Testa pré-processamento de ECG."""
        # Processar sinal
        processed = ecg_processor.preprocess_signal(
            sample_ecg_signal,
            sampling_rate=500
        )
        
        assert processed is not None
        assert processed.shape == sample_ecg_signal.shape
        assert not np.array_equal(processed, sample_ecg_signal)  # Deve ser diferente
        
        # Testar com diferentes taxas de amostragem
        for rate in [250, 500, 1000]:
            result = ecg_processor.preprocess_signal(sample_ecg_signal, rate)
            assert result is not None
            
    def test_feature_extraction(self, ecg_processor, sample_ecg_signal):
        """Testa extração de características."""
        features = ecg_processor.extract_features(
            sample_ecg_signal,
            sampling_rate=500
        )
        
        assert isinstance(features, dict)
        assert "heart_rate" in features
        assert "pr_interval" in features
        assert "qrs_duration" in features
        assert "qt_interval" in features
        assert "qt_corrected" in features
        assert "rr_intervals" in features
        assert "hrv_metrics" in features
        
        # Validar ranges
        assert 40 <= features["heart_rate"] <= 200
        assert 0 <= features["pr_interval"] <= 400
        assert 0 <= features["qrs_duration"] <= 200
        assert 0 <= features["qt_interval"] <= 600
        
    def test_signal_quality_assessment(self, signal_analyzer, sample_ecg_signal):
        """Testa avaliação de qualidade do sinal."""
        quality = signal_analyzer.assess_quality(
            sample_ecg_signal,
            sampling_rate=500
        )
        
        assert isinstance(quality, dict)
        assert "overall_quality" in quality
        assert "snr" in quality
        assert "baseline_wander" in quality
        assert "power_line_interference" in quality
        assert "muscle_artifact" in quality
        assert "lead_quality" in quality
        
        # Validar scores
        assert 0 <= quality["overall_quality"] <= 1
        assert quality["snr"] > 0
        assert isinstance(quality["lead_quality"], list)
        assert len(quality["lead_quality"]) == sample_ecg_signal.shape[0]
        
    def test_arrhythmia_detection(self, ecg_processor, sample_ecg_signal):
        """Testa detecção de arritmias."""
        # Mock do modelo de ML
        with patch.object(ecg_processor, 'ml_model') as mock_model:
            mock_model.predict.return_value = np.array([[0.1, 0.8, 0.1]])
            
            arrhythmias = ecg_processor.detect_arrhythmias(
                sample_ecg_signal,
                sampling_rate=500
            )
            
            assert isinstance(arrhythmias, list)
            assert all(isinstance(a, dict) for a in arrhythmias)
            
            if arrhythmias:
                assert all("type" in a for a in arrhythmias)
                assert all("confidence" in a for a in arrhythmias)
                assert all("location" in a for a in arrhythmias)
                
    def test_ecg_processing_errors(self, ecg_processor):
        """Testa tratamento de erros no processamento."""
        # Sinal inválido
        with pytest.raises(ECGProcessingException):
            ecg_processor.preprocess_signal(None, 500)
            
        # Taxa de amostragem inválida
        with pytest.raises(ValueError):
            ecg_processor.preprocess_signal(
                np.zeros((12, 1000)), 
                sampling_rate=0
            )
            
        # Sinal com dimensões incorretas
        with pytest.raises(ECGProcessingException):
            ecg_processor.preprocess_signal(
                np.zeros((13, 1000)),  # 13 derivações (inválido)
                sampling_rate=500
            )
            
    def test_signal_quality_edge_cases(self, signal_analyzer):
        """Testa casos extremos de qualidade de sinal."""
        # Sinal completamente ruidoso
        noisy_signal = np.random.normal(0, 1, (12, 5000))
        quality = signal_analyzer.assess_quality(noisy_signal, 500)
        assert quality["overall_quality"] < 0.5
        
        # Sinal perfeito (sintético)
        t = np.linspace(0, 10, 5000)
        perfect_signal = np.sin(2 * np.pi * 1.0 * t)
        perfect_signal = np.tile(perfect_signal, (12, 1))
        quality = signal_analyzer.assess_quality(perfect_signal, 500)
        assert quality["overall_quality"] > 0.7
        
        # Sinal com eletrodo desconectado
        disconnected = np.zeros((12, 5000))
        disconnected[5, :] = np.random.normal(0, 10, 5000)  # Lead 6 desconectado
        quality = signal_analyzer.assess_quality(disconnected, 500)
        assert quality["lead_quality"][5] < 0.3
        
    def test_complete_ecg_pipeline(self, ecg_processor, sample_ecg_signal):
        """Testa pipeline completo de processamento ECG."""
        # Pipeline completo
        result = ecg_processor.process_complete(
            sample_ecg_signal,
            sampling_rate=500,
            patient_age=45,
            patient_gender="M"
        )
        
        assert isinstance(result, dict)
        assert "preprocessed_signal" in result
        assert "features" in result
        assert "quality_assessment" in result
        assert "detected_conditions" in result
        assert "clinical_interpretation" in result
        assert "recommendations" in result
        
        # Verificar interpretação clínica
        interpretation = result["clinical_interpretation"]
        assert isinstance(interpretation, str)
        assert len(interpretation) > 0
        
        # Verificar recomendações
        recommendations = result["recommendations"]
        assert isinstance(recommendations, list)
'''
        
        with open(ecg_test, 'w', encoding='utf-8') as f:
            f.write(content)
            
        print("✅ Testes de processamento ECG criados!")
        
    def create_medical_validation_tests(self):
        """Cria testes para validação médica."""
        print("🏥 Criando testes de validação médica...")
        
        validation_test = self.tests_dir / "test_medical_validation_complete.py"
        
        content = '''"""Testes completos para validação médica - 100% cobertura."""

import pytest
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch
from app.services.validation_service import ValidationService
from app.core.constants import ValidationStatus, ClinicalUrgency, UserRoles
from app.models.validation import ValidationModel, ClinicalValidation
from app.core.exceptions import ValidationException, PermissionDeniedException


class TestMedicalValidationComplete:
    """Testes completos para validação médica."""
    
    @pytest.fixture
    def validation_service(self):
        """Mock do serviço de validação."""
        mock_db = AsyncMock()
        mock_notification = AsyncMock()
        return ValidationService(
            db=mock_db,
            notification_service=mock_notification
        )
        
    @pytest.mark.asyncio
    async def test_create_validation_request(self, validation_service):
        """Testa criação de solicitação de validação."""
        analysis_id = "analysis_123"
        urgency = ClinicalUrgency.URGENT
        
        # Mock do repositório
        validation_service.repository.create_validation = AsyncMock(
            return_value=ValidationModel(
                id=1,
                analysis_id=analysis_id,
                status=ValidationStatus.PENDING,
                clinical_urgency=urgency,
                created_at=datetime.utcnow()
            )
        )
        
        result = await validation_service.create_validation_request(
            analysis_id=analysis_id,
            urgency=urgency,
            requested_by="user_123"
        )
        
        assert result is not None
        assert result.analysis_id == analysis_id
        assert result.status == ValidationStatus.PENDING
        assert result.clinical_urgency == urgency
        
    @pytest.mark.asyncio
    async def test_submit_clinical_validation(self, validation_service):
        """Testa submissão de validação clínica."""
        validation_id = 1
        validator_id = "physician_123"
        
        validation_data = {
            "approved": True,
            "clinical_notes": "ECG normal, ritmo sinusal regular",
            "findings": ["Normal sinus rhythm", "No ST-T changes"],
            "recommendations": ["Routine follow-up"],
            "urgency_confirmed": True,
            "requires_immediate_action": False
        }
        
        # Mock do validador
        mock_validator = Mock(
            id=validator_id,
            role=UserRoles.PHYSICIAN,
            specialization="Cardiology"
        )
        
        # Mock da validação existente
        mock_validation = Mock(
            id=validation_id,
            status=ValidationStatus.PENDING,
            validator_id=None,
            clinical_urgency=ClinicalUrgency.ROUTINE
        )
        
        validation_service.repository.get_validation_by_id = AsyncMock(
            return_value=mock_validation
        )
        validation_service.repository.update_validation = AsyncMock(
            return_value=mock_validation
        )
        validation_service.user_repository.get_user_by_id = AsyncMock(
            return_value=mock_validator
        )
        
        result = await validation_service.submit_validation(
            validation_id=validation_id,
            validator_id=validator_id,
            validation_data=validation_data
        )
        
        assert result is not None
        validation_service.repository.update_validation.assert_called_once()
        
    @pytest.mark.asyncio
    async def test_validation_permission_checks(self, validation_service):
        """Testa verificações de permissão para validação."""
        # Usuário sem permissão (técnico tentando validar)
        mock_user = Mock(
            id="tech_123",
            role=UserRoles.TECHNICIAN
        )
        
        validation_service.user_repository.get_user_by_id = AsyncMock(
            return_value=mock_user
        )
        
        with pytest.raises(PermissionDeniedException):
            await validation_service.submit_validation(
                validation_id=1,
                validator_id="tech_123",
                validation_data={"approved": True}
            )
            
    @pytest.mark.asyncio
    async def test_validation_workflow_states(self, validation_service):
        """Testa estados do workflow de validação."""
        # Testar transições de estado válidas
        transitions = [
            (ValidationStatus.PENDING, ValidationStatus.APPROVED),
            (ValidationStatus.PENDING, ValidationStatus.REJECTED),
            (ValidationStatus.PENDING, ValidationStatus.REQUIRES_REVIEW),
            (ValidationStatus.REQUIRES_REVIEW, ValidationStatus.APPROVED),
            (ValidationStatus.REQUIRES_REVIEW, ValidationStatus.REJECTED),
        ]
        
        for from_status, to_status in transitions:
            is_valid = validation_service.is_valid_transition(
                from_status, to_status
            )
            assert is_valid is True
            
        # Testar transições inválidas
        invalid_transitions = [
            (ValidationStatus.APPROVED, ValidationStatus.PENDING),
            (ValidationStatus.REJECTED, ValidationStatus.APPROVED),
        ]
        
        for from_status, to_status in invalid_transitions:
            is_valid = validation_service.is_valid_transition(
                from_status, to_status
            )
            assert is_valid is False
            
    @pytest.mark.asyncio
    async def test_clinical_metrics_calculation(self, validation_service):
        """Testa cálculo de métricas clínicas."""
        validations = [
            Mock(
                status=ValidationStatus.APPROVED,
                created_at=datetime.utcnow(),
                validated_at=datetime.utcnow(),
                clinical_urgency=ClinicalUrgency.URGENT
            )
            for _ in range(10)
        ]
        
        metrics = validation_service.calculate_clinical_metrics(validations)
        
        assert "total_validations" in metrics
        assert "approval_rate" in metrics
        assert "average_turnaround_time" in metrics
        assert "urgency_distribution" in metrics
        
        assert metrics["total_validations"] == 10
        assert metrics["approval_rate"] == 1.0  # 100% aprovado
        
    @pytest.mark.asyncio
    async def test_validation_audit_trail(self, validation_service):
        """Testa trilha de auditoria de validação."""
        validation_id = 1
        changes = {
            "status": ValidationStatus.APPROVED,
            "clinical_notes": "Updated notes"
        }
        
        validation_service.audit_logger = AsyncMock()
        
        await validation_service.log_validation_change(
            validation_id=validation_id,
            user_id="physician_123",
            changes=changes
        )
        
        validation_service.audit_logger.log_event.assert_called_once()
        call_args = validation_service.audit_logger.log_event.call_args
        assert call_args[1]["event_type"] == "validation_update"
        assert call_args[1]["validation_id"] == validation_id
        
    @pytest.mark.asyncio
    async def test_critical_finding_escalation(self, validation_service):
        """Testa escalação de achados críticos."""
        critical_findings = [
            "ST-segment elevation",
            "Ventricular tachycardia",
            "Complete heart block"
        ]
        
        validation_service.notification_service.send_critical_alert = AsyncMock()
        
        await validation_service.escalate_critical_findings(
            validation_id=1,
            findings=critical_findings,
            patient_id="patient_123"
        )
        
        # Verificar que alertas foram enviados
        assert validation_service.notification_service.send_critical_alert.called
        call_count = validation_service.notification_service.send_critical_alert.call_count
        assert call_count >= 1  # Pelo menos um alerta enviado
        
    def test_validation_data_integrity(self):
        """Testa integridade dos dados de validação."""
        # Dados válidos
        valid_data = {
            "approved": True,
            "clinical_notes": "Normal ECG",
            "signal_quality_rating": 5,
            "ai_confidence_rating": 4
        }
        
        service = ValidationService(Mock(), Mock())
        is_valid = service.validate_submission_data(valid_data)
        assert is_valid is True
        
        # Dados inválidos (rating fora do range)
        invalid_data = {
            "approved": True,
            "signal_quality_rating": 6  # Max é 5
        }
        
        with pytest.raises(ValidationException):
            service.validate_submission_data(invalid_data)
'''
        
        with open(validation_test, 'w', encoding='utf-8') as f:
            f.write(content)
            
        print("✅ Testes de validação médica criados!")
        
    def create_core_services_tests(self):
        """Cria testes para serviços principais."""
        print("🔧 Criando testes de serviços principais...")
        
        services_test = self.tests_dir / "test_core_services_complete.py"
        
        content = '''"""Testes completos para serviços principais - alta cobertura."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
import numpy as np
from datetime import datetime
from app.services.ecg_service import ECGService
from app.services.patient_service import PatientService
from app.services.ml_service import MLService
from app.services.notification_service import NotificationService
from app.core.constants import AnalysisStatus, NotificationType


class TestCoreServicesComplete:
    """Testes completos para serviços principais."""
    
    @pytest.mark.asyncio
    async def test_ecg_service_complete_flow(self):
        """Testa fluxo completo do serviço ECG."""
        # Mocks
        mock_db = AsyncMock()
        mock_ml_service = AsyncMock()
        mock_notification = AsyncMock()
        
        ecg_service = ECGService(
            db=mock_db,
            ml_service=mock_ml_service,
            notification_service=mock_notification
        )
        
        # Dados de teste
        ecg_data = {
            "patient_id": "patient_123",
            "signal_data": np.random.randn(12, 5000).tolist(),
            "sampling_rate": 500,
            "acquisition_date": datetime.utcnow()
        }
        
        # Mock do repositório
        ecg_service.repository.create_ecg = AsyncMock(
            return_value=Mock(id="ecg_123", status=AnalysisStatus.PENDING)
        )
        
        # Mock do processamento ML
        mock_ml_service.analyze_ecg.return_value = {
            "predictions": {"normal": 0.95, "afib": 0.05},
            "features": {"heart_rate": 72},
            "interpretation": "Normal sinus rhythm"
        }
        
        # Executar análise
        result = await ecg_service.create_and_analyze_ecg(ecg_data)
        
        assert result is not None
        assert result["id"] == "ecg_123"
        assert "analysis_results" in result
        
        # Verificar que notificação foi enviada
        mock_notification.send_notification.assert_called()
        
    @pytest.mark.asyncio
    async def test_patient_service_operations(self):
        """Testa operações do serviço de pacientes."""
        mock_db = AsyncMock()
        patient_service = PatientService(db=mock_db)
        
        # Criar paciente
        patient_data = {
            "name": "João Silva",
            "date_of_birth": "1978-05-15",
            "gender": "M",
            "medical_record_number": "MRN12345",
            "phone": "+5511999999999",
            "email": "joao.silva@example.com"
        }
        
        mock_patient = Mock(
            id="patient_123",
            **patient_data,
            created_at=datetime.utcnow()
        )
        
        patient_service.repository.create_patient = AsyncMock(
            return_value=mock_patient
        )
        
        result = await patient_service.create_patient(patient_data)
        
        assert result.id == "patient_123"
        assert result.name == patient_data["name"]
        
        # Buscar histórico do paciente
        patient_service.repository.get_patient_history = AsyncMock(
            return_value={
                "ecg_count": 5,
                "last_ecg_date": datetime.utcnow(),
                "conditions": ["Hypertension"],
                "medications": ["Atenolol 50mg"]
            }
        )
        
        history = await patient_service.get_patient_history("patient_123")
        assert history["ecg_count"] == 5
        assert "conditions" in history
        assert "medications" in history
        
    @pytest.mark.asyncio
    async def test_ml_service_batch_processing(self):
        """Testa processamento em lote do serviço ML."""
        ml_service = MLService()
        
        # Mock do modelo
        with patch.object(ml_service, 'model') as mock_model:
            mock_model.predict_batch.return_value = np.array([
                [0.9, 0.1],  # Normal
                [0.3, 0.7],  # Arritmia
                [0.8, 0.2],  # Normal
            ])
            
            ecg_batch = [
                np.random.randn(12, 5000),
                np.random.randn(12, 5000),
                np.random.randn(12, 5000),
            ]
            
            results = await ml_service.analyze_batch(ecg_batch)
            
            assert len(results) == 3
            assert results[0]["primary_diagnosis"] == "normal"
            assert results[1]["primary_diagnosis"] == "arrhythmia"
            assert all("confidence" in r for r in results)
            
    @pytest.mark.asyncio
    async def test_notification_service_priorities(self):
        """Testa prioridades do serviço de notificação."""
        notification_service = NotificationService()
        
        # Mock do cliente de email/SMS
        notification_service.email_client = AsyncMock()
        notification_service.sms_client = AsyncMock()
        notification_service.push_client = AsyncMock()
        
        # Notificação crítica
        await notification_service.send_critical_notification(
            user_id="physician_123",
            patient_id="patient_456",
            message="Critical arrhythmia detected",
            findings=["Ventricular tachycardia"]
        )
        
        # Verificar que todos os canais foram usados para crítico
        notification_service.email_client.send.assert_called()
        notification_service.sms_client.send.assert_called()
        notification_service.push_client.send.assert_called()
        
        # Notificação rotineira
        notification_service.email_client.reset_mock()
        notification_service.sms_client.reset_mock()
        notification_service.push_client.reset_mock()
        
        await notification_service.send_routine_notification(
            user_id="physician_123",
            message="ECG analysis completed"
        )
        
        # Apenas email para rotina
        notification_service.email_client.send.assert_called()
        notification_service.sms_client.send.assert_not_called()
        
    def test_service_error_handling(self):
        """Testa tratamento de erros nos serviços."""
        # ECG Service com erro de processamento
        ecg_service = ECGService(Mock(), Mock(), Mock())
        
        with pytest.raises(Exception):
            # Sinal inválido deve gerar erro
            ecg_service.validate_ecg_signal(None)
            
        # Patient Service com dados inválidos
        patient_service = PatientService(Mock())
        
        with pytest.raises(ValueError):
            # Data de nascimento futura
            patient_service.validate_patient_data({
                "date_of_birth": "2030-01-01"
            })
            
    @pytest.mark.asyncio
    async def test_service_transaction_handling(self):
        """Testa transações nos serviços."""
        mock_db = AsyncMock()
        
        # Simular transação bem-sucedida
        mock_db.begin = AsyncMock()
        mock_db.commit = AsyncMock()
        mock_db.rollback = AsyncMock()
        
        ecg_service = ECGService(db=mock_db, ml_service=Mock(), notification_service=Mock())
        
        # Operação com sucesso
        async with ecg_service.transaction():
            # Operações do banco
            pass
            
        mock_db.commit.assert_called_once()
        mock_db.rollback.assert_not_called()
        
        # Operação com falha
        mock_db.reset_mock()
        
        with pytest.raises(Exception):
            async with ecg_service.transaction():
                raise Exception("Erro simulado")
                
        mock_db.rollback.assert_called_once()
        mock_db.commit.assert_not_called()
'''
        
        with open(services_test, 'w', encoding='utf-8') as f:
            f.write(content)
            
        print("✅ Testes de serviços principais criados!")
        
    def create_utils_tests(self):
        """Cria testes para utilitários."""
        print("🛠️ Criando testes de utilitários...")
        
        utils_test = self.tests_dir / "test_utils_complete.py"
        
        content = '''"""Testes completos para utilitários - alta cobertura."""

import pytest
import numpy as np
from datetime import datetime, timedelta
from app.utils.validators import (
    validate_ecg_signal,
    validate_patient_data,
    validate_email,
    validate_phone_number,
    validate_cpf,
    validate_date_format
)
from app.utils.formatters import (
    format_ecg_report,
    format_patient_name,
    format_date_brazilian,
    format_phone_brazilian
)
from app.utils.medical_calculations import (
    calculate_heart_rate,
    calculate_qt_corrected,
    calculate_bmi,
    calculate_age,
    calculate_cardiovascular_risk
)


class TestValidatorsComplete:
    """Testes completos para validadores."""
    
    def test_ecg_signal_validation(self):
        """Testa validação de sinal ECG."""
        # Sinal válido
        valid_signal = np.random.randn(12, 5000)
        assert validate_ecg_signal(valid_signal, sampling_rate=500) is True
        
        # Sinal inválido - dimensões erradas
        invalid_signal = np.random.randn(13, 5000)
        assert validate_ecg_signal(invalid_signal, sampling_rate=500) is False
        
        # Sinal muito curto
        short_signal = np.random.randn(12, 100)
        assert validate_ecg_signal(short_signal, sampling_rate=500) is False
        
        # Taxa de amostragem inválida
        assert validate_ecg_signal(valid_signal, sampling_rate=50) is False
        
    def test_patient_data_validation(self):
        """Testa validação de dados do paciente."""
        # Dados válidos
        valid_data = {
            "name": "Maria da Silva",
            "date_of_birth": "1980-05-15",
            "gender": "F",
            "cpf": "123.456.789-00"
        }
        assert validate_patient_data(valid_data) is True
        
        # Nome inválido
        invalid_name = valid_data.copy()
        invalid_name["name"] = "A"  # Muito curto
        assert validate_patient_data(invalid_name) is False
        
        # Data de nascimento futura
        future_birth = valid_data.copy()
        future_birth["date_of_birth"] = "2030-01-01"
        assert validate_patient_data(future_birth) is False
        
        # Gênero inválido
        invalid_gender = valid_data.copy()
        invalid_gender["gender"] = "X"
        assert validate_patient_data(invalid_gender) is False
        
    def test_email_validation(self):
        """Testa validação de email."""
        # Emails válidos
        valid_emails = [
            "user@example.com",
            "user.name@example.com",
            "user+tag@example.co.uk",
            "user123@example-domain.com"
        ]
        
        for email in valid_emails:
            assert validate_email(email) is True
            
        # Emails inválidos
        invalid_emails = [
            "invalid.email",
            "@example.com",
            "user@",
            "user @example.com",
            "user@example",
            ""
        ]
        
        for email in invalid_emails:
            assert validate_email(email) is False
            
    def test_phone_validation(self):
        """Testa validação de telefone brasileiro."""
        # Telefones válidos
        valid_phones = [
            "+5511999999999",
            "11999999999",
            "(11) 99999-9999",
            "11 9 9999-9999",
            "+55 11 99999-9999"
        ]
        
        for phone in valid_phones:
            assert validate_phone_number(phone) is True
            
        # Telefones inválidos
        invalid_phones = [
            "999999999",  # Falta DDD
            "119999999999",  # Muitos dígitos
            "11 9999-9999",  # Falta o 9
            "(11) 9999-999",  # Formato errado
            "abcdefghijk"  # Não é número
        ]
        
        for phone in invalid_phones:
            assert validate_phone_number(phone) is False
            
    def test_cpf_validation(self):
        """Testa validação de CPF."""
        # CPFs válidos (formato exemplo)
        assert validate_cpf("123.456.789-00") is True
        assert validate_cpf("12345678900") is True
        
        # CPFs inválidos
        assert validate_cpf("111.111.111-11") is False  # Todos iguais
        assert validate_cpf("123.456.789-99") is False  # Dígito errado
        assert validate_cpf("12345678") is False  # Incompleto
        

class TestFormattersComplete:
    """Testes completos para formatadores."""
    
    def test_ecg_report_formatting(self):
        """Testa formatação de relatório ECG."""
        report_data = {
            "patient_name": "João Silva",
            "exam_date": datetime.now(),
            "heart_rate": 72,
            "rhythm": "Ritmo sinusal",
            "findings": ["Normal", "Sem alterações ST-T"],
            "interpretation": "ECG dentro dos limites normais"
        }
        
        formatted = format_ecg_report(report_data)
        
        assert "João Silva" in formatted
        assert "72 bpm" in formatted
        assert "Ritmo sinusal" in formatted
        assert "Normal" in formatted
        
    def test_name_formatting(self):
        """Testa formatação de nomes."""
        # Nomes para testar
        test_cases = [
            ("joão da silva", "João da Silva"),
            ("MARIA DOS SANTOS", "Maria dos Santos"),
            ("pedro de souza jr.", "Pedro de Souza Jr."),
            ("ana  clara   silva", "Ana Clara Silva"),  # Espaços extras
        ]
        
        for input_name, expected in test_cases:
            assert format_patient_name(input_name) == expected
            
    def test_date_formatting(self):
        """Testa formatação de datas brasileiras."""
        test_date = datetime(2025, 6, 17, 14, 30, 0)
        
        # Formato completo
        full_format = format_date_brazilian(test_date, include_time=True)
        assert full_format == "17/06/2025 14:30"
        
        # Apenas data
        date_only = format_date_brazilian(test_date, include_time=False)
        assert date_only == "17/06/2025"
        
    def test_phone_formatting(self):
        """Testa formatação de telefone brasileiro."""
        test_cases = [
            ("11999999999", "(11) 99999-9999"),
            ("+5511999999999", "+55 (11) 99999-9999"),
            ("1199999999", "(11) 9999-9999"),  # Fixo
        ]
        
        for input_phone, expected in test_cases:
            assert format_phone_brazilian(input_phone) == expected
            

class TestMedicalCalculationsComplete:
    """Testes completos para cálculos médicos."""
    
    def test_heart_rate_calculation(self):
        """Testa cálculo de frequência cardíaca."""
        # RR intervals em ms
        rr_intervals = [833, 850, 825, 840, 830]  # ~72 bpm
        
        hr = calculate_heart_rate(rr_intervals)
        assert 70 <= hr <= 74
        
        # Teste com taquicardia
        fast_rr = [500, 510, 495, 505]  # ~120 bpm
        fast_hr = calculate_heart_rate(fast_rr)
        assert 115 <= fast_hr <= 125
        
    def test_qt_corrected_calculation(self):
        """Testa cálculo de QT corrigido."""
        # QT normal com FC normal
        qt_ms = 400
        heart_rate = 70
        
        qtc = calculate_qt_corrected(qt_ms, heart_rate)
        assert 380 <= qtc <= 440  # Faixa normal
        
        # QT prolongado
        long_qt = 500
        long_qtc = calculate_qt_corrected(long_qt, heart_rate)
        assert long_qtc > 440
        
    def test_bmi_calculation(self):
        """Testa cálculo de IMC."""
        # Peso normal
        bmi = calculate_bmi(weight_kg=70, height_cm=175)
        assert 22 <= bmi <= 23
        
        # Obeso
        obese_bmi = calculate_bmi(weight_kg=100, height_cm=170)
        assert obese_bmi > 30
        
        # Teste com valores extremos
        assert calculate_bmi(weight_kg=0, height_cm=170) == 0
        assert calculate_bmi(weight_kg=70, height_cm=0) is None
        
    def test_age_calculation(self):
        """Testa cálculo de idade."""
        # Data de nascimento conhecida
        birth_date = datetime(1980, 6, 15)
        reference_date = datetime(2025, 6, 17)
        
        age = calculate_age(birth_date, reference_date)
        assert age == 45
        
        # Aniversário ainda não chegou
        birth_date2 = datetime(1980, 12, 31)
        age2 = calculate_age(birth_date2, reference_date)
        assert age2 == 44
        
    def test_cardiovascular_risk_calculation(self):
        """Testa cálculo de risco cardiovascular."""
        # Risco baixo
        low_risk = calculate_cardiovascular_risk(
            age=30,
            gender="M",
            systolic_bp=120,
            total_cholesterol=180,
            hdl_cholesterol=50,
            smoker=False,
            diabetic=False
        )
        assert low_risk < 10
        
        # Risco alto
        high_risk = calculate_cardiovascular_risk(
            age=65,
            gender="M",
            systolic_bp=160,
            total_cholesterol=280,
            hdl_cholesterol=35,
            smoker=True,
            diabetic=True
        )
        assert high_risk > 20
        
        # Validação de parâmetros
        with pytest.raises(ValueError):
            calculate_cardiovascular_risk(
                age=-5,  # Idade inválida
                gender="M",
                systolic_bp=120,
                total_cholesterol=180,
                hdl_cholesterol=50,
                smoker=False,
                diabetic=False
            )
'''
        
        with open(utils_test, 'w', encoding='utf-8') as f:
            f.write(content)
            
        print("✅ Testes de utilitários criados!")


if __name__ == "__main__":
    creator = CriticalTestsCreator()
    creator.create_all_critical_tests()
