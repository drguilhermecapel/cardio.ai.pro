"""
Utilitários auxiliares para testes
"""
import uuid
import random
from datetime import datetime, timedelta
from typing import Dict, Any, Optional


class ECGTestGenerator:
    """Gerador de dados de teste para ECG."""
    
    @staticmethod
    def generate_ecg_data(
        patient_id: Optional[int] = None,
        custom_fields: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Gera dados de ECG para teste."""
        base_data = {
            "patient_id": patient_id or random.randint(1, 1000),
            "file_url": f"https://storage.example.com/ecg/{uuid.uuid4()}.pdf",
            "file_type": random.choice(["image", "pdf", "dicom"]),
            "analysis_type": "standard",
            "priority": random.choice(["low", "normal", "high", "critical"]),
            "metadata": {
                "device": "ECG-Device-X1",
                "leads": 12,
                "duration": random.randint(10, 30),
                "sample_rate": 500
            }
        }
        
        if custom_fields:
            base_data.update(custom_fields)
            
        return base_data
        
    @staticmethod
    def generate_findings() -> Dict[str, Any]:
        """Gera findings aleatórios para teste."""
        return {
            "heart_rate": random.randint(50, 120),
            "pr_interval": random.randint(120, 200),
            "qrs_duration": random.randint(80, 120),
            "qt_interval": random.randint(350, 450),
            "abnormalities": random.choice([[], ["ST elevation"], ["T wave inversion"]]),
            "interpretation": "Teste de interpretação automática"
        }
        

def create_test_user(role: str = "user") -> Dict[str, Any]:
    """Cria dados de usuário para teste."""
    user_id = str(uuid.uuid4())
    return {
        "id": user_id,
        "email": f"test_{user_id[:8]}@example.com",
        "name": f"Test User {user_id[:8]}",
        "role": role,
        "is_active": True
    }
    

def create_auth_headers(token: str = None) -> Dict[str, str]:
    """Cria headers de autenticação para teste."""
    if not token:
        token = f"test_token_{uuid.uuid4()}"
    return {"Authorization": f"Bearer {token}"}
    

def generate_patient_data(
    name: Optional[str] = None,
    cpf: Optional[str] = None
) -> Dict[str, Any]:
    """Gera dados de paciente para teste."""
    if not name:
        name = f"Paciente Teste {random.randint(1000, 9999)}"
    
    if not cpf:
        # Gera CPF válido para teste
        cpf = f"{random.randint(100, 999)}.{random.randint(100, 999)}.{random.randint(100, 999)}-{random.randint(10, 99)}"
    
    return {
        "name": name,
        "cpf": cpf,
        "birth_date": (datetime.now() - timedelta(days=random.randint(7300, 29200))).date().isoformat(),
        "gender": random.choice(["M", "F"]),
        "email": f"{name.lower().replace(' ', '.')}@example.com",
        "phone": f"(11) 9{random.randint(1000, 9999)}-{random.randint(1000, 9999)}"
    }
