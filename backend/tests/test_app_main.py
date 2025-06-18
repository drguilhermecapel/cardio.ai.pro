"""Testes para o módulo principal."""
import pytest
from app.main import get_app_info, health_check, CardioAIApp

class TestAppMain:
    """Testes do módulo principal."""
    
    def test_get_app_info(self):
        """Testa informações da app."""
        info = get_app_info()
        assert info["name"] == "CardioAI Pro"
        assert info["version"] == "1.0.0"
        assert info["status"] == "running"
    
    def test_health_check(self):
        """Testa health check."""
        health = health_check()
        assert health["status"] == "healthy"
        assert health["service"] == "cardioai-pro"
    
    def test_cardioai_app_init(self):
        """Testa inicialização da app."""
        app = CardioAIApp()
        assert app.name == "CardioAI Pro"
        assert app.version == "1.0.0"
        assert app.modules == []
    
    def test_cardioai_app_modules(self):
        """Testa gerenciamento de módulos."""
        app = CardioAIApp()
        
        # Adicionar módulos
        assert app.add_module("ecg") is True
        assert app.add_module("analysis") is True
        
        # Verificar módulos
        modules = app.get_modules()
        assert len(modules) == 2
        assert "ecg" in modules
        assert "analysis" in modules
    
    def test_process_ecg_valid(self):
        """Testa processamento válido de ECG."""
        app = CardioAIApp()
        result = app.process_ecg({"signal": [1, 2, 3]})
        assert result["status"] == "processed"
        assert result["data"] == {"signal": [1, 2, 3]}
    
    def test_process_ecg_invalid(self):
        """Testa processamento inválido de ECG."""
        app = CardioAIApp()
        with pytest.raises(ValueError, match="Dados inválidos"):
            app.process_ecg(None)
    
    def test_process_ecg_empty(self):
        """Testa processamento com dados vazios."""
        app = CardioAIApp()
        with pytest.raises(ValueError):
            app.process_ecg({})
