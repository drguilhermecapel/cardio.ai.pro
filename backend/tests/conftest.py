import sys
import os
from unittest.mock import MagicMock

# Configurar ambiente de teste
os.environ["ENVIRONMENT"] = "test"
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"
os.environ["SECRET_KEY"] = "test-secret-key"

# Mock do pyedflib
sys.modules["pyedflib"] = MagicMock()
