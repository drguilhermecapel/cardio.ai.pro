import pytest
import asyncio
import sys
from unittest.mock import MagicMock, AsyncMock

sys.modules['celery'] = MagicMock()
sys.modules['redis'] = MagicMock()
sys.modules['wfdb'] = MagicMock()
sys.modules['pyedflib'] = MagicMock()

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()

@pytest.fixture(scope="function")
def mock_db():
    """Create a mock database session."""
    return AsyncMock()

@pytest.fixture(scope="function") 
def sample_signal():
    """Create sample ECG signal for testing."""
    import numpy as np
    return np.sin(np.linspace(0, 10, 1000))
