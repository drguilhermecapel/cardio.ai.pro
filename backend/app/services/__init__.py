"""Services module."""
# Importações condicionais para evitar erros de sintaxe
try:
    from .ecg_service import ECGAnalysisService
except SyntaxError:
    # Fallback se houver erro de sintaxe
    class ECGAnalysisService:
        def __init__(self, *args, **kwargs):
            pass
