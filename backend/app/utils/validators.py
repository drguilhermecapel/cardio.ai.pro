import re
from typing import Optional, Union
from datetime import datetime

def validate_email(email: str) -> bool:
    """
    Valida formato de email
    
    Args:
        email (str): Email a ser validado
        
    Returns:
        bool: True se o email for válido, False caso contrário
    """
    if not email or not isinstance(email, str):
        return False
    
    # Padrão regex para validação de email
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email.strip()))

def validate_cpf(cpf: str) -> bool:
    """
    Valida CPF brasileiro
    
    Args:
        cpf (str): CPF a ser validado (com ou sem formatação)
        
    Returns:
        bool: True se o CPF for válido, False caso contrário
    """
    if not cpf or not isinstance(cpf, str):
        return False
    
    # Remove formatação
    cpf = re.sub(r'[^0-9]', '', cpf)
    
    # Verifica se tem 11 dígitos
    if len(cpf) != 11:
        return False
    
    # Verifica se todos os dígitos são iguais
    if cpf == cpf[0] * 11:
        return False
    
    # Calcula primeiro dígito verificador
    soma = sum(int(cpf[i]) * (10 - i) for i in range(9))
    resto = soma % 11
    digito1 = 0 if resto < 2 else 11 - resto
    
    # Calcula segundo dígito verificador
    soma = sum(int(cpf[i]) * (11 - i) for i in range(10))
    resto = soma % 11
    digito2 = 0 if resto < 2 else 11 - resto
    
    # Verifica se os dígitos calculados conferem
    return cpf[-2:] == f"{digito1}{digito2}"

def validate_phone(phone: str) -> bool:
    """
    Valida número de telefone brasileiro
    
    Args:
        phone (str): Telefone a ser validado
        
    Returns:
        bool: True se o telefone for válido, False caso contrário
    """
    if not phone or not isinstance(phone, str):
        return False
    
    # Remove formatação
    phone = re.sub(r'[^0-9]', '', phone)
    
    # Verifica se tem 10 ou 11 dígitos (com DDD)
    if len(phone) not in [10, 11]:
        return False
    
    # Verifica se o DDD é válido (11-99)
    ddd = int(phone[:2])
    if ddd < 11 or ddd > 99:
        return False
    
    # Para celular (11 dígitos), o primeiro dígito após o DDD deve ser 9
    if len(phone) == 11 and phone[2] != '9':
        return False
    
    return True

def validate_patient_id(patient_id: Union[int, str]) -> bool:
    """
    Valida ID de paciente
    
    Args:
        patient_id: ID do paciente a ser validado
        
    Returns:
        bool: True se o ID for válido, False caso contrário
    """
    try:
        pid = int(patient_id)
        return pid > 0
    except (ValueError, TypeError):
        return False

def validate_ecg_data(ecg_data: dict) -> tuple[bool, list]:
    """
    Valida dados de ECG
    
    Args:
        ecg_data (dict): Dados do ECG a serem validados
        
    Returns:
        tuple: (is_valid, list_of_errors)
    """
    errors = []
    
    if not isinstance(ecg_data, dict):
        errors.append("ECG data must be a dictionary")
        return False, errors
    
    # Campos obrigatórios
    required_fields = ['leads', 'duration', 'sample_rate']
    for field in required_fields:
        if field not in ecg_data:
            errors.append(f"Missing required field: {field}")
    
    # Validação específica dos campos
    if 'leads' in ecg_data:
        if not isinstance(ecg_data['leads'], list) or len(ecg_data['leads']) == 0:
            errors.append("Leads must be a non-empty list")
    
    if 'duration' in ecg_data:
        try:
            duration = float(ecg_data['duration'])
            if duration <= 0:
                errors.append("Duration must be positive")
        except (ValueError, TypeError):
            errors.append("Duration must be a number")
    
    if 'sample_rate' in ecg_data:
        try:
            sample_rate = int(ecg_data['sample_rate'])
            if sample_rate <= 0:
                errors.append("Sample rate must be positive")
        except (ValueError, TypeError):
            errors.append("Sample rate must be an integer")
    
    return len(errors) == 0, errors

def validate_date_range(start_date: str, end_date: str) -> bool:
    """
    Valida intervalo de datas
    
    Args:
        start_date (str): Data de início (formato ISO)
        end_date (str): Data de fim (formato ISO)
        
    Returns:
        bool: True se o intervalo for válido, False caso contrário
    """
    try:
        start = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
        end = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
        return start <= end
    except (ValueError, AttributeError):
        return False

def sanitize_filename(filename: str) -> str:
    """
    Sanitiza nome de arquivo removendo caracteres perigosos
    
    Args:
        filename (str): Nome do arquivo a ser sanitizado
        
    Returns:
        str: Nome do arquivo sanitizado
    """
    if not filename:
        return "unnamed_file"
    
    # Remove caracteres perigosos
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove espaços extras e pontos no início/fim
    sanitized = sanitized.strip('. ')
    
    # Limita o tamanho
    if len(sanitized) > 255:
        name, ext = sanitized.rsplit('.', 1) if '.' in sanitized else (sanitized, '')
        sanitized = name[:250] + ('.' + ext if ext else '')
    
    return sanitized or "unnamed_file"

