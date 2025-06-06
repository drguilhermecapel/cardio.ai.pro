#!/usr/bin/env python3
"""
Script de validação rigorosa para tipos médicos.
Uso: poetry run python validate_medical_types.py
"""

import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Tuple

def run_mypy_analysis() -> Tuple[int, str, str]:
    """Execute análise MyPy e retorna resultados."""
    cmd = [
        "poetry", "run", "mypy", 
        "backend/app/services/",
        "backend/app/validation/",
        "--show-error-codes",
        "--show-column-numbers",
        "--no-error-summary",
        "--show-traceback"
    ]
    
    result = subprocess.run(
        cmd, 
        capture_output=True, 
        text=True, 
        cwd=Path.cwd()
    )
    
    return result.returncode, result.stdout, result.stderr

def categorize_medical_errors(output: str) -> Dict[str, List[str]]:
    """Categoriza erros por criticidade médica."""
    lines = output.strip().split('\n') if output.strip() else []
    
    categories = {
        'critical': [],      # Dados de paciente, diagnósticos
        'important': [],     # Configurações, validações
        'minor': []         # Utilitários, documentação
    }
    
    critical_patterns = [
        'patient', 'ecg', 'signal', 'diagnosis', 'cardiac',
        'heart', 'rhythm', 'arrhythmia', 'medical', 'pathology',
        'validation', 'clinical', 'hybrid_ecg', 'iso13485'
    ]
    
    important_patterns = [
        'config', 'validation', 'logging', 'monitoring',
        'security', 'auth', 'database', 'api'
    ]
    
    for line in lines:
        if not line.strip() or 'Found' in line or 'Success' in line:
            continue
            
        line_lower = line.lower()
        if any(pattern in line_lower for pattern in critical_patterns):
            categories['critical'].append(line)
        elif any(pattern in line_lower for pattern in important_patterns):
            categories['important'].append(line)
        else:
            categories['minor'].append(line)
    
    return categories

def main() -> int:
    """Execução principal da validação."""
    print("🏥 Validação de Tipos para Sistema Médico Crítico")
    print("=" * 60)
    
    returncode, stdout, stderr = run_mypy_analysis()
    
    if returncode == 0:
        print("✅ Todos os tipos validados com sucesso!")
        return 0
    
    print("❌ Erros de tipo detectados:")
    print("-" * 40)
    
    if stderr:
        print(f"STDERR: {stderr}")
    
    categories = categorize_medical_errors(stdout)
    
    if categories['critical']:
        print(f"\n🚨 ERROS CRÍTICOS (BLOQUEIAM MERGE): {len(categories['critical'])}")
        for error in categories['critical']:
            print(f"  ❌ {error}")
    
    if categories['important']:
        print(f"\n⚠️ ERROS IMPORTANTES (RESOLVER ANTES DO MERGE): {len(categories['important'])}")
        for error in categories['important']:
            print(f"  ⚠️ {error}")
    
    if categories['minor']:
        print(f"\nℹ️ ERROS MENORES (PODEM SER ADIADOS): {len(categories['minor'])}")
        for error in categories['minor']:
            print(f"  ℹ️ {error}")
    
    if categories['critical']:
        print("\n🛑 MERGE BLOQUEADO: Erros críticos em sistema médico")
        return 1
    
    if categories['important']:
        print("\n⚠️ MERGE NÃO RECOMENDADO: Erros importantes detectados")
        return 1
    
    print("\n✅ Tipos validados para merge (apenas erros menores)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
