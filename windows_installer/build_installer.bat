@echo off
echo ========================================
echo MedAI Radiologia - Preparação do Build
echo ========================================
echo.
echo Este script prepara o ambiente para construir o instalador
echo.

REM Criar estrutura de diretórios
echo Criando estrutura de diretórios...
if not exist "build" mkdir build
if not exist "build\assets" mkdir build\assets
if not exist "src" mkdir src

REM Criar arquivo main.py de exemplo se não existir
if not exist "main.py" (
    echo Criando arquivo main.py de exemplo...
    (
        echo import sys
        echo import os
        echo from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton, QFileDialog, QMessageBox
        echo from PyQt5.QtCore import Qt
        echo.
        echo class MedAIRadiologia(QMainWindow^):
        echo     def __init__(self^):
        echo         super(^).__init__(^)
        echo         self.setWindowTitle("MedAI Radiologia - Sistema de Análise Radiológica"^)
        echo         self.setGeometry(100, 100, 800, 600^)
        echo         
        echo         # Widget central
        echo         central_widget = QWidget(^)
        echo         self.setCentralWidget(central_widget^)
        echo         
        echo         # Layout
        echo         layout = QVBoxLayout(^)
        echo         central_widget.setLayout(layout^)
        echo         
        echo         # Label de boas-vindas
        echo         welcome_label = QLabel("Bem-vindo ao MedAI Radiologia"^)
        echo         welcome_label.setAlignment(Qt.AlignCenter^)
        echo         welcome_label.setStyleSheet("font-size: 24px; font-weight: bold; padding: 20px;"^)
        echo         layout.addWidget(welcome_label^)
        echo         
        echo         # Botão para carregar DICOM
        echo         load_button = QPushButton("Carregar Imagem DICOM"^)
        echo         load_button.setStyleSheet("font-size: 16px; padding: 10px;"^)
        echo         load_button.clicked.connect(self.load_dicom^)
        echo         layout.addWidget(load_button^)
        echo         
        echo         # Label de status
        echo         self.status_label = QLabel("Pronto para carregar imagens..."^)
        echo         self.status_label.setAlignment(Qt.AlignCenter^)
        echo         layout.addWidget(self.status_label^)
        echo         
        echo         layout.addStretch(^)
        echo     
        echo     def load_dicom(self^):
        echo         file_path, _ = QFileDialog.getOpenFileName(
        echo             self, 
        echo             "Selecionar arquivo DICOM", 
        echo             "", 
        echo             "Arquivos DICOM (*.dcm);;Todos os arquivos (*.*)"
        echo         ^)
        echo         
        echo         if file_path:
        echo             self.status_label.setText(f"Arquivo carregado: {os.path.basename(file_path)}"^)
        echo             QMessageBox.information(self, "Sucesso", "Arquivo DICOM carregado com sucesso!"^)
        echo.
        echo def main(^):
        echo     app = QApplication(sys.argv^)
        echo     window = MedAIRadiologia(^)
        echo     window.show(^)
        echo     sys.exit(app.exec_(^)^)
        echo.
        echo if __name__ == "__main__":
        echo     main(^)
    ) > main.py
    echo Arquivo main.py criado!
) else (
    echo Arquivo main.py já existe.
)

REM Mover arquivos para diretório build
echo.
echo Copiando arquivos para diretório build...
copy "build_installer.bat" "build\build_installer.bat" >nul 2>&1
copy "medai_installer.nsi" "build\medai_installer.nsi" >nul 2>&1

REM Criar requirements.txt se não existir
if not exist "requirements.txt" (
    echo Criando requirements.txt...
    (
        echo # MedAI Radiologia - Dependências
        echo numpy==1.24.3
        echo opencv-python==4.8.0.74
        echo pydicom==2.4.3
        echo Pillow==10.0.0
        echo PyQt5==5.15.9
        echo torch>=2.0.0
        echo torchvision>=0.15.0
        echo matplotlib==3.7.2
        echo scikit-image==0.21.0
        echo scipy==1.11.1
        echo pyinstaller==5.13.0
    ) > requirements.txt
    echo Arquivo requirements.txt criado!
) else (
    echo Arquivo requirements.txt já existe.
)

echo.
echo ========================================
echo Preparação concluída!
echo ========================================
echo.
echo Próximos passos:
echo 1. Certifique-se de ter o Python instalado (3.8+ recomendado)
echo 2. Instale o NSIS de: https://nsis.sourceforge.io/Download
echo 3. Execute: pip install -r requirements.txt
echo 4. Navegue para a pasta build: cd build
echo 5. Execute: build_installer.bat
echo.
echo Estrutura criada:
echo - main.py (arquivo principal do aplicativo)
echo - requirements.txt (dependências Python)
echo - build\ (diretório com scripts de build)
echo   - build_installer.bat
echo   - medai_installer.nsi
echo.
pause
