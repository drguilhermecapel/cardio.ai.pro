# CardioAI Pro Windows Installer

This directory contains all the necessary files and scripts to build a standalone Windows installer for CardioAI Pro, an AI-powered electronic medical record system with ECG analysis capabilities.

## Overview

The Windows installer creates a self-contained desktop application that:
- Runs the FastAPI backend as a standalone executable
- Serves the React frontend through a built-in web server
- Uses SQLite for data persistence (no external database required)
- Includes all AI models and dependencies
- Provides a simple desktop shortcut for users
- Requires no Docker, command line, or technical setup from end users

## Prerequisites

Before building the installer, ensure you have the following tools installed:

### Required Tools

1. **Python 3.8+**
   - Download from: https://python.org
   - Make sure Python is added to your PATH during installation
   - Verify installation: `python --version`

2. **Node.js 16+**
   - Download from: https://nodejs.org
   - Choose the LTS version
   - Verify installation: `node --version` and `npm --version`

3. **Poetry** (Python dependency manager)
   - Will be automatically installed by the build script if not present
   - Or install manually: `pip install poetry`

### Optional Tools

4. **NSIS (Nullsoft Scriptable Install System)**
   - Download from: https://nsis.sourceforge.io/
   - Required only for final installer compilation
   - If not installed, the build script will create all files but you'll need to compile manually

## Build Process

### Automated Build (Recommended)

1. Open Command Prompt or PowerShell as Administrator
2. Navigate to the `windows_installer` directory:
   ```cmd
   cd path\to\cardio.ai.pro\windows_installer
   ```
3. Run the automated build script:
   ```cmd
   build_installer.bat
   ```

The script will:
- Check for required tools
- Build the backend executable using PyInstaller
- Build the frontend using npm/yarn
- Create all necessary configuration files
- Compile the Windows installer (if NSIS is available)

### Manual Build Steps

If you prefer to build manually or need to troubleshoot:

#### Step 1: Build Backend
```cmd
python build_backend.py
```
This creates:
- `cardioai-backend.exe` - Standalone backend executable
- `start_backend.bat` - Backend startup script

#### Step 2: Build Frontend
```cmd
python build_frontend.py
```
This creates:
- `frontend_build/` - Built React application
- `serve_frontend.py` - Frontend web server

#### Step 3: Compile Installer (requires NSIS)
```cmd
makensis cardioai_installer.nsi
```
This creates:
- `CardioAI-Pro-Installer.exe` - Final Windows installer

## Output Files

After a successful build, you'll have:

- **CardioAI-Pro-Installer.exe** - The main installer file for distribution
- **cardioai-backend.exe** - Standalone backend executable
- **frontend_build/** - Built React frontend files
- **serve_frontend.py** - Python script to serve the frontend
- **start_backend.bat** - Script to start the backend
- **LICENSE.txt** - License file for the installer
- **.env.example** - Environment configuration template

## Installation for End Users

End users simply need to:

1. Download `CardioAI-Pro-Installer.exe`
2. Run the installer as Administrator
3. Follow the installation wizard
4. Launch CardioAI Pro from the desktop shortcut

The installed application will:
- Start both backend and frontend automatically
- Open the application in the default web browser
- Store data in a local SQLite database
- Run entirely offline (no internet connection required for core functionality)

## Troubleshooting

### Common Build Issues

**"Python is not installed or not in PATH"**
- Install Python from python.org
- During installation, check "Add Python to PATH"
- Restart Command Prompt after installation

**"Node.js is not installed or not in PATH"**
- Install Node.js from nodejs.org
- Restart Command Prompt after installation

**"Poetry not found"**
- The build script will attempt to install Poetry automatically
- If it fails, install manually: `pip install poetry`

**"NSIS compilation failed"**
- Ensure NSIS is properly installed
- Check that `makensis` is in your PATH
- Try running `makensis /VERSION` to verify installation

### Backend Build Issues

**"PyInstaller failed"**
- Ensure all Python dependencies are installed
- Try running `poetry install` in the backend directory first
- Check for missing system libraries

**"Models not found"**
- The build script creates placeholder model files
- Replace with actual ONNX model files if available
- Ensure model files are in the `backend/models/` directory

### Frontend Build Issues

**"npm install failed"**
- Try deleting `node_modules` and `package-lock.json`
- Run `npm install` again
- Check for Node.js version compatibility

**"Build failed"**
- Check for TypeScript errors in the frontend code
- Ensure all dependencies are compatible
- Try `npm run build` manually in the frontend directory

## Configuration

### Environment Variables

The installer uses these default settings:
- `STANDALONE_MODE=true` - Enables standalone operation
- `DATABASE_URL=sqlite+aiosqlite:///./cardioai.db` - SQLite database
- `REDIS_URL=` - Disabled for standalone mode
- `CELERY_BROKER_URL=` - Disabled for standalone mode

### Customization

To customize the installer:

1. **Change application details**: Edit variables at the top of `cardioai_installer.nsi`
2. **Add custom icon**: Replace `cardioai.ico` with your icon file
3. **Modify license**: Edit `LICENSE.txt`
4. **Change default settings**: Modify `.env.example`

## Security Considerations

- The installer requires Administrator privileges to install system-wide
- The application stores medical data locally in SQLite
- No network services are exposed by default
- All data remains on the local machine

## Support

For build issues or questions:
1. Check the troubleshooting section above
2. Verify all prerequisites are installed
3. Try the manual build steps to isolate issues
4. Check the GitHub repository for updates and issues

## License

This installer and the CardioAI Pro application are distributed under the MIT License. See LICENSE.txt for details.
