@echo off
echo Testing Python command syntax...
echo.

REM Test 1: Import statement
echo Test 1: Testing import statement...
python -c "from datetime import datetime; print('Test 1: OK')"
if %errorlevel% neq 0 (
    echo Test 1 FAILED
    goto :error
)

REM Test 2: Multiple imports
echo Test 2: Testing multiple imports...
python -c "import sys; import os; print('Test 2: OK')"
if %errorlevel% neq 0 (
    echo Test 2 FAILED
    goto :error
)

REM Test 3: In FOR loop
echo Test 3: Testing FOR loop with Python command...
for /f "tokens=*" %%i in ('python -c "print('Test 3: OK')"') do echo %%i
if %errorlevel% neq 0 (
    echo Test 3 FAILED
    goto :error
)

REM Test 4: Version check (similar to build_installer.bat)
echo Test 4: Testing Python version check...
python -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)" >nul 2>&1
if %errorlevel% neq 0 (
    echo Test 4 FAILED - Python version check
    goto :error
)
echo Test 4: OK

REM Test 5: Complex command with semicolons
echo Test 5: Testing complex command with semicolons...
python -c "import platform; print('Python version:', platform.python_version()); print('Test 5: OK')"
if %errorlevel% neq 0 (
    echo Test 5 FAILED
    goto :error
)

echo.
echo ========================================
echo All syntax tests PASSED successfully!
echo ========================================
echo.
echo This confirms that Python command syntax is correct
echo and should not cause "from: foi inesperado" errors.
echo.
pause
exit /b 0

:error
echo.
echo ========================================
echo SYNTAX TEST FAILED!
echo ========================================
echo.
echo One or more Python commands failed to execute properly.
echo This indicates a syntax issue that needs to be resolved.
echo.
pause
exit /b 1
