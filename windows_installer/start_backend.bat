@echo off
echo Starting CardioAI Pro Backend...

REM Set environment variables for standalone mode
set STANDALONE_MODE=true
set DATABASE_URL=sqlite+aiosqlite:///./cardioai.db
set REDIS_URL=
set CELERY_BROKER_URL=
set CELERY_RESULT_BACKEND=

REM Start the backend server
cardioai-backend.exe

pause
