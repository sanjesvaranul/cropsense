@echo off
echo ========================================================
echo  🌾 CropSense: AgriLens — Launching Public Demo Link
echo ========================================================
echo.

:: 1. Check if FastAPI is running on Port 8000
netstat -ano | findstr :8000 >nul
if %errorlevel% neq 0 (
    echo [*] Starting FastAPI Backend on Port 8000...
    start /b .\cropsense-env\Scripts\python.exe -m uvicorn inference.router:app --host 127.0.0.1 --port 8000
    timeout /t 3 /nobreak >nul
) else (
    echo [v] FastAPI Backend is already running on Port 8000.
)

:: 2. Check if Streamlit is running on Port 8501
netstat -ano | findstr :8501 >nul
if %errorlevel% neq 0 (
    echo [*] Starting Streamlit App on Port 8501...
    start /b .\cropsense-env\Scripts\python.exe -m streamlit run demo/app.py --server.port 8501 --server.headless true
    timeout /t 4 /nobreak >nul
) else (
    echo [v] Streamlit App is already running on Port 8501.
)

echo.
echo [*] Starting Cloudflare Public Secure Tunnel...
echo [*] A public HTTPS URL will appear below for the FarmAI team:
echo.
.\cloudflared.exe tunnel --url http://localhost:8501
pause
