@echo off
REM Navigate to the Prometheus directory
cd /d C:\prometheus-3.6.0-rc.0.windows-amd64

REM Start Prometheus with the default configuration
start prometheus.exe --config.file=prometheus.yml

REM Optional: Open Prometheus in the default browser
timeout /t 3 >nul
start http://localhost:9090

echo Prometheus is starting...
pause
