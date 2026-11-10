@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" goto noinstall
.venv\Scripts\python.exe -c "import cv2, numpy, mss, PySide6.QtWidgets" >nul 2>nul
if errorlevel 1 goto noinstall
if not "%~1"=="" goto direct

start "" ".venv\Scripts\pythonw.exe" -m colorvision.gui
exit /b 0

:direct
.venv\Scripts\python.exe -m colorvision %*
exit /b %errorlevel%

:noinstall
title OSRS Color Vision
echo Program nie jest zainstalowany, instalacja zostala przerwana albo wymaga aktualizacji.
echo Kliknij dwukrotnie install.bat, nie zamykaj okna i poczekaj na napis GOTOWE.
echo.
pause
exit /b 1
