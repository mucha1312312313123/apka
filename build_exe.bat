@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Najpierw uruchom install.bat.
  pause
  exit /b 1
)
.venv\Scripts\python.exe -m pip install --disable-pip-version-check pyinstaller
if errorlevel 1 goto fail
.venv\Scripts\python.exe build_exe.py
if errorlevel 1 goto fail
echo.
echo Gotowy plik jest w folderze dist.
pause
exit /b 0

:fail
echo.
echo BLAD: budowanie nie powiodlo sie. Przeczytaj komunikat powyzej.
pause
exit /b 1
