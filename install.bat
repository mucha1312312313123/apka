@echo off
setlocal
cd /d "%~dp0"
title OSRS Color Vision - instalacja
echo === OSRS Color Vision: instalacja ===
echo Folder: %CD%
echo.

echo [1/4] Szukam Pythona...
set "PY="
py -3 --version >nul 2>nul
if not errorlevel 1 set "PY=py -3"
if defined PY goto found
python --version >nul 2>nul
if not errorlevel 1 set "PY=python"
if defined PY goto found
goto nopython

:found
%PY% --version
%PY% -c "import sys; sys.exit(sys.version_info < (3, 10))"
if errorlevel 1 goto oldpython
echo.

echo [2/4] Tworze srodowisko .venv. Moze potrwac do minuty, NIE zamykaj tego okna...
%PY% -m venv --clear .venv
if errorlevel 1 goto venvfail
echo.

echo [3/4] Instaluje biblioteki. Pobieranie moze potrwac kilka minut, potrzebny internet...
.venv\Scripts\python.exe -m pip install --disable-pip-version-check -r requirements.txt
if errorlevel 1 goto pipfail
echo.

echo [4/4] Sprawdzam instalacje...
.venv\Scripts\python.exe -c "import cv2, numpy, mss, PySide6.QtWidgets, colorvision.gui.main_window; print('OK, OpenCV', cv2.__version__)"
if errorlevel 1 goto pipfail
echo.
echo GOTOWE. Teraz kliknij dwukrotnie run.bat - otworzy sie okno programu.
goto end

:nopython
echo.
echo BLAD: nie znaleziono Pythona na tym komputerze.
echo Zainstaluj Python 3.10 lub nowszy ze strony python.org.
echo W instalatorze zaznacz "Add python.exe to PATH".
echo Potem uruchom install.bat ponownie.
goto fail

:oldpython
echo.
echo BLAD: ta wersja Pythona jest za stara. Potrzebny jest Python 3.10 lub nowszy.
goto fail

:venvfail
echo.
echo BLAD: nie udalo sie utworzyc srodowiska .venv. Przeczytaj komunikat powyzej.
goto fail

:pipfail
echo.
echo BLAD: instalacja bibliotek nie powiodla sie. Sprawdz internet i komunikat powyzej.
goto fail

:fail
echo.
pause
exit /b 1

:end
echo.
pause
exit /b 0
