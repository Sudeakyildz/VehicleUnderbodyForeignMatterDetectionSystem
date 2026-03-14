@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo === GitHub'a Push ===
git branch -M main
git push -u origin main

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Hata: Push basarisiz. GitHub giris bilgisi gerekebilir.
    echo GitHub'da: Settings - Developer settings - Personal access tokens
    echo ile token olusturup sifre yerine kullanin.
)
pause
