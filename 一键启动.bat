@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
title AutoVidDub Setup
cls
echo ============================================================
echo   AutoVidDub First-Time Setup
echo ============================================================
echo.
if not exist ".venv\Scripts\python.exe" (
  echo [1/5] Creating Python environment...
  python -m venv .venv
  if errorlevel 1 goto :error
) else (
  echo [1/5] Python environment already exists.
)
call ".venv\Scripts\activate.bat"
echo [2/5] Updating installer...
python -m pip install -U pip setuptools wheel
if errorlevel 1 goto :error
echo [3/5] Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 goto :error
if not exist "vendor\CosyVoice\cosyvoice\cli\cosyvoice.py" (
  echo [4/5] Downloading official CosyVoice source...
  git clone --depth 1 --recursive https://github.com/FunAudioLLM/CosyVoice.git vendor\CosyVoice
  if errorlevel 1 goto :error
) else (
  echo [4/5] CosyVoice source already exists.
)
nvidia-smi >nul 2>nul
if not errorlevel 1 (
  python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" >nul 2>nul
  if errorlevel 1 (
    echo Installing CUDA 12.8 PyTorch for NVIDIA GPU...
    pip install -U --force-reinstall torch torchaudio --index-url https://download.pytorch.org/whl/cu128
    if errorlevel 1 goto :error
  ) else (
    echo CUDA PyTorch is ready.
  )
)
echo [5/6] Installing React frontend dependencies...
pushd frontend
call npm install
if errorlevel 1 (
  popd
  goto :error
)
popd
echo [6/6] Downloading models from ModelScope...
python download_models.py --source modelscope
if errorlevel 1 goto :error
echo.
echo Setup complete. Starting the backend terminal...
call 启动.bat
exit /b %errorlevel%
:error
echo.
echo ============================================================
echo Setup failed. Review the error messages above.
echo ============================================================
pause
exit /b 1
