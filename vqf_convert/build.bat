@echo off
setlocal
rem ================================================================
rem  vqf_convert one-click build (Windows / MinGW gcc x86_64)
rem  Output: .\vqf_conv.exe
rem ================================================================
cd /d "%~dp0"

where gcc >nul 2>nul
if errorlevel 1 (
    echo [ERR] gcc not found in PATH. Install MinGW-w64 first.
    exit /b 1
)

echo [build] vqf_convert (x86) -O2 ...
gcc -O2 -Wall -Wextra -I src ^
    src\conv_main.c src\model_cfgio.c src\model_layers.c ^
    src\quant_kernels.c src\qk_repack.c src\vqf_vision.c ^
    src\vllm_crypto.c ^
    -o vqf_conv.exe -lm -lbcrypt
if errorlevel 1 (
    echo [ERR] build failed (see errors above)
    exit /b 1
)

echo [ok] artifact: %cd%\vqf_conv.exe
vqf_conv.exe --help
endlocal
