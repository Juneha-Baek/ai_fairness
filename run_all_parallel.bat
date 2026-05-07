@echo off
REM ============================================================
REM Run All Power Analyses (Parallel) - Windows
REM ============================================================
REM Launches all four analysis modes in parallel windows.
REM Total wall-clock time is roughly the longest single mode (~1.5 hours).
REM
REM Usage:
REM   1. Place this file in the same folder as the two Python scripts.
REM   2. Double-click, or run from a command prompt:
REM      run_all_parallel.bat
REM ============================================================

setlocal

REM Use the script's location as the working directory
cd /d "%~dp0"

echo ==========================================================
echo Power Analysis - Parallel Execution
echo Started: %DATE% %TIME%
echo ==========================================================
echo.
echo Working directory: %CD%
echo.

REM Check that both Python scripts exist in the current folder
if not exist "Study1_PowerAnalysis.py" (
    echo ERROR: Study1_PowerAnalysis.py not found in current folder
    pause
    exit /b 1
)
if not exist "Study2_PowerAnalysis.py" (
    echo ERROR: Study2_PowerAnalysis.py not found in current folder
    pause
    exit /b 1
)

REM Verify Python is on PATH
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Make sure Python is installed and added to PATH.
    pause
    exit /b 1
)

echo Launching 4 simulations in parallel...
echo Each will open in its own window. Do not close them until they finish.
echo.

REM Launch each mode in its own command window
start "Study 1 H1H2H3 (~30-45min)" cmd /k "python Study1_PowerAnalysis.py --mode h1h2h3 && echo. && echo === DONE === && pause"

start "Study 1 H4H5 (~50-60min)" cmd /k "python Study1_PowerAnalysis.py --mode h4h5 && echo. && echo === DONE === && pause"

start "Study 2 H6789 (~20min)" cmd /k "python Study2_PowerAnalysis.py --mode h6789 && echo. && echo === DONE === && pause"

start "Study 2 H1011 (~1-1.5hr)" cmd /k "python Study2_PowerAnalysis.py --mode h1011 && echo. && echo === DONE === && pause"

echo.
echo ==========================================================
echo All 4 simulations launched in separate windows.
echo Estimated total time: ~1.5 hours (longest task)
echo.
echo Each window will show its progress and stay open when done.
echo Results will be saved in folders named:
echo   results_study1_h1h2h3_YYYYMMDD_HHMMSS\
echo   results_study1_h4h5_YYYYMMDD_HHMMSS\
echo   results_study2_h6789_YYYYMMDD_HHMMSS\
echo   results_study2_h1011_YYYYMMDD_HHMMSS\
echo ==========================================================
echo.

pause
endlocal
