@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Follow START_HERE.md to create the Python environment first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m streamlit run app.py --browser.gatherUsageStats false
pause
