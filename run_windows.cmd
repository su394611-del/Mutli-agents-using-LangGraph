@echo off
setlocal
cd /d "%~dp0"
where python >nul 2>&1
if errorlevel 1 (
  echo Python was not found on PATH. Activate your virtual environment first.
  exit /b 1
)
python "langgraph_example_multi_agents.py" --self-test
if errorlevel 1 exit /b %errorlevel%
python "langgraph_example_multi_agents.py"
exit /b %errorlevel%
