@echo off
setlocal
cd /d "%~dp0"
MCP_Langchain_Langraph_Tools.py --self-test
if errorlevel 1 exit /b %errorlevel%
MCP_Langchain_Langraph_Tools.py --topic all
endlocal
