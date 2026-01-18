@echo off
echo Starting RAG Knowledge Assistant...
echo.
cd /d %~dp0
streamlit run web_app/app.py
pause
