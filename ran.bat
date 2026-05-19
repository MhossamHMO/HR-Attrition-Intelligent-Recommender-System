@echo off
cd /d "%~dp0"
call .idss\Scripts\activate
streamlit run app.py
pause