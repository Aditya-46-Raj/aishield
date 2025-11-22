@echo off
cd /d "%~dp0frontend"
streamlit run app.py --server.port 8501
