@echo off
cd /d C:\NEWW\volatix_ai
"%~dp0.venv\Scripts\python.exe" scripts\nightly_etl.py >> logs\nightly_etl.log 2>&1
