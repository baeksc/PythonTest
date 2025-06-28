@echo off
REM 가상환경 활성화 및 예측 코드 실행
cd /d %~dp0
call .\my_312_project_env\Scripts\activate.bat
python samsung_arima_forecast.py
pause
