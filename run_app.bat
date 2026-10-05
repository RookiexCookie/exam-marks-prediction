@echo off
echo ===================================================
echo Starting Exam Marks Prediction System (FastAPI)
echo ===================================================
echo Backend API and Dashboard running on http://127.0.0.1:8080
echo Swagger API Documentation on http://127.0.0.1:8080/docs
echo ===================================================
start "" "http://127.0.0.1:8080"
.venv\Scripts\uvicorn.exe backend.main:app --host 127.0.0.1 --port 8080 --reload
