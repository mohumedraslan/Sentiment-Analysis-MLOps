@echo off
echo Running preprocessing...
python src/preprocess.py
if errorlevel 1 (
    echo Preprocessing failed
    exit /b 1
)
echo Running training...
python src/train.py
if errorlevel 1 (
    echo Training failed
    exit /b 1
)
echo Building Docker image...
docker build -t sentiment-api .
if errorlevel 1 (
    echo Docker build failed
    exit /b 1
)
echo Running Docker container...
docker run -d -p 8000:8000 sentiment-api
if errorlevel 1 (
    echo Docker run failed
    exit /b 1
)
echo Pipeline completed successfully!