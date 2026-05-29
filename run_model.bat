@echo off
REM Update PROJ_DIR and PYTHON_EXE to match your installation.
REM Full paths are required — Task Scheduler does not inherit user PATH.
SET PROJ_DIR=C:\Users\super\baseball-model
SET PYTHON_EXE=C:\Python314\python.exe

cd /d %PROJ_DIR%
%PYTHON_EXE% model.py >> %PROJ_DIR%\logs\model_out.txt 2>> %PROJ_DIR%\logs\model_err.txt
