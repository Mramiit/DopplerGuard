@echo off
set PYTHON_EXE=C:\Users\itzmr\AppData\Local\Programs\Python\Python313\python.exe

if not exist "%PYTHON_EXE%" (
    set PYTHON_EXE=python
)

"%PYTHON_EXE%" simulation/run_simulation.py %*
