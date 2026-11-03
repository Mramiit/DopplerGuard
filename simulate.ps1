# DopplerGuard Quick Simulation Runner for PowerShell
param(
    [string]$Scenario = "direct_ingress",
    [int]$Duration = 25
)

$PythonExe = "C:\Users\itzmr\AppData\Local\Programs\Python\Python313\python.exe"

if (-not (Test-Path $PythonExe)) {
    $PythonExe = "python"
}

Write-Host ">>> Running DopplerGuard Simulation: Scenario='$Scenario', Duration=${Duration}s <<<" -ForegroundColor Cyan
& $PythonExe simulation/run_simulation.py --scenario $Scenario --duration $Duration

Write-Host "`n>>> Rendering Tactical Radar HUD Scope... <<<" -ForegroundColor Cyan
& $PythonExe simulation/plot_simulation_run.py
