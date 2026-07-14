param(
    [switch]$Smoke,
    [int]$Epochs = 100,
    [int]$Batch = 1,
    [string]$Name = "scratch-rtdetr-l-btdse-100ep",
    [int]$MaxRestarts = 20,
    [int]$RestartDelaySeconds = 30
)

$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent $PSScriptRoot
$Python = "C:\uav_env\Scripts\python.exe"
$LogDir = Join-Path $Root "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$RunName = if ($Smoke) { "$Name-smoke" } else { $Name }
$RunDir = Join-Path $Root "runs\btdse\$RunName"
$TrainingLog = Join-Path $LogDir "btdse_local_latest.log"
$SupervisorLog = Join-Path $LogDir "btdse_local_supervisor.log"

function Write-SupervisorEvent {
    param([string]$Message)
    $Line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Write-Host $Line -ForegroundColor Cyan
    Add-Content -Path $SupervisorLog -Value $Line
}

function Test-TrainingComplete {
    $Results = Join-Path $RunDir "results.csv"
    if (-not (Test-Path $Results)) {
        return $false
    }
    try {
        $LastResult = Import-Csv $Results | Select-Object -Last 1
        return ([int]$LastResult.epoch -ge $Epochs)
    }
    catch {
        return $false
    }
}

function Find-ResumeCheckpoint {
    $Output = & $Python "scripts\find_btdse_checkpoint.py" "--run-dir" $RunDir 2>$null
    if ($LASTEXITCODE -ne 0 -or -not $Output) {
        return $null
    }
    return ($Output | Select-Object -Last 1).ToString().Trim()
}

Add-Type @"
using System;
using System.Runtime.InteropServices;
public static class TrainingPowerState {
    [DllImport("kernel32.dll")]
    public static extern uint SetThreadExecutionState(uint flags);
}
"@
$ES_CONTINUOUS = 0x80000000
$ES_SYSTEM_REQUIRED = 0x00000001
[TrainingPowerState]::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED) | Out-Null

Set-Location $Root
$ExitCode = 1
$RestartCount = 0
$ResumeCheckpoint = Find-ResumeCheckpoint

try {
    if (Test-TrainingComplete) {
        Write-SupervisorEvent "Training already contains $Epochs completed epochs: $RunDir"
        $ExitCode = 0
    }
    else {
        while ($true) {
            $Arguments = @(
                "scripts\train_rtdetr_btdse.py",
                "--epochs", $Epochs,
                "--batch", $Batch,
                "--imgsz", 640,
                "--workers", 2,
                "--device", 0,
                "--name", $Name
            )
            if ($Smoke) {
                $Arguments += "--smoke"
            }
            if ($ResumeCheckpoint) {
                $Arguments += @("--resume", $ResumeCheckpoint)
                Write-SupervisorEvent "Starting attempt $($RestartCount + 1) from $ResumeCheckpoint"
            }
            else {
                Write-SupervisorEvent "Starting attempt $($RestartCount + 1) from scratch"
            }

            & $Python @Arguments 2>&1 | ForEach-Object { $_.ToString() } | Tee-Object -FilePath $TrainingLog -Append
            $ExitCode = $LASTEXITCODE

            if ($ExitCode -eq 0 -or (Test-TrainingComplete)) {
                Write-SupervisorEvent "Training finished successfully with exit code $ExitCode"
                $ExitCode = 0
                break
            }

            $RestartCount += 1
            if ($RestartCount -gt $MaxRestarts) {
                Write-SupervisorEvent "Stopped after $MaxRestarts automatic restarts; last exit code was $ExitCode"
                break
            }

            $ResumeCheckpoint = Find-ResumeCheckpoint
            if ($ResumeCheckpoint) {
                Write-SupervisorEvent "Validated recovery checkpoint: $ResumeCheckpoint"
            }
            else {
                Write-SupervisorEvent "No complete checkpoint yet; the next attempt will restart from scratch"
            }
            Write-SupervisorEvent "Restarting in $RestartDelaySeconds seconds"
            Start-Sleep -Seconds $RestartDelaySeconds
        }
    }
}
finally {
    [TrainingPowerState]::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null
}
exit $ExitCode
