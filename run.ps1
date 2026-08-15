# run.ps1 — simple PowerShell entry point for the forex ML pipeline.
#
# Usage:
#   .\run.ps1 download --start 2017-02-18
#   .\run.ps1 migrate
#   .\run.ps1 validate
#   .\run.ps1 data --start 2017-02-18
#   .\run.ps1 train
#   .\run.ps1 train --quick
#   .\run.ps1 backtest
#   .\run.ps1 all --start 2017-02-18

param(
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidateSet("download", "migrate", "validate", "data", "train", "backtest", "all")]
    [string]$Command,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Args
)

$Python = Join-Path $PSScriptRoot ".venv-gpu\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python"
}

$Pipeline = Join-Path $PSScriptRoot "scripts\run_pipeline.py"
& $Python $Pipeline $Command @Args
exit $LASTEXITCODE
