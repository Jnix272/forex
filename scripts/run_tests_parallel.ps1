#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Run the forex_scaling test suite in parallel using PowerShell background jobs.
    Each test group runs in its own Python process (avoids pyarrow/execnet crash).
.PARAMETER Group
    Test category: "all", "fast", "models", "features", "training", "data",
    "system", "smoke", "labeling", or a comma-separated list of test files.
.PARAMETER Workers
    Max parallel jobs (default: auto = half of CPU count).
.PARAMETER NoSkip
    Include normally-skipped tests (Stooq-dependent, slow E2E).
.PARAMETER ExitFirst
    Stop on first failure (-x).
.PARAMETER Slow
    Show 10 slowest tests per group.
.EXAMPLE
    .\scripts\run_tests_parallel.ps1 -Group fast -Workers 4
    .\scripts\run_tests_parallel.ps1 -Group smoke -ExitFirst
    .\scripts\run_tests_parallel.ps1 -Group "tests/test_smoke.py,tests/test_all.py"
#>

param(
    [string]$Group = "all",
    [int]$Workers = [Math]::Max(1, [System.Environment]::ProcessorCount / 2),
    [switch]$NoSkip,
    [switch]$ExitFirst,
    [switch]$Slow
)

$RepoRoot = Split-Path -Parent $PSScriptRoot
$Python   = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$Env:PYTHONPATH = $RepoRoot

if (-not (Test-Path $Python)) {
    Write-Error ".venv not found at $RepoRoot\.venv"
    exit 1
}

# Map named groups to test file paths
$groups = @{
    all      = @("tests/")
    fast     = @("tests/", "--ignore=tests/test_e2e_real_data.py",
                 "--ignore=tests/test_intermarket_fred.py",
                 "--ignore=tests/test_model_full_data_flow.py",
                 "--ignore=tests/test_training_smoke.py")
    models   = "tests/test_models.py", "tests/test_model_behavior.py",
               "tests/test_model_diagnostics.py", "tests/test_all_models_build.py",
               "tests/test_inference_consistency.py"
    features = "tests/test_feature_pipeline.py", "tests/test_intermarket_fred.py",
               "tests/test_review_fixes_smoke.py"
    training = "tests/test_training_smoke.py", "tests/test_training_utils.py",
               "tests/test_training_memory_compat.py", "tests/test_adaptive_curriculum.py",
               "tests/test_priority5_model_training.py"
    data     = "tests/test_data_download.py", "tests/test_labeling_pipeline.py",
               "tests/test_dashboard.py", "tests/test_download_historical_news.py",
               "tests/test_scrape_historical_news.py"
    system   = "tests/test_system.py", "tests/test_smoke.py",
               "tests/test_api.py", "tests/test_api_signature_compat.py",
               "tests/test_risk_execution.py", "tests/test_execution_realism.py",
               "tests/test_ensemble_risk.py", "tests/test_ensemble_deep.py"
    smoke    = "tests/test_smoke.py", "tests/test_system.py",
               "tests/test_all.py", "tests/test_all_models_build.py",
               "tests/test_config.py"
    labeling = "tests/test_labeling_pipeline.py", "tests/test_execution_realism.py",
               "tests/test_rl_report.py", "tests/test_rl_market_arrays.py",
               "tests/test_rl_train_window.py"
}

if ($Group -match ",") {
    # Custom comma-separated file list
    $testPaths = $Group -split "," | ForEach-Object { $_.Trim() }
} elseif ($groups.ContainsKey($Group)) {
    $testPaths = $groups[$Group]
} else {
    Write-Error "Unknown group '$Group'. Valid: $($groups.Keys -join ', ')"
    exit 1
}

# Build base pytest args
$pytestArgs = @("--tb=short", "-q")
if ($Slow) { $pytestArgs += "--durations=10" }
if ($ExitFirst) { $ExitFirstOverride = "-x" } else { $ExitFirstOverride = "" }

$skipFlag = ""
if (-not $NoSkip) {
    $skipFlag = "-k not slow"
}

Write-Host "=" x 60
Write-Host "  forex_scaling  |  Group: $Group  |  Jobs: $Workers  |  $(Get-Date -Format 'yyyy-MM-dd HH:mm')"
Write-Host "  Test paths: $($testPaths -join ' ')"
Write-Host "=" x 60

# If group is "all" or "fast", run as single job (pytest handles collection)
if ($Group -in @("all", "fast")) {
    $cmd = @($Python, "-m", "pytest") + $pytestArgs + @($skipFlag, $ExitFirstOverride) + $testPaths
    & $cmd 2>&1
    exit $LASTEXITCODE
}

# Split into file-level chunks and run in parallel via background jobs
$jobs = @()
$sw = [System.Diagnostics.Stopwatch]::StartNew()

foreach ($path in $testPaths) {
    $cmdArgs = @($pytestArgs)
    if ($ExitFirstOverride) { $cmdArgs += $ExitFirstOverride }
    $cmdArgs += $path

    $job = Start-Job -ScriptBlock {
        param($python, $argsList, $skip)
        $fullArgs = @("-m", "pytest") + $argsList
        if ($skip) { $fullArgs += @("-k", "not slow") }
        $result = & $python $fullArgs 2>&1 | Out-String
        $exitCode = $LASTEXITCODE
        @{ Output = $result; ExitCode = $exitCode }
    } -ArgumentList $Python, $cmdArgs, $skipFlag

    $jobs += @{ Job = $job; Path = $path }
}

# Wait for jobs and collect results
$failed = @()
$totalPassed = 0; $totalFailed = 0; $totalSkipped = 0; $totalErrors = 0
$global:allOutput = [System.Text.StringBuilder]::new()

while ($jobs.Count -gt 0) {
    $done = $jobs | Where-Object { $_.Job.State -ne "Running" }
    foreach ($item in $done) {
        $result = Receive-Job -Job $item.Job -AutoRemoveJob -Wait
        $output = $result.Output
        $exitCode = $result.ExitCode

        $null = $global:allOutput.AppendLine("--- $($item.Path) ---")
        $null = $global:allOutput.AppendLine($output)

        # Parse summary line: "N passed, M skipped, K failed, L errors"
        if ($output -match "(\d+) passed") { $totalPassed += [int]$Matches[1] }
        if ($output -match "(\d+) failed") { $totalFailed += [int]$Matches[1] }
        if ($output -match "(\d+) skipped") { $totalSkipped += [int]$Matches[1] }
        if ($output -match "(\d+) errors") { $totalErrors += [int]$Matches[1] }

        if ($exitCode -ne 0) {
            $failed += $item.Path
            if ($ExitFirst) { break }
        }

        $jobs = $jobs | Where-Object { $_.Job.Id -ne $item.Job.Id }
    }
    if ($ExitFirst -and $failed.Count -gt 0) { break }
    Start-Sleep -Milliseconds 200
}

# Stop any remaining jobs
$jobs | ForEach-Object { Stop-Job -Job $_.Job -ErrorAction SilentlyContinue; Remove-Job -Job $_.Job -ErrorAction SilentlyContinue }
$sw.Stop()

# Print collected output
Write-Host $global:allOutput.ToString()

Write-Host "=" x 60
Write-Host "  RESULTS: $totalPassed passed, $totalFailed failed, $totalSkipped skipped, $totalErrors errors"
Write-Host "  Duration: $($sw.Elapsed.TotalSeconds.ToString('F1'))s"
if ($failed.Count -gt 0) {
    Write-Host "  FAILED: $($failed -join ', ')"
}
Write-Host "=" x 60
exit ($failed.Count -gt 0 ? 1 : 0)
