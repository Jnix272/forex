# scripts/run_phase1_oanda.ps1
# Loads credentials from .env and starts the Phase 1 OANDA live daemon.

param(
    [string]$BearerToken = "",
    [string]$AccountId   = "",
    [string]$EnvType     = "",
    [string]$Pairs       = "EURUSD,GBPUSD,USDCAD,USDJPY",
    [double]$Equity      = 98721.52,
    [double]$MaxLots     = 0.2
)

# 1. Load .env if present
$envPath = Join-Path $PSScriptRoot "..\.env"
if (Test-Path $envPath) {
    Write-Host "[Phase 1] Loading configuration from .env..." -ForegroundColor DarkGray
    Get-Content $envPath | Where-Object { $_ -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$' } | ForEach-Object {
        $k = $matches[1].Trim()
        $v = $matches[2].Trim().Trim('"').Trim("'")
        if (-not [string]::IsNullOrWhiteSpace($k)) {
            [System.Environment]::SetEnvironmentVariable($k, $v, [System.EnvironmentVariableTarget]::Process)
        }
    }
}

if (-not $BearerToken) {
    if ($env:OANDA_API_KEY) { $BearerToken = $env:OANDA_API_KEY }
    elseif ($env:OANDA_BEARER_TOKEN) { $BearerToken = $env:OANDA_BEARER_TOKEN }
}
if (-not $AccountId) {
    if ($env:OANDA_ACCOUNT_ID) { $AccountId = $env:OANDA_ACCOUNT_ID }
    else { $AccountId = "101-001-38834567-001" }
}
if (-not $EnvType) {
    if ($env:OANDA_ENV) { $EnvType = $env:OANDA_ENV }
    else { $EnvType = "practice" }
}

if (-not $BearerToken) {
    Write-Error "No OANDA API key found! Please specify OANDA_API_KEY in .env or pass -BearerToken."
    exit 1
}

$env:OANDA_API_KEY       = $BearerToken
$env:OANDA_BEARER_TOKEN  = $BearerToken
$env:OANDA_ACCOUNT_ID    = $AccountId
$env:OANDA_ENV           = $EnvType
$env:OANDA_UNITS_PER_LOT = "10000"
$env:CROSS_ASSET_SOURCE  = "yahoo"
$env:SENTIMENT_CACHE_DIR = "data/embeddings/live"
# Real-tick ZMQ feed from C++ oanda_stream (sub-ms, no REST polling). Leave unset to use REST polling.
if (-not $env:OANDA_ZMQ_ENDPOINT) { $env:OANDA_ZMQ_ENDPOINT = "tcp://127.0.0.1:5557" }
# FREENEWS_API_KEY already loaded from .env (a50b...); fallback is data/news/latest_headlines.json with headlines array

Write-Host "[Phase 1] Launching OANDA Live Paper Trading Daemon on $EnvType account $AccountId" -ForegroundColor Cyan
Write-Host "[Phase 1] Pairs: $Pairs | Equity: `$$Equity | Max Lots: $MaxLots" -ForegroundColor Cyan
Write-Host "[Phase 1] ZMQ: $env:OANDA_ZMQ_ENDPOINT | Guard: pair-adaptive max_spread (EURUSD 2.5/2.5, USDCAD/JPY 3.0/3.5)" -ForegroundColor DarkGray
Write-Host "[Phase 1] Sizing: RCK confidence-scaled (0.02..max_lots) | Hedge: per-pair hedge_weights_{pair}.json (verify after 100 bars)" -ForegroundColor DarkGray

while ($true) {
    & "d:/forex-main/.venv311/Scripts/python.exe" -u trading/live_engine.py `
        --broker oanda `
        --pairs $Pairs `
        --model ensemble `
        --equity $Equity `
        --max-lots $MaxLots `
        --strategy-mode scalping `
        --sentiment-mode finbert `
        --journal-path logs/live/oanda_paper_phase1.jsonl

    $code = $LASTEXITCODE
    Write-Host "[Phase 1 Daemon] Process exited (code $code). Auto-restarting in 5s (Ctrl+C to abort)..." -ForegroundColor Yellow
    Start-Sleep -Seconds 5
}
