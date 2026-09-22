param(
    [string]$BearerToken = $env:OANDA_BEARER_TOKEN,
    [string]$AccountId   = $env:OANDA_ACCOUNT_ID,
    [string]$EnvType     = "practice",
    [string]$Pairs       = "EURUSD,GBPUSD,USDCAD,USDJPY",
    [double]$Equity      = 98723.17,
    [double]$MaxLots     = 0.2
)

if (-not $BearerToken) {
    $BearerToken = "3359b89663aa9ea4d24caec312f48b0f-3d7e17c5fe2f300ca46b780f79099a6b"
}
if (-not $AccountId) {
    $AccountId = "101-001-38834567-001"
}

$env:OANDA_BEARER_TOKEN = $BearerToken
$env:OANDA_ACCOUNT_ID   = $AccountId
$env:OANDA_ENV          = $EnvType
$env:OANDA_UNITS_PER_LOT = "10000"
$env:CROSS_ASSET_SOURCE = "none"

Write-Host "[Phase 1] Launching OANDA Live Paper Trading Daemon on $EnvType account $AccountId" -ForegroundColor Cyan
Write-Host "[Phase 1] Pairs: $Pairs | Equity: `$$Equity | Max Lots: $MaxLots" -ForegroundColor Cyan

& "d:/forex-main/.venv311/Scripts/python.exe" -u trading/live_engine.py `
    --broker oanda `
    --pairs $Pairs `
    --model ensemble `
    --equity $Equity `
    --max-lots $MaxLots `
    --strategy-mode scalping `
    --max-spread-pips 15.0 `
    --sentiment-mode off `
    --journal-path logs/live/oanda_paper_phase1.jsonl
