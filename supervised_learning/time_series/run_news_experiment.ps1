param(
    [string]$DataFile = "btcusd_bitstamp_1min_2012-2025.csv.gz",
    [string]$NewsFile = "cryptovision.csv",
    [string]$StartDate = "2017-01-01",
    [string]$EndDate = "2025-01-07 00:00:00",
    [int]$Epochs = 50
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $DataFile)) {
    throw "Market dataset not found: $DataFile"
}
if (-not (Test-Path $NewsFile)) {
    throw "News dataset not found: $NewsFile. Download and extract the CSV first."
}

$Python = Join-Path ".venv" "Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "Virtual environment not found. Run the market-only experiment first."
}

& $Python preprocess_data.py $DataFile --news-file $NewsFile --start-date $StartDate --end-date $EndDate --output btc_with_news_returns.npz
if ($LASTEXITCODE -ne 0) {
    throw "News preprocessing failed. Check the dataset columns and timestamps."
}

& $Python forecast_btc.py --data btc_with_news_returns.npz --model models\btc_with_news_returns.keras --plot-dir results\with_news_returns --epochs $Epochs --batch-size 64
if ($LASTEXITCODE -ne 0) {
    throw "News model training or evaluation failed. Check the error above."
}

Write-Host ""
Write-Host "News experiment complete."
Write-Host "Compare these metrics with the market-only results from results\market_only_returns."
Write-Host "News model outputs: results\with_news_returns"
