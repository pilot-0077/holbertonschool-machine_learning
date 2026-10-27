param(
    [string]$DataFile = "btcusd_bitstamp_1min_2012-2025.csv.gz",
    [string]$StartDate = "2017-01-01",
    [string]$EndDate = "2025-01-07 00:00:00",
    [int]$Epochs = 50
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $DataFile)) {
    throw "Dataset not found: $DataFile. Download the CSV.GZ file or pass its path with -DataFile."
}

if (-not (Test-Path ".venv")) {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        py -3 -m venv .venv
    } else {
        python -m venv .venv
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create the Python virtual environment."
    }
}

$Python = Join-Path ".venv" "Scripts\python.exe"

& $Python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    throw "Failed to upgrade pip. The experiment has stopped."
}

& $Python -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    throw "Dependency installation failed. The experiment has stopped before preprocessing or training."
}

& $Python preprocess_data.py $DataFile --start-date $StartDate --end-date $EndDate --output btc_market_only_returns.npz
if ($LASTEXITCODE -ne 0) {
    throw "Data preprocessing failed. The experiment has stopped before training."
}

& $Python forecast_btc.py --data btc_market_only.npz --model models\btc_market_only.keras --plot-dir results\market_only --epochs $Epochs --batch-size 64
if ($LASTEXITCODE -ne 0) {
    throw "Model training or evaluation failed. Check the error above; no successful experiment is reported."
}

Write-Host ""
Write-Host "Experiment complete. Check results\market_only for the plots and CSV."
