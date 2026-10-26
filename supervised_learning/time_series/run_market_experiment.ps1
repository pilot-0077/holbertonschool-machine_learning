param(
    [string]$DataFile = "btcusd_bitstamp_1min_2012-2025.csv.gz",
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
}

$Python = Join-Path ".venv" "Scripts\python.exe"

& $Python -m pip install --upgrade pip
& $Python -m pip install -r requirements.txt

& $Python preprocess_data.py $DataFile --output btc_market_only.npz
& $Python forecast_btc.py --data btc_market_only.npz --model models\btc_market_only.keras --plot-dir results\market_only --epochs $Epochs --batch-size 64

Write-Host ""
Write-Host "Experiment complete. Check results\market_only for the plots and CSV."
