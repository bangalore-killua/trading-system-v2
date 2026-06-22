# setup.ps1 - One-shot setup for trading-system-v2
#
# Run from repo root after main requirements.txt install:
#   cd C:\Users\ASUS\git_repos\trading-system-v2
#   .\scripts\setup.ps1
#
# Steps:
#   1. Install pandas-ta (excluded from requirements.txt due to pip memory crash)
#   2. Install TA-Lib (Windows wheel -- needed for 60 candlestick pattern features)
#   3. Pre-download FinBERT model (~400 MB) to avoid delay on first news event
#   4. Verify spaCy en_core_web_sm is installed
#   5. Smoke-test tradingview-screener v3.2.1 API compatibility
#   6. Bootstrap LightGBM (optional, ~30 min)
#
# Going LIVE checklist (manual, not in this script):
#   [ ] Get free OpenBB PAT at https://my.openbb.co - add to .env as OPENBB_PAT=
#   [ ] .env: ALPACA_PAPER=false + live Alpaca API keys
#   [ ] config.json: "alpaca": { "paper": false }
#   [ ] Apply for Alpaca options trading (Level 2) if options.enabled=true
#   [ ] Set up nightly pg_dump backup via Task Scheduler
#   [ ] Train YOLOv8 chart pattern model: .\scripts\train_yolov8.ps1

$ErrorActionPreference = "Continue"
$repo = Split-Path -Parent $PSScriptRoot

Write-Host ""
Write-Host "=== trading-system-v2 Setup ===" -ForegroundColor Cyan
Write-Host "Repo: $repo"
Write-Host ""

# Activate venv if present
$venv = "$repo\venv\Scripts\Activate.ps1"
if (Test-Path $venv) {
    . $venv
    Write-Host "[OK] venv activated" -ForegroundColor Green
} else {
    Write-Host "[WARN] venv not found - using system Python" -ForegroundColor Yellow
}

Set-Location $repo

# ────────────────────────────────────────────────────────────────────────────
# Step 1: pandas-ta
# ────────────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "Step 1/6: Installing pandas-ta 0.4.71b0..." -ForegroundColor Cyan

uv pip install "pandas-ta==0.4.71b0" --prerelease=allow
if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] pandas-ta installed" -ForegroundColor Green
} else {
    Write-Host "[FAIL] pandas-ta install failed" -ForegroundColor Red
    Write-Host "  Try manually: uv pip install pandas-ta==0.4.71b0 --prerelease=allow"
}

# ────────────────────────────────────────────────────────────────────────────
# Step 2: TA-Lib (Windows pre-compiled wheel)
# pandas-ta delegates all cdl_pattern() calls to TA-Lib. Without it, all
# 60 candlestick features in the LightGBM vector are permanently zero.
# The wheel is distributed via cgohlke's GitHub releases (unofficial but
# the standard approach for Windows Python users without conda).
# ────────────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "Step 2/6: Installing TA-Lib (candlestick pattern engine)..." -ForegroundColor Cyan
Write-Host "         Without this, all 60 CDL features in LightGBM are zero."

# Check if already installed
python -c "import talib; print('  TA-Lib', talib.__version__, 'already installed')" 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] TA-Lib already present" -ForegroundColor Green
} else {
    # ta-lib-binary only supports Python ≤3.8; use cgohlke's pre-compiled wheel
    $py_ver   = python -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')"
    $whl_name = "TA_Lib-0.4.32-cp${py_ver}-cp${py_ver}-win_amd64.whl"
    $whl_url  = "https://github.com/cgohlke/talib-build/releases/download/v0.4.32/$whl_name"
    $whl_path = "$env:TEMP\$whl_name"
    Write-Host "  Downloading pre-compiled wheel for Python $py_ver from cgohlke/talib-build..."
    try {
        Invoke-WebRequest -Uri $whl_url -OutFile $whl_path -UseBasicParsing
        uv pip install $whl_path
        Remove-Item $whl_path -ErrorAction SilentlyContinue
        if ($LASTEXITCODE -eq 0) {
            Write-Host "[OK] TA-Lib installed from cgohlke wheel" -ForegroundColor Green
        } else {
            Write-Host "[FAIL] TA-Lib wheel install failed" -ForegroundColor Red
            Write-Host "  Manual fallback: conda install -c conda-forge ta-lib" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "[FAIL] Could not download TA-Lib wheel (no internet?): $_" -ForegroundColor Red
        Write-Host "  Manual options:" -ForegroundColor Yellow
        Write-Host "    conda install -c conda-forge ta-lib" -ForegroundColor Yellow
        Write-Host "    or download wheel from https://github.com/cgohlke/talib-build/releases" -ForegroundColor Yellow
    }
}

# ────────────────────────────────────────────────────────────────────────────
# Step 3: FinBERT pre-download
# Write Python to a temp file to avoid PowerShell here-string quoting issues
# ────────────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "Step 3/6: Pre-downloading FinBERT model (~400 MB)..." -ForegroundColor Cyan
Write-Host "         Prevents 2-4 min delay on the first live news event."

$finbert_script = "$repo\scripts\_tmp_finbert_download.py"
Set-Content -Path $finbert_script -Encoding utf8 -Value @'
# if __name__ == "__main__" is REQUIRED on Windows.
# from_pretrained() internally spawns a subprocess for weight loading;
# without this guard, Python raises RuntimeError about bootstrapping phase.
if __name__ == "__main__":
    import os
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    model_name = "yiyanghkust/finbert-tone"
    print("  Downloading", model_name, "...")
    AutoTokenizer.from_pretrained(model_name)
    AutoModelForSequenceClassification.from_pretrained(model_name)
    print("  FinBERT cached and ready.")
'@

python $finbert_script
if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] FinBERT cached" -ForegroundColor Green
} else {
    Write-Host "[FAIL] FinBERT download failed - will auto-download on first use" -ForegroundColor Yellow
}
Remove-Item $finbert_script -ErrorAction SilentlyContinue

# ────────────────────────────────────────────────────────────────────────────
# Step 4: spaCy model
# ────────────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "Step 4/6: Verifying spaCy en_core_web_sm..." -ForegroundColor Cyan

python -c "import spacy; spacy.load('en_core_web_sm'); print('  en_core_web_sm OK')"
if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] spaCy model present" -ForegroundColor Green
} else {
    Write-Host "  Not installed - downloading now..."
    python -m spacy download en_core_web_sm
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] spaCy model installed" -ForegroundColor Green
    } else {
        Write-Host "[FAIL] spaCy model install failed" -ForegroundColor Red
    }
}

# ────────────────────────────────────────────────────────────────────────────
# Step 5: tradingview-screener API smoke test
# ────────────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "Step 5/6: Testing tradingview-screener v3.2.1 API..." -ForegroundColor Cyan

$tv_script = "$repo\scripts\_tmp_tv_test.py"
Set-Content -Path $tv_script -Encoding utf8 -Value @'
from tradingview_screener import Query, Column
print("  Running test scan (limit=3)...")
r = Query().select("name", "close").limit(3).get_scanner_data()
count, df = r if isinstance(r, tuple) else (0, r)
print("  Got", len(df), "rows - API compatible")
'@

python $tv_script
if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] tradingview-screener API working" -ForegroundColor Green
} else {
    Write-Host "[FAIL] tradingview-screener API test failed" -ForegroundColor Red
    Write-Host "  Check: data/tv_screener.py line 84 and compare with the library's current README"
}
Remove-Item $tv_script -ErrorAction SilentlyContinue

# ────────────────────────────────────────────────────────────────────────────
# Step 6: LightGBM bootstrap (optional, interactive)
# ────────────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "Step 6/6: LightGBM meta-model bootstrap" -ForegroundColor Cyan
Write-Host "  Pulls 5 years of OHLCV for 10 core symbols, labels each bar"
Write-Host "  via triple-barrier method, trains LightGBM. Takes ~30 minutes."
Write-Host "  Run this AFTER TA-Lib is installed so CDL features are non-zero."
Write-Host ""

$talib_ok = python -c "import talib" 2>$null; $talib_installed = ($LASTEXITCODE -eq 0)
if (-not $talib_installed) {
    Write-Host "  [WARN] TA-Lib not installed - CDL features will be zero in model." -ForegroundColor Yellow
    Write-Host "  Install TA-Lib first, then re-run bootstrap for a better model." -ForegroundColor Yellow
    Write-Host ""
}

try {
    $answer = Read-Host "  Run LightGBM bootstrap now? (y/N)"
} catch {
    $answer = "N"
    Write-Host "  (non-interactive shell - skipping prompt, answer: N)" -ForegroundColor Yellow
}
if ($answer -match "^[Yy]$") {
    Write-Host "  Starting - do not close this window..." -ForegroundColor Yellow
    python -m strategy.training_pipeline --symbols AAPL MSFT NVDA GOOGL META AMZN TSLA AMD CRM NFLX --years 5
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] Bootstrap complete - model saved to models/" -ForegroundColor Green
        Write-Host "  NEXT: set config.json -> model -> meta_model_enabled: true"
    } else {
        Write-Host "[FAIL] Bootstrap failed - check logs/system.log" -ForegroundColor Red
    }
} else {
    Write-Host "  Skipped. Run later:" -ForegroundColor Yellow
    Write-Host "    python -m strategy.training_pipeline --symbols AAPL MSFT NVDA GOOGL META AMZN TSLA AMD CRM NFLX --years 5"
    Write-Host "  Then: config.json -> model -> meta_model_enabled: true"
}

# ────────────────────────────────────────────────────────────────────────────
# Summary
# ────────────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "=== Setup Complete ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "Start paper trading:" -ForegroundColor White
Write-Host "  python run_server.py"
Write-Host ""
Write-Host "YOLOv8 chart pattern training (optional, improves 15 features):" -ForegroundColor Yellow
Write-Host "  .\scripts\train_yolov8.ps1"
Write-Host ""
Write-Host "Going live checklist:" -ForegroundColor Yellow
Write-Host "  [ ] Get free OpenBB PAT at https://my.openbb.co -> .env OPENBB_PAT="
Write-Host '  [ ] config.json "alpaca" -> "paper": false'
Write-Host "  [ ] .env: ALPACA_PAPER=false + live Alpaca API keys"
Write-Host "  [ ] After TA-Lib + retrain: config.json model -> meta_model_enabled: true"
Write-Host "  [ ] Nightly pg_dump backup via Windows Task Scheduler"
Write-Host ""
