# setup.ps1 - One-shot setup for trading-system-v2
#
# Run from repo root after main requirements.txt install:
#   cd C:\Users\ASUS\git_repos\trading-system-v2
#   .\scripts\setup.ps1
#
# Steps:
#   1. Install pandas-ta (excluded from requirements.txt due to pip memory crash)
#   2. Pre-download FinBERT model (~400 MB) to avoid delay on first news event
#   3. Verify spaCy en_core_web_sm is installed
#   4. Smoke-test tradingview-screener v3.2.1 API compatibility
#   5. Bootstrap LightGBM (optional, ~30 min)
#
# Going LIVE checklist (manual, not in this script):
#   [ ] Rotate API keys - Alpaca, Telegram, FMP, NewsAPI were in git history
#   [ ] Get free OpenBB PAT at https://my.openbb.co - add to .env as OPENBB_PAT=
#   [ ] .env: ALPACA_PAPER=false + live Alpaca API keys
#   [ ] config.json: "alpaca": { "paper": false }
#   [ ] Apply for Alpaca options trading (Level 2) if options.enabled=true
#   [ ] Set up nightly pg_dump backup via Task Scheduler

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
Write-Host "Step 1/5: Installing pandas-ta 0.4.71b0..." -ForegroundColor Cyan

uv pip install "pandas-ta==0.4.71b0" --prerelease=allow
if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] pandas-ta installed" -ForegroundColor Green
} else {
    Write-Host "[FAIL] pandas-ta install failed" -ForegroundColor Red
    Write-Host "  Try manually: uv pip install pandas-ta==0.4.71b0 --prerelease=allow"
}

# ────────────────────────────────────────────────────────────────────────────
# Step 2: FinBERT pre-download
# Write Python to a temp file to avoid PowerShell here-string quoting issues
# ────────────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "Step 2/5: Pre-downloading FinBERT model (~400 MB)..." -ForegroundColor Cyan
Write-Host "         Prevents 2-4 min delay on the first live news event."

$finbert_script = "$repo\scripts\_tmp_finbert_download.py"
Set-Content -Path $finbert_script -Encoding utf8 -Value @'
from transformers import AutoTokenizer, AutoModelForSequenceClassification
model_name = "yiyanghkust/finbert-tone"
print("  Downloading", model_name, "...")
AutoTokenizer.from_pretrained(model_name)
AutoModelForSequenceClassification.from_pretrained(model_name)
print("  FinBERT download complete.")
'@

python $finbert_script
if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] FinBERT cached" -ForegroundColor Green
} else {
    Write-Host "[FAIL] FinBERT download failed - will auto-download on first use" -ForegroundColor Yellow
}
Remove-Item $finbert_script -ErrorAction SilentlyContinue

# ────────────────────────────────────────────────────────────────────────────
# Step 3: spaCy model
# ────────────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "Step 3/5: Verifying spaCy en_core_web_sm..." -ForegroundColor Cyan

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
# Step 4: tradingview-screener API smoke test
# ────────────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "Step 4/5: Testing tradingview-screener v3.2.1 API..." -ForegroundColor Cyan

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
    Write-Host "  The Query().select().where().get_scanner_data() chain may have changed in v3.2.1"
    Write-Host "  Check: data/tv_screener.py line 84 and compare with the library's current README"
}
Remove-Item $tv_script -ErrorAction SilentlyContinue

# ────────────────────────────────────────────────────────────────────────────
# Step 5: LightGBM bootstrap (optional, interactive)
# ────────────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "Step 5/5: LightGBM meta-model bootstrap" -ForegroundColor Cyan
Write-Host "  Pulls 5 years of OHLCV for 10 core symbols, labels each bar"
Write-Host "  via triple-barrier method, trains LightGBM. Takes ~30 minutes."
Write-Host "  Without it: all trades use full size (no P(win) filter)."
Write-Host ""

$answer = Read-Host "  Run LightGBM bootstrap now? (y/N)"
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
Write-Host "Going live checklist:" -ForegroundColor Yellow
Write-Host "  [ ] Rotate API keys (were in git history before sanitization)"
Write-Host "  [ ] Get free OpenBB PAT at https://my.openbb.co -> .env OPENBB_PAT="
Write-Host "  [ ] .env: ALPACA_PAPER=false + live Alpaca API keys"
Write-Host '  [ ] config.json "alpaca" -> "paper": false'
Write-Host "  [ ] After bootstrap: config.json model -> meta_model_enabled: true"
Write-Host "  [ ] Nightly pg_dump backup via Windows Task Scheduler"
Write-Host ""
