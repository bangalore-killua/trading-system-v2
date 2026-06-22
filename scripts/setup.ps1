# setup.ps1 — One-shot setup for trading-system-v2
#
# Run this from the repo root AFTER the main requirements.txt install:
#   cd C:\Users\ASUS\git_repos\trading-system-v2
#   .\scripts\setup.ps1
#
# What it does (in order):
#   1. Install pandas-ta (pre-release beta — excluded from requirements.txt
#      because pip's resolver runs out of memory on the full dependency tree)
#   2. Pre-download FinBERT model from HuggingFace (~400 MB, cached locally)
#   3. Verify spaCy en_core_web_sm is installed
#   4. Verify tradingview-screener API compatibility with v3.2.1
#   5. Bootstrap LightGBM: pull 5yr OHLCV and label it (takes ~30 min)
#
# Going LIVE checklist (one-time, manual steps not in this script):
#   [ ] Rotate API keys — Alpaca, Telegram bot, FMP, NewsAPI were in git history
#   [ ] Get OpenBB PAT at https://my.openbb.co (free) — add to .env OPENBB_PAT=
#   [ ] Change .env: ALPACA_PAPER=false + use live Alpaca API keys
#   [ ] Change config.json: "alpaca": { "paper": false }
#   [ ] Apply for Alpaca options trading (Level 2) if options.enabled=true
#   [ ] Set up nightly PostgreSQL backup (Task Scheduler + pg_dump)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot

Write-Host ""
Write-Host "=== trading-system-v2 Setup ===" -ForegroundColor Cyan
Write-Host "Repo: $repo"
Write-Host ""

# ── Activate venv ─────────────────────────────────────────────────────────────
$venv = "$repo\venv\Scripts\Activate.ps1"
if (Test-Path $venv) {
    . $venv
    Write-Host "[OK] venv activated" -ForegroundColor Green
} else {
    Write-Host "[WARN] venv not found at $venv — using system Python" -ForegroundColor Yellow
}

Set-Location $repo

# ── Step 1: pandas-ta ─────────────────────────────────────────────────────────
Write-Host ""
Write-Host "Step 1/5: Installing pandas-ta 0.4.71b0..." -ForegroundColor Cyan
try {
    uv pip install "pandas-ta==0.4.71b0" --prerelease=allow
    Write-Host "[OK] pandas-ta installed" -ForegroundColor Green
} catch {
    Write-Host "[FAIL] pandas-ta install failed: $_" -ForegroundColor Red
    Write-Host "  Try manually: uv pip install 'pandas-ta==0.4.71b0' --prerelease=allow"
}

# ── Step 2: FinBERT pre-download ──────────────────────────────────────────────
Write-Host ""
Write-Host "Step 2/5: Pre-downloading FinBERT model (~400 MB)..." -ForegroundColor Cyan
Write-Host "         This prevents a 2-4 min delay on the first live news event."
try {
    python -c @"
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
model_name = 'yiyanghkust/finbert-tone'
print(f'  Downloading {model_name} ...')
AutoTokenizer.from_pretrained(model_name)
AutoModelForSequenceClassification.from_pretrained(model_name)
print('  FinBERT download complete.')
"@
    Write-Host "[OK] FinBERT cached" -ForegroundColor Green
} catch {
    Write-Host "[FAIL] FinBERT download failed: $_" -ForegroundColor Red
    Write-Host "  Will auto-download on first use (but that first news trade will be slow)."
}

# ── Step 3: spaCy model ───────────────────────────────────────────────────────
Write-Host ""
Write-Host "Step 3/5: Verifying spaCy en_core_web_sm..." -ForegroundColor Cyan
try {
    python -c "import spacy; spacy.load('en_core_web_sm'); print('  en_core_web_sm OK')"
    Write-Host "[OK] spaCy model present" -ForegroundColor Green
} catch {
    Write-Host "  Not installed — downloading now..."
    python -m spacy download en_core_web_sm
    Write-Host "[OK] spaCy model installed" -ForegroundColor Green
}

# ── Step 4: tradingview-screener API smoke test ────────────────────────────────
Write-Host ""
Write-Host "Step 4/5: Testing tradingview-screener v3.2.1 API..." -ForegroundColor Cyan
try {
    python -c @"
from tradingview_screener import Query, Column
print('  Running test scan (limit=3)...')
result = Query().select('name', 'close').limit(3).get_scanner_data()
count, df = result if isinstance(result, tuple) else (0, result)
print(f'  Got {len(df)} rows — API compatible')
"@
    Write-Host "[OK] tradingview-screener API working" -ForegroundColor Green
} catch {
    Write-Host "[FAIL] tradingview-screener API test failed: $_" -ForegroundColor Red
    Write-Host "  The tv_screener.py code may need updating for v3.2.1."
    Write-Host "  Check: data/tv_screener.py:84 — Query().select().where().get_scanner_data()"
}

# ── Step 5: LightGBM bootstrap ────────────────────────────────────────────────
Write-Host ""
Write-Host "Step 5/5: LightGBM meta-model bootstrap" -ForegroundColor Cyan
Write-Host "  This pulls 5 years of OHLCV for 10 core symbols and labels them."
Write-Host "  Takes approximately 25-40 minutes. You can skip and run later."
Write-Host ""
$answer = Read-Host "  Run LightGBM bootstrap now? (y/N)"
if ($answer -match "^[Yy]$") {
    Write-Host "  Starting bootstrap — do not close this window..." -ForegroundColor Yellow
    python -m strategy.training_pipeline `
        --symbols AAPL MSFT NVDA GOOGL META AMZN TSLA AMD CRM NFLX `
        --years 5
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] LightGBM bootstrap complete — model saved to models/" -ForegroundColor Green
        Write-Host "  Now set config.json: `"meta_model_enabled`": true"
    } else {
        Write-Host "[FAIL] Bootstrap exited with error — check logs/system.log" -ForegroundColor Red
    }
} else {
    Write-Host "  Skipped. Run later with:" -ForegroundColor Yellow
    Write-Host "    python -m strategy.training_pipeline --symbols AAPL MSFT NVDA GOOGL META AMZN TSLA AMD CRM NFLX --years 5"
    Write-Host "  Then set config.json: `"meta_model_enabled`": true"
}

# ── Summary ───────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "=== Setup Complete ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "Start paper trading:" -ForegroundColor White
Write-Host "  python run_server.py"
Write-Host ""
Write-Host "Going live checklist:" -ForegroundColor White
Write-Host "  [ ] Rotate API keys (they were in git history)"
Write-Host "  [ ] Get free OpenBB PAT at https://my.openbb.co -> add to .env"
Write-Host "  [ ] .env: ALPACA_PAPER=false + live Alpaca keys"
Write-Host "  [ ] config.json: `"alpaca`": { `"paper`": false }"
Write-Host "  [ ] After bootstrap: config.json: `"meta_model_enabled`": true"
Write-Host "  [ ] Set up nightly pg_dump backup via Task Scheduler"
Write-Host ""
