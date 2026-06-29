# run_overnight.ps1 — Overnight training + backtest runner
# Run once before sleeping: .\run_overnight.ps1
# Results will be in logs\MORNING_SUMMARY.txt

$ErrorActionPreference = "Continue"
$start = Get-Date

if (-not (Test-Path "logs")) { New-Item -ItemType Directory "logs" | Out-Null }

function Log($msg) {
    $ts = (Get-Date).ToString("HH:mm:ss")
    $line = "[$ts] $msg"
    Write-Host $line
    Add-Content -Path "logs\overnight_progress.log" -Value $line
}

Log "=== OVERNIGHT RUN STARTED ==="
Log "Training universe: 147 symbols | 5 years | All 4 models"

# ── Step 1a: Train entry model (all other models depend on this) ───────────────
Log "[1a/4] Training ENTRY model — 147 symbols × 5 years (est. 60-90 min)..."
& venv\Scripts\python.exe -m strategy.training_pipeline `
    --years 5 `
    --autogluon `
    2>&1 | Tee-Object -FilePath logs\train_overnight.log
Log "[1a/4] Entry model done. Exit code: $LASTEXITCODE"

# ── Step 1b: Train exit model ─────────────────────────────────────────────────
Log "[1b/4] Training EXIT model (est. 30-60 min)..."
& venv\Scripts\python.exe -m strategy.training_pipeline `
    --years 5 `
    --autogluon `
    --exit-model `
    2>&1 | Tee-Object -Append -FilePath logs\train_overnight.log
Log "[1b/4] Exit model done. Exit code: $LASTEXITCODE"

# ── Step 1c: Train dip detector ───────────────────────────────────────────────
Log "[1c/4] Training DIP DETECTOR model (est. 30 min)..."
& venv\Scripts\python.exe -m strategy.training_pipeline `
    --years 5 `
    --autogluon `
    --dip-model `
    2>&1 | Tee-Object -Append -FilePath logs\train_overnight.log
Log "[1c/4] Dip model done. Exit code: $LASTEXITCODE"

# ── Step 1d: Train tranche model ──────────────────────────────────────────────
Log "[1d/4] Training TRANCHE model (est. 30 min)..."
& venv\Scripts\python.exe -m strategy.training_pipeline `
    --years 5 `
    --autogluon `
    --tranche-model `
    2>&1 | Tee-Object -Append -FilePath logs\train_overnight.log
Log "[1d/4] Tranche model done. Exit code: $LASTEXITCODE"

# ── Step 2: Full universe backtest ────────────────────────────────────────────
Log "[2/4] Backtest — full universe, 2 years (est. 30-45 min)..."
& venv\Scripts\python.exe -m backtester.ml_backtest `
    --full-universe --years 2 `
    2>&1 | Tee-Object -FilePath logs\backtest_full.log
Log "[2/4] Full universe backtest complete. Exit code: $LASTEXITCODE"

# ── Step 3: Full universe + options ───────────────────────────────────────────
Log "[3/4] Backtest — full universe + options overlay, 2 years (est. 30-45 min)..."
& venv\Scripts\python.exe -m backtester.ml_backtest `
    --full-universe --years 2 --options `
    2>&1 | Tee-Object -FilePath logs\backtest_options.log
Log "[3/4] Options backtest complete. Exit code: $LASTEXITCODE"

# ── Step 4: OOS generalization test ───────────────────────────────────────────
Log "[4/4] Backtest — OOS generalization (28 never-seen symbols), 2 years (est. 20-30 min)..."
& venv\Scripts\python.exe -m backtester.ml_backtest `
    --oos --years 2 `
    2>&1 | Tee-Object -FilePath logs\backtest_oos.log
Log "[4/4] OOS backtest complete. Exit code: $LASTEXITCODE"

# ── Summary ───────────────────────────────────────────────────────────────────
$elapsed = [int]((Get-Date) - $start).TotalMinutes

$summary = @()
$summary += "============================================================"
$summary += "  OVERNIGHT RUN COMPLETE — $(Get-Date -Format 'yyyy-MM-dd HH:mm')"
$summary += "  Total duration: $elapsed minutes"
$summary += "============================================================"
$summary += ""
$summary += "--- ENTRY MODEL (models\scorecard_entry_v1.json) ---"
if (Test-Path "models\scorecard_entry_v1.json") {
    $summary += (Get-Content "models\scorecard_entry_v1.json")
} else {
    $summary += "  WARNING: scorecard file not found — check train log"
}
$summary += ""
$summary += "--- FULL UNIVERSE BACKTEST (2yr) ---"
$fullLines = Select-String -Path "logs\backtest_full.log" -Pattern "Return|Sharpe|Win|Hold|Trades|Alpha|MaxDD" -ErrorAction SilentlyContinue
if ($fullLines) { $summary += ($fullLines | Select-Object -Last 10 | ForEach-Object { "  " + $_.Line.Trim() }) }
else { $summary += "  (no results found — check logs\backtest_full.log)" }
$summary += ""
$summary += "--- FULL UNIVERSE + OPTIONS (2yr) ---"
$optLines = Select-String -Path "logs\backtest_options.log" -Pattern "Return|Sharpe|Win|Hold|Trades|Alpha|MaxDD" -ErrorAction SilentlyContinue
if ($optLines) { $summary += ($optLines | Select-Object -Last 10 | ForEach-Object { "  " + $_.Line.Trim() }) }
else { $summary += "  (no results found — check logs\backtest_options.log)" }
$summary += ""
$summary += "--- OOS GENERALIZATION TEST (28 unseen symbols, 2yr) ---"
$oosLines = Select-String -Path "logs\backtest_oos.log" -Pattern "Return|Sharpe|Win|Hold|Trades|Alpha|MaxDD" -ErrorAction SilentlyContinue
if ($oosLines) { $summary += ($oosLines | Select-Object -Last 10 | ForEach-Object { "  " + $_.Line.Trim() }) }
else { $summary += "  (no results found — check logs\backtest_oos.log)" }
$summary += ""
$summary += "--- MODEL FILES ---"
$summary += (Get-ChildItem "models\autogluon_*" -Directory -ErrorAction SilentlyContinue | ForEach-Object {
    $files = (Get-ChildItem $_.FullName -Recurse -File -ErrorAction SilentlyContinue).Count
    "  $($_.Name): $files files"
})
$summary += ""
$summary += "--- NEXT STEPS ---"
$summary += "  1. Review OOS Sharpe — if > 0.5, model generalises well"
$summary += "  2. If OOS negative, model overfit — reduce training years to 3"
$summary += "  3. When ready for MiniCPM: pip install peft accelerate && ollama pull minicpm"
$summary += "     then: python scripts\finetune_minicpm_sentiment.py"
$summary += "============================================================"

$summary | Tee-Object -FilePath "logs\MORNING_SUMMARY.txt"
Log "=== Done. Read logs\MORNING_SUMMARY.txt for results ==="
