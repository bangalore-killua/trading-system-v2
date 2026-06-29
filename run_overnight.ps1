# run_overnight.ps1 - Overnight training + backtest runner
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
Log "Training universe: 147 symbols, 5 years, all 4 models"

# Step 1a: Entry model (all other models depend on this completing first)
Log "[1a] Training ENTRY model - 147 symbols x 5 years (est. 60-90 min)..."
& venv\Scripts\python.exe -m strategy.training_pipeline `
    --years 5 `
    --autogluon `
    2>&1 | Tee-Object -FilePath logs\train_overnight.log
Log "[1a] Entry model done. Exit code: $LASTEXITCODE"

# Step 1b: Exit model
Log "[1b] Training EXIT model (est. 30-60 min)..."
& venv\Scripts\python.exe -m strategy.training_pipeline `
    --years 5 `
    --autogluon `
    --exit-model `
    2>&1 | Tee-Object -Append -FilePath logs\train_overnight.log
Log "[1b] Exit model done. Exit code: $LASTEXITCODE"

# Step 1c: Dip detector
Log "[1c] Training DIP DETECTOR model (est. 30 min)..."
& venv\Scripts\python.exe -m strategy.training_pipeline `
    --years 5 `
    --autogluon `
    --dip-model `
    2>&1 | Tee-Object -Append -FilePath logs\train_overnight.log
Log "[1c] Dip model done. Exit code: $LASTEXITCODE"

# Step 1d: Tranche model
Log "[1d] Training TRANCHE model (est. 30 min)..."
& venv\Scripts\python.exe -m strategy.training_pipeline `
    --years 5 `
    --autogluon `
    --tranche-model `
    2>&1 | Tee-Object -Append -FilePath logs\train_overnight.log
Log "[1d] Tranche model done. Exit code: $LASTEXITCODE"

# Step 2: Full universe backtest
Log "[2] Backtest - full universe, 2 years (est. 30-45 min)..."
& venv\Scripts\python.exe -m backtester.ml_backtest `
    --full-universe --years 2 `
    2>&1 | Tee-Object -FilePath logs\backtest_full.log
Log "[2] Full universe backtest done. Exit code: $LASTEXITCODE"

# Step 3: Full universe + options
Log "[3] Backtest - full universe + options overlay, 2 years (est. 30-45 min)..."
& venv\Scripts\python.exe -m backtester.ml_backtest `
    --full-universe --years 2 --options `
    2>&1 | Tee-Object -FilePath logs\backtest_options.log
Log "[3] Options backtest done. Exit code: $LASTEXITCODE"

# Step 4: OOS generalization test
Log "[4] Backtest - OOS 28 unseen symbols, 2 years (est. 20-30 min)..."
& venv\Scripts\python.exe -m backtester.ml_backtest `
    --oos --years 2 `
    2>&1 | Tee-Object -FilePath logs\backtest_oos.log
Log "[4] OOS backtest done. Exit code: $LASTEXITCODE"

# Build morning summary
$elapsed = [int]((Get-Date) - $start).TotalMinutes
$ts = (Get-Date).ToString("yyyy-MM-dd HH:mm")

$lines = @(
    "============================================================",
    "  OVERNIGHT RUN COMPLETE - $ts",
    "  Total duration: $elapsed minutes",
    "============================================================",
    "",
    "--- ENTRY MODEL ---"
)

if (Test-Path "models\scorecard_entry_v1.json") {
    $lines += (Get-Content "models\scorecard_entry_v1.json")
} else {
    $lines += "  WARNING: scorecard file not found - check logs\train_overnight.log"
}

$lines += ""
$lines += "--- FULL UNIVERSE BACKTEST (2yr) ---"
$fullLines = Select-String -Path "logs\backtest_full.log" -Pattern "Return|Sharpe|Win|Hold|Trades|Alpha|MaxDD" -ErrorAction SilentlyContinue
if ($fullLines) {
    $lines += ($fullLines | Select-Object -Last 10 | ForEach-Object { "  " + $_.Line.Trim() })
} else {
    $lines += "  no results found - check logs\backtest_full.log"
}

$lines += ""
$lines += "--- FULL UNIVERSE + OPTIONS (2yr) ---"
$optLines = Select-String -Path "logs\backtest_options.log" -Pattern "Return|Sharpe|Win|Hold|Trades|Alpha|MaxDD" -ErrorAction SilentlyContinue
if ($optLines) {
    $lines += ($optLines | Select-Object -Last 10 | ForEach-Object { "  " + $_.Line.Trim() })
} else {
    $lines += "  no results found - check logs\backtest_options.log"
}

$lines += ""
$lines += "--- OOS GENERALIZATION TEST (28 unseen symbols, 2yr) ---"
$oosLines = Select-String -Path "logs\backtest_oos.log" -Pattern "Return|Sharpe|Win|Hold|Trades|Alpha|MaxDD" -ErrorAction SilentlyContinue
if ($oosLines) {
    $lines += ($oosLines | Select-Object -Last 10 | ForEach-Object { "  " + $_.Line.Trim() })
} else {
    $lines += "  no results found - check logs\backtest_oos.log"
}

$lines += ""
$lines += "--- MODEL FILES ---"
$modelDirs = Get-ChildItem "models\autogluon_*" -Directory -ErrorAction SilentlyContinue
foreach ($d in $modelDirs) {
    $count = (Get-ChildItem $d.FullName -Recurse -File -ErrorAction SilentlyContinue).Count
    $lines += "  $($d.Name): $count files"
}

$lines += ""
$lines += "--- NEXT STEPS ---"
$lines += "  1. OOS Sharpe > 0.5 = model generalises well"
$lines += "  2. OOS negative = overfit, retrain with --years 3"
$lines += "  3. MiniCPM when ready: pip install peft accelerate"
$lines += "     ollama pull minicpm"
$lines += "     python scripts\finetune_minicpm_sentiment.py"
$lines += "============================================================"

$lines | Out-File -FilePath "logs\MORNING_SUMMARY.txt" -Encoding utf8
$lines | Write-Host
Log "=== Done. Read logs\MORNING_SUMMARY.txt for results ==="
