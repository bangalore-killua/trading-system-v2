# train_yolov8.ps1 - Download chart pattern dataset from Roboflow and fine-tune YOLOv8n
#
# Run from repo root:
#   cd C:\Users\ASUS\git_repos\trading-system-v2
#   .\scripts\train_yolov8.ps1
#
# Pre-requisites:
#   - venv active with ultralytics installed (done by requirements.txt)
#   - Free Roboflow account at https://roboflow.com
#     (search "stock chart pattern" -> Universe -> pick a public dataset)
#
# After training:
#   - Weights are saved to models/chart_patterns_yolov8n/weights/best.pt
#   - Script copies best.pt to models/chart_patterns_yolov8n.pt
#   - config.json model.chart_pattern_model_path already points there
#
# Expected time: ~20 min on CPU (RTX 3060 ~4 min)

$ErrorActionPreference = "Continue"
$repo = Split-Path -Parent $PSScriptRoot

# Activate venv
$venv = "$repo\venv\Scripts\Activate.ps1"
if (Test-Path $venv) { . $venv }

Set-Location $repo

Write-Host ""
Write-Host "=== YOLOv8 Chart Pattern Training ===" -ForegroundColor Cyan
Write-Host ""

# ────────────────────────────────────────────────────────────────────────────
# Check: model already exists?
# ────────────────────────────────────────────────────────────────────────────
$model_path = "$repo\models\chart_patterns_yolov8n.pt"
if (Test-Path $model_path) {
    Write-Host "Model already exists at models\chart_patterns_yolov8n.pt" -ForegroundColor Green
    try { $retrain = Read-Host "Retrain anyway? (y/N)" } catch { $retrain = "N" }
    if ($retrain -notmatch "^[Yy]$") {
        Write-Host "Exiting - model is ready."
        exit 0
    }
}

# ────────────────────────────────────────────────────────────────────────────
# Check: dataset already downloaded?
# ────────────────────────────────────────────────────────────────────────────
$dataset_yaml = "$repo\datasets\chart_patterns\data.yaml"

if (-not (Test-Path $dataset_yaml)) {
    Write-Host "No dataset found at datasets\chart_patterns\data.yaml" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "OPTION A - Roboflow automatic download (recommended):" -ForegroundColor White
    Write-Host "  1. Sign up free at https://roboflow.com"
    Write-Host "  2. Go to https://universe.roboflow.com/search?q=stock+chart+patterns"
    Write-Host "  3. Pick any dataset with 500+ images (e.g. 'Stock Chart Patterns' by Roboflow)"
    Write-Host "  4. Click Download -> YOLOv8 format -> 'Show download code'"
    Write-Host "  5. Copy the API key from the download snippet"
    Write-Host ""
    Write-Host "OPTION B - Manual download:" -ForegroundColor White
    Write-Host "  1. Download a dataset in YOLOv8 format from Roboflow Universe"
    Write-Host "  2. Unzip it so datasets\chart_patterns\data.yaml exists"
    Write-Host "  3. Re-run this script"
    Write-Host ""

    try { $api_key = Read-Host "Enter your Roboflow API key (or press Enter to exit)" } catch { $api_key = "" }

    if ([string]::IsNullOrWhiteSpace($api_key)) {
        Write-Host "No key provided. Download the dataset manually and re-run." -ForegroundColor Yellow
        Write-Host "  Expected layout:"
        Write-Host "    datasets/chart_patterns/data.yaml"
        Write-Host "    datasets/chart_patterns/train/images/*.jpg"
        Write-Host "    datasets/chart_patterns/train/labels/*.txt"
        Write-Host "    datasets/chart_patterns/valid/images/*.jpg"
        Write-Host "    datasets/chart_patterns/valid/labels/*.txt"
        exit 0
    }

    Write-Host ""
    Write-Host "Installing roboflow Python SDK..." -ForegroundColor Cyan
    uv pip install roboflow 2>$null

    $rf_script = "$repo\scripts\_tmp_rf_download.py"
    Set-Content -Path $rf_script -Encoding utf8 -Value @"
import sys, os
api_key = "$api_key"

try:
    from roboflow import Roboflow
except ImportError:
    print("ERROR: roboflow package not installed")
    sys.exit(1)

rf = Roboflow(api_key=api_key)

# Try known public dataset first; if access denied, print instructions
try:
    project = rf.workspace("roboflow-100").project("stock-market-charts")
    version = project.version(1)
    dataset = version.download("yolov8", location="datasets/chart_patterns", overwrite=True)
    print("Dataset downloaded to datasets/chart_patterns/")
except Exception as e:
    print(f"Could not auto-download ({{e}})")
    print("")
    print("Manual download steps:")
    print("  1. Visit https://universe.roboflow.com/search?q=stock+chart+patterns")
    print("  2. Pick a dataset with YOLOv8 export support")
    print("  3. Click Download -> YOLOv8 format -> save to datasets/chart_patterns/")
    sys.exit(1)
"@

    python $rf_script
    Remove-Item $rf_script -ErrorAction SilentlyContinue

    if (-not (Test-Path $dataset_yaml)) {
        Write-Host "[FAIL] Dataset not found after download attempt. Download manually." -ForegroundColor Red
        exit 1
    }
}

Write-Host "[OK] Dataset found: $dataset_yaml" -ForegroundColor Green

# ────────────────────────────────────────────────────────────────────────────
# Read dataset info
# ────────────────────────────────────────────────────────────────────────────
$info_script = "$repo\scripts\_tmp_dataset_info.py"
Set-Content -Path $info_script -Encoding utf8 -Value @'
import yaml, os
from pathlib import Path

yaml_path = Path("datasets/chart_patterns/data.yaml")
with open(yaml_path) as f:
    d = yaml.safe_load(f)

nc = d.get("nc", "?")
names = d.get("names", [])
train_dir = Path("datasets/chart_patterns") / d.get("train", "train/images")
n_train = len(list(train_dir.glob("*.jpg")) + list(train_dir.glob("*.png"))) if train_dir.exists() else "?"
print(f"Classes: {nc} | Names: {names[:5]}... | Train images: {n_train}")
'@

python $info_script
Remove-Item $info_script -ErrorAction SilentlyContinue

# ────────────────────────────────────────────────────────────────────────────
# Training
# ────────────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "Starting YOLOv8n fine-tuning..." -ForegroundColor Cyan
Write-Host "  Base model: yolov8n.pt (COCO pretrained)"
Write-Host "  Output: models/chart_patterns_yolov8n/"
Write-Host "  Epochs: 50 (reduce with --epochs N if too slow)"
Write-Host ""

$epochs = 50
foreach ($arg in $args) {
    if ($arg -match "--epochs=(\d+)") { $epochs = $Matches[1] }
    if ($arg -match "^--epochs$") {
        $idx = [array]::IndexOf($args, $arg)
        if ($idx -lt $args.Length - 1) { $epochs = $args[$idx + 1] }
    }
}

$train_script = "$repo\scripts\_tmp_yolo_train.py"
Set-Content -Path $train_script -Encoding utf8 -Value @"
import sys
from pathlib import Path

if __name__ == '__main__':
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed (check requirements.txt)")
        sys.exit(1)

    model = YOLO("yolov8n.pt")
    model.train(
        data="datasets/chart_patterns/data.yaml",
        epochs=$epochs,
        imgsz=640,
        batch=16,
        project="models",
        name="chart_patterns_yolov8n",
        exist_ok=True,
        patience=15,
        cache=True,
        device=0 if __import__('torch').cuda.is_available() else 'cpu',
    )

    best = Path("models/chart_patterns_yolov8n/weights/best.pt")
    dest = Path("models/chart_patterns_yolov8n.pt")
    if best.exists():
        import shutil
        shutil.copy2(best, dest)
        print(f"Best weights copied to {dest}")
    else:
        print("WARNING: best.pt not found after training")
        sys.exit(1)
"@

python $train_script $args
$exit_code = $LASTEXITCODE
Remove-Item $train_script -ErrorAction SilentlyContinue

if ($exit_code -eq 0 -and (Test-Path $model_path)) {
    Write-Host ""
    Write-Host "[OK] Training complete!" -ForegroundColor Green
    Write-Host "  Model: $model_path"
    Write-Host "  The model is automatically used when scoring (chart_* features in LightGBM)"
    Write-Host ""
    Write-Host "NEXT: retrain LightGBM to capture the 15 chart pattern features"
    Write-Host "  python -m strategy.training_pipeline --symbols AAPL MSFT NVDA GOOGL META AMZN TSLA AMD CRM NFLX --years 5"
} else {
    Write-Host ""
    Write-Host "[FAIL] Training failed or model not saved" -ForegroundColor Red
    Write-Host "  Check the output above for errors"
    Write-Host "  Common issues:"
    Write-Host "    - Dataset labels don't match the 15 pattern names in chart_pattern_detector.py"
    Write-Host "    - Out of GPU/CPU memory: try smaller --epochs or batch=8"
    exit 1
}
