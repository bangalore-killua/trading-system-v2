#!/usr/bin/env bash
# Run after first deploy to train all 4 AutoGluon models on the server.
# Models are excluded from git (too large) — must be generated here.
# Expected runtime: ~4 hours total (1h per model).

set -e

echo "=== Training entry model ==="
python -m strategy.training_pipeline --autogluon --years 5
echo "Entry model done. Check models/scorecard_entry_v1.json for IC > 0."

echo "=== Training exit model ==="
python -m strategy.training_pipeline --autogluon --exit-model --years 5
echo "Exit model done."

echo "=== Training dip model ==="
python -m strategy.training_pipeline --autogluon --dip-model --years 5
echo "Dip model done."

echo "=== Training tranche model ==="
python -m strategy.training_pipeline --autogluon --tranche-model --years 5
echo "Tranche model done."

echo "=== All 4 models trained. Starting server... ==="
python run_server.py
