#!/usr/bin/env bash
# Run after first deploy to train all 3 AutoGluon models on the server.
# Models are excluded from git (too large) — must be generated here.
# Expected runtime: ~3 hours total (1h per model).

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

echo "=== All models trained. Starting server... ==="
python run_server.py
