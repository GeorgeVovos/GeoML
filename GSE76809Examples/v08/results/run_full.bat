@echo off
cd /d C:\dev\GeoML\GSE76809Examples\v08
set PYTHONIOENCODING=utf-8
python -u sample_efficiency.py --skip-quantum-kernel --models quantum_vqc classical_mlp classical_svm classical_xgb classical_logreg --subsample-seeds 20 --vqc-init-seeds 3 --n-per-class 10 20 > results\v08_full_run.txt 2>&1
