@echo off

python -B mean.py train2e-00260 | tee composition.log
python -B mean.py train5-00260 | tee -a composition.log
python -B mean.py train6e-00260 | tee -a composition.log
python -B mean.py train12-00260 | tee -a composition.log
python -B mean.py train18-00260 | tee -a composition.log
python -B mean.py train20e-00260 | tee -a composition.log
python -B mean.py train30-00260 | tee -a composition.log
python -B mean.py train33-00260 | tee -a composition.log
python -B mean.py train33e-00260 | tee -a composition.log
python -B mean.py train36-00260 | tee -a composition.log
python -B mean.py train38-00260 | tee -a composition.log
python -B mean.py train39-00260 | tee -a composition.log
python -B mean.py train42e-00260 | tee -a composition.log
python -B mean.py train44-00260 | tee -a composition.log
python -B mean.py train52e-00260 | tee -a composition.log
python -B mean.py train59e-00260 | tee -a composition.log
python -B mean.py train61e-00260 | tee -a composition.log
python -B mean.py train33ei-00260 | tee -a composition.log
python -B mean.py train32i-00260 | tee -a composition.log
python -B mean.py train36i-00260 | tee -a composition.log
python -B mean.py train56i-00260 | tee -a composition.log
python -B mean.py train38u-00260 | tee -a composition.log
python -B mean.py train38eu-00260 | tee -a composition.log
python -B mean.py train33eu-00260 | tee -a composition.log
python -B mean.py train6eu-00260 | tee -a composition.log
python -B mean.py train6u-00260 | tee -a composition.log
python -B mean.py train59eu-00260 | tee -a composition.log
python -B mean.py train36u-00260 | tee -a composition.log


python -B composition.py | tee -a composition.log

pause
