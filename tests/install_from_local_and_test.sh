#!/usr/bin/env bash
set -e

./tests/install_from_local.sh
./tests/run_code_checks.sh

python3 ./tests/run_pt_dataset_test.py
python3 ./tests/run_tf_dataset_test.py