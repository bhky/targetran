#!/usr/bin/env bash
set -e

./install_from_locals.sh
./run_code_checks.sh

python3 ./run_pt_dataset_test.py
python3 ./run_tf_dataset_test.py