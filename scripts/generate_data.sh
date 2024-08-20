#!/bin/bash

# Generate Data
python generate_data.py --problem tsp --name test --seed 1234 -f

# Eval commands
echo "STARTING UNIF EVAL"
python concorde_baseline.py --data_path data/tsp/tsp_unif50_test_seed1234.pkl
rm *.res
echo "STARTING GMM EVAL"
python concorde_baseline.py --data_path data/tsp/tsp_gmm50_test_seed1234.pkl
rm *.res
echo "STARTING TSPLIB50 EVAL"
python concorde_baseline.py --data_path data/tsp/tsp_tsplib50_test_seed1234.pkl
rm *.res
echo "STARTING DIAG EVAL"
python concorde_baseline.py --data_path data/tsp/tsp_diag50_test_seed1234.pkl
rm *.res

rm *.pul
rm *.sav
rm *.sol