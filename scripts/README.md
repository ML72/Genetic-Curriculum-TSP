# Training and Evaluation Scripts

This folder contains all exact scripts used for data generation, training and evaluation, as well as plot generation and statistical testing.

Steps for reproducing results:
1. Unzip `/data/tsplib.zip` (at the root directory of this project) and place all contained files inside a new folder `/data/tsplib/`.
2. Run commands in `generate_data.sh` to generate data and run the oracle solver on them.
3. Run commands in `train.sh` to train models.
4. Run commands in `eval.sh` to run model evaluations.
5. Generate plots with the Python files under `plots/`. Remember to modify constants defined at the top of the file as necessary, as the same file may be used to generate multiple plots. Resulting plots will appear in `/results/plots/`.
6. Conduct statistical tests with the Python files under `tests/`. Remember to modify constants defined at the top of the file as necessary.
