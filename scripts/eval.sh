#!/bin/bash

# Eval commands for preliminary study plots
python eval.py data/tsp/tsp_tsplib50_test_seed1234.pkl --model outputs/tsp_50/hac_fullbatch_1 --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_tsplib50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsplib --load_tsplib --model outputs/tsp_50/hac_fullbatch_1 --decode_strategy greedy --eval_batch_size 1000 -f

# Eval commands for Unif distribution
python eval.py data/tsp/tsp_unif50_test_seed1234.pkl --model outputs/tsp_50/curriculum_ewc_genome_1 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_unif50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_unif50_test_seed1234.pkl --model outputs/tsp_50/curriculum_ewc_genome_2 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_unif50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_unif50_test_seed1234.pkl --model outputs/tsp_50/curriculum_ewc_genome_3 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_unif50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_unif50_test_seed1234.pkl --model outputs/tsp_50/curriculum_ewc_genome_4 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_unif50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_unif50_test_seed1234.pkl --model outputs/tsp_50/curriculum_ewc_genome_5 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_unif50_test_seed1234/concorde_costs.pkl -f

python eval.py data/tsp/tsp_unif50_test_seed1234.pkl --model outputs/tsp_50/baseline_unif_1 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_unif50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_unif50_test_seed1234.pkl --model outputs/tsp_50/baseline_unif_2 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_unif50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_unif50_test_seed1234.pkl --model outputs/tsp_50/baseline_unif_3 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_unif50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_unif50_test_seed1234.pkl --model outputs/tsp_50/baseline_unif_4 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_unif50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_unif50_test_seed1234.pkl --model outputs/tsp_50/baseline_unif_5 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_unif50_test_seed1234/concorde_costs.pkl -f

python eval.py data/tsp/tsp_unif50_test_seed1234.pkl --model outputs/tsp_50/hac_fullbatch_1 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_unif50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_unif50_test_seed1234.pkl --model outputs/tsp_50/hac_fullbatch_2 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_unif50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_unif50_test_seed1234.pkl --model outputs/tsp_50/hac_fullbatch_3 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_unif50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_unif50_test_seed1234.pkl --model outputs/tsp_50/hac_fullbatch_4 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_unif50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_unif50_test_seed1234.pkl --model outputs/tsp_50/hac_fullbatch_5 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_unif50_test_seed1234/concorde_costs.pkl -f

python eval.py data/tsp/tsp_unif50_test_seed1234.pkl --model outputs/tsp_50/ablation_noewc_1 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_unif50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_unif50_test_seed1234.pkl --model outputs/tsp_50/ablation_noewc_2 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_unif50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_unif50_test_seed1234.pkl --model outputs/tsp_50/ablation_noewc_3 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_unif50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_unif50_test_seed1234.pkl --model outputs/tsp_50/ablation_noewc_4 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_unif50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_unif50_test_seed1234.pkl --model outputs/tsp_50/ablation_noewc_5 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_unif50_test_seed1234/concorde_costs.pkl -f

python eval.py data/tsp/tsp_unif50_test_seed1234.pkl --model outputs/tsp_50/ablation_nogenome_1 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_unif50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_unif50_test_seed1234.pkl --model outputs/tsp_50/ablation_nogenome_2 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_unif50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_unif50_test_seed1234.pkl --model outputs/tsp_50/ablation_nogenome_3 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_unif50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_unif50_test_seed1234.pkl --model outputs/tsp_50/ablation_nogenome_4 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_unif50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_unif50_test_seed1234.pkl --model outputs/tsp_50/ablation_nogenome_5 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_unif50_test_seed1234/concorde_costs.pkl -f

# Eval commands for Gaussian Mixture distribution
python eval.py data/tsp/tsp_gmm50_test_seed1234.pkl --model outputs/tsp_50/curriculum_ewc_genome_1 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_gmm50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_gmm50_test_seed1234.pkl --model outputs/tsp_50/curriculum_ewc_genome_2 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_gmm50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_gmm50_test_seed1234.pkl --model outputs/tsp_50/curriculum_ewc_genome_3 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_gmm50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_gmm50_test_seed1234.pkl --model outputs/tsp_50/curriculum_ewc_genome_4 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_gmm50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_gmm50_test_seed1234.pkl --model outputs/tsp_50/curriculum_ewc_genome_5 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_gmm50_test_seed1234/concorde_costs.pkl -f

python eval.py data/tsp/tsp_gmm50_test_seed1234.pkl --model outputs/tsp_50/baseline_unif_1 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_gmm50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_gmm50_test_seed1234.pkl --model outputs/tsp_50/baseline_unif_2 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_gmm50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_gmm50_test_seed1234.pkl --model outputs/tsp_50/baseline_unif_3 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_gmm50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_gmm50_test_seed1234.pkl --model outputs/tsp_50/baseline_unif_4 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_gmm50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_gmm50_test_seed1234.pkl --model outputs/tsp_50/baseline_unif_5 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_gmm50_test_seed1234/concorde_costs.pkl -f

python eval.py data/tsp/tsp_gmm50_test_seed1234.pkl --model outputs/tsp_50/hac_fullbatch_1 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_gmm50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_gmm50_test_seed1234.pkl --model outputs/tsp_50/hac_fullbatch_2 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_gmm50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_gmm50_test_seed1234.pkl --model outputs/tsp_50/hac_fullbatch_3 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_gmm50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_gmm50_test_seed1234.pkl --model outputs/tsp_50/hac_fullbatch_4 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_gmm50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_gmm50_test_seed1234.pkl --model outputs/tsp_50/hac_fullbatch_5 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_gmm50_test_seed1234/concorde_costs.pkl -f

python eval.py data/tsp/tsp_gmm50_test_seed1234.pkl --model outputs/tsp_50/ablation_noewc_1 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_gmm50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_gmm50_test_seed1234.pkl --model outputs/tsp_50/ablation_noewc_2 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_gmm50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_gmm50_test_seed1234.pkl --model outputs/tsp_50/ablation_noewc_3 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_gmm50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_gmm50_test_seed1234.pkl --model outputs/tsp_50/ablation_noewc_4 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_gmm50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_gmm50_test_seed1234.pkl --model outputs/tsp_50/ablation_noewc_5 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_gmm50_test_seed1234/concorde_costs.pkl -f

python eval.py data/tsp/tsp_gmm50_test_seed1234.pkl --model outputs/tsp_50/ablation_nogenome_1 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_gmm50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_gmm50_test_seed1234.pkl --model outputs/tsp_50/ablation_nogenome_2 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_gmm50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_gmm50_test_seed1234.pkl --model outputs/tsp_50/ablation_nogenome_3 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_gmm50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_gmm50_test_seed1234.pkl --model outputs/tsp_50/ablation_nogenome_4 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_gmm50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_gmm50_test_seed1234.pkl --model outputs/tsp_50/ablation_nogenome_5 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_gmm50_test_seed1234/concorde_costs.pkl -f

# Eval commands for TSPLib50 distribution
python eval.py data/tsp/tsp_tsplib50_test_seed1234.pkl --model outputs/tsp_50/curriculum_ewc_genome_1 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_tsplib50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_tsplib50_test_seed1234.pkl --model outputs/tsp_50/curriculum_ewc_genome_2 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_tsplib50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_tsplib50_test_seed1234.pkl --model outputs/tsp_50/curriculum_ewc_genome_3 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_tsplib50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_tsplib50_test_seed1234.pkl --model outputs/tsp_50/curriculum_ewc_genome_4 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_tsplib50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_tsplib50_test_seed1234.pkl --model outputs/tsp_50/curriculum_ewc_genome_5 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_tsplib50_test_seed1234/concorde_costs.pkl -f

python eval.py data/tsp/tsp_tsplib50_test_seed1234.pkl --model outputs/tsp_50/baseline_unif_1 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_tsplib50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_tsplib50_test_seed1234.pkl --model outputs/tsp_50/baseline_unif_2 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_tsplib50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_tsplib50_test_seed1234.pkl --model outputs/tsp_50/baseline_unif_3 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_tsplib50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_tsplib50_test_seed1234.pkl --model outputs/tsp_50/baseline_unif_4 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_tsplib50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_tsplib50_test_seed1234.pkl --model outputs/tsp_50/baseline_unif_5 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_tsplib50_test_seed1234/concorde_costs.pkl -f

python eval.py data/tsp/tsp_tsplib50_test_seed1234.pkl --model outputs/tsp_50/hac_fullbatch_1 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_tsplib50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_tsplib50_test_seed1234.pkl --model outputs/tsp_50/hac_fullbatch_2 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_tsplib50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_tsplib50_test_seed1234.pkl --model outputs/tsp_50/hac_fullbatch_3 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_tsplib50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_tsplib50_test_seed1234.pkl --model outputs/tsp_50/hac_fullbatch_4 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_tsplib50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_tsplib50_test_seed1234.pkl --model outputs/tsp_50/hac_fullbatch_5 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_tsplib50_test_seed1234/concorde_costs.pkl -f

python eval.py data/tsp/tsp_tsplib50_test_seed1234.pkl --model outputs/tsp_50/ablation_noewc_1 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_tsplib50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_tsplib50_test_seed1234.pkl --model outputs/tsp_50/ablation_noewc_2 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_tsplib50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_tsplib50_test_seed1234.pkl --model outputs/tsp_50/ablation_noewc_3 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_tsplib50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_tsplib50_test_seed1234.pkl --model outputs/tsp_50/ablation_noewc_4 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_tsplib50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_tsplib50_test_seed1234.pkl --model outputs/tsp_50/ablation_noewc_5 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_tsplib50_test_seed1234/concorde_costs.pkl -f

python eval.py data/tsp/tsp_tsplib50_test_seed1234.pkl --model outputs/tsp_50/ablation_nogenome_1 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_tsplib50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_tsplib50_test_seed1234.pkl --model outputs/tsp_50/ablation_nogenome_2 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_tsplib50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_tsplib50_test_seed1234.pkl --model outputs/tsp_50/ablation_nogenome_3 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_tsplib50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_tsplib50_test_seed1234.pkl --model outputs/tsp_50/ablation_nogenome_4 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_tsplib50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_tsplib50_test_seed1234.pkl --model outputs/tsp_50/ablation_nogenome_5 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_tsplib50_test_seed1234/concorde_costs.pkl -f

# Eval commands for Diag distribution
python eval.py data/tsp/tsp_diag50_test_seed1234.pkl --model outputs/tsp_50/curriculum_ewc_genome_1 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_diag50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_diag50_test_seed1234.pkl --model outputs/tsp_50/curriculum_ewc_genome_2 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_diag50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_diag50_test_seed1234.pkl --model outputs/tsp_50/curriculum_ewc_genome_3 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_diag50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_diag50_test_seed1234.pkl --model outputs/tsp_50/curriculum_ewc_genome_4 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_diag50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_diag50_test_seed1234.pkl --model outputs/tsp_50/curriculum_ewc_genome_5 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_diag50_test_seed1234/concorde_costs.pkl -f

python eval.py data/tsp/tsp_diag50_test_seed1234.pkl --model outputs/tsp_50/baseline_unif_1 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_diag50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_diag50_test_seed1234.pkl --model outputs/tsp_50/baseline_unif_2 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_diag50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_diag50_test_seed1234.pkl --model outputs/tsp_50/baseline_unif_3 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_diag50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_diag50_test_seed1234.pkl --model outputs/tsp_50/baseline_unif_4 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_diag50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_diag50_test_seed1234.pkl --model outputs/tsp_50/baseline_unif_5 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_diag50_test_seed1234/concorde_costs.pkl -f

python eval.py data/tsp/tsp_diag50_test_seed1234.pkl --model outputs/tsp_50/hac_fullbatch_1 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_diag50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_diag50_test_seed1234.pkl --model outputs/tsp_50/hac_fullbatch_2 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_diag50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_diag50_test_seed1234.pkl --model outputs/tsp_50/hac_fullbatch_3 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_diag50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_diag50_test_seed1234.pkl --model outputs/tsp_50/hac_fullbatch_4 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_diag50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_diag50_test_seed1234.pkl --model outputs/tsp_50/hac_fullbatch_5 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_diag50_test_seed1234/concorde_costs.pkl -f

python eval.py data/tsp/tsp_diag50_test_seed1234.pkl --model outputs/tsp_50/ablation_noewc_1 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_diag50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_diag50_test_seed1234.pkl --model outputs/tsp_50/ablation_noewc_2 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_diag50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_diag50_test_seed1234.pkl --model outputs/tsp_50/ablation_noewc_3 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_diag50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_diag50_test_seed1234.pkl --model outputs/tsp_50/ablation_noewc_4 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_diag50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_diag50_test_seed1234.pkl --model outputs/tsp_50/ablation_noewc_5 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_diag50_test_seed1234/concorde_costs.pkl -f

python eval.py data/tsp/tsp_diag50_test_seed1234.pkl --model outputs/tsp_50/ablation_nogenome_1 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_diag50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_diag50_test_seed1234.pkl --model outputs/tsp_50/ablation_nogenome_2 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_diag50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_diag50_test_seed1234.pkl --model outputs/tsp_50/ablation_nogenome_3 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_diag50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_diag50_test_seed1234.pkl --model outputs/tsp_50/ablation_nogenome_4 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_diag50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_diag50_test_seed1234.pkl --model outputs/tsp_50/ablation_nogenome_5 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_diag50_test_seed1234/concorde_costs.pkl -f
