#!/bin/bash

# Training commands
echo "TRAINING TSP_50 MODEL 1"
python run.py --problem tsp --graph_size 50 --baseline rollout --epoch_size 65536 --batch_size 1024 --n_epochs 151 --checkpoint_epochs 5 --lr_decay 0.98 --bl_warmup_epochs 0 --pretrain_path pretrained/tsp_50 --hardness_adaptive_percent 100 --use_genome --ewc_lambda 1 --ewc_warmup_epochs 20 --run_name curriculum_ewc_genome_1
python run.py --problem tsp --graph_size 50 --baseline rollout --epoch_size 65536 --batch_size 1024 --n_epochs 151 --checkpoint_epochs 5 --lr_decay 0.98 --bl_warmup_epochs 0 --pretrain_path pretrained/tsp_50 --run_name baseline_unif_1
python run.py --problem tsp --graph_size 50 --baseline rollout --epoch_size 65536 --batch_size 1024 --n_epochs 151 --checkpoint_epochs 5 --lr_decay 0.98 --bl_warmup_epochs 0 --pretrain_path pretrained/tsp_50 --hardness_adaptive_percent 100 --run_name hac_fullbatch_1
python run.py --problem tsp --graph_size 50 --baseline rollout --epoch_size 65536 --batch_size 1024 --n_epochs 151 --checkpoint_epochs 5 --lr_decay 0.98 --bl_warmup_epochs 0 --pretrain_path pretrained/tsp_50 --hardness_adaptive_percent 100 --use_genome --run_name ablation_noewc_1
python run.py --problem tsp --graph_size 50 --baseline rollout --epoch_size 65536 --batch_size 1024 --n_epochs 151 --checkpoint_epochs 5 --lr_decay 0.98 --bl_warmup_epochs 0 --pretrain_path pretrained/tsp_50 --hardness_adaptive_percent 100 --ewc_lambda 1 --ewc_warmup_epochs 20 --run_name ablation_nogenome_1

echo "TRAINING TSP_50 MODEL 2"
python run.py --problem tsp --graph_size 50 --baseline rollout --epoch_size 65536 --batch_size 1024 --n_epochs 151 --checkpoint_epochs 5 --lr_decay 0.98 --bl_warmup_epochs 0 --pretrain_path pretrained/tsp_50 --hardness_adaptive_percent 100 --use_genome --ewc_lambda 1 --ewc_warmup_epochs 20 --run_name curriculum_ewc_genome_2
python run.py --problem tsp --graph_size 50 --baseline rollout --epoch_size 65536 --batch_size 1024 --n_epochs 151 --checkpoint_epochs 5 --lr_decay 0.98 --bl_warmup_epochs 0 --pretrain_path pretrained/tsp_50 --run_name baseline_unif_2
python run.py --problem tsp --graph_size 50 --baseline rollout --epoch_size 65536 --batch_size 1024 --n_epochs 151 --checkpoint_epochs 5 --lr_decay 0.98 --bl_warmup_epochs 0 --pretrain_path pretrained/tsp_50 --hardness_adaptive_percent 100 --run_name hac_fullbatch_2
python run.py --problem tsp --graph_size 50 --baseline rollout --epoch_size 65536 --batch_size 1024 --n_epochs 151 --checkpoint_epochs 5 --lr_decay 0.98 --bl_warmup_epochs 0 --pretrain_path pretrained/tsp_50 --hardness_adaptive_percent 100 --use_genome --run_name ablation_noewc_2
python run.py --problem tsp --graph_size 50 --baseline rollout --epoch_size 65536 --batch_size 1024 --n_epochs 151 --checkpoint_epochs 5 --lr_decay 0.98 --bl_warmup_epochs 0 --pretrain_path pretrained/tsp_50 --hardness_adaptive_percent 100 --ewc_lambda 1 --ewc_warmup_epochs 20 --run_name ablation_nogenome_2

echo "TRAINING TSP_50 MODEL 3"
python run.py --problem tsp --graph_size 50 --baseline rollout --epoch_size 65536 --batch_size 1024 --n_epochs 151 --checkpoint_epochs 5 --lr_decay 0.98 --bl_warmup_epochs 0 --pretrain_path pretrained/tsp_50 --hardness_adaptive_percent 100 --use_genome --ewc_lambda 1 --ewc_warmup_epochs 20 --run_name curriculum_ewc_genome_3
python run.py --problem tsp --graph_size 50 --baseline rollout --epoch_size 65536 --batch_size 1024 --n_epochs 151 --checkpoint_epochs 5 --lr_decay 0.98 --bl_warmup_epochs 0 --pretrain_path pretrained/tsp_50 --run_name baseline_unif_3
python run.py --problem tsp --graph_size 50 --baseline rollout --epoch_size 65536 --batch_size 1024 --n_epochs 151 --checkpoint_epochs 5 --lr_decay 0.98 --bl_warmup_epochs 0 --pretrain_path pretrained/tsp_50 --hardness_adaptive_percent 100 --run_name hac_fullbatch_3
python run.py --problem tsp --graph_size 50 --baseline rollout --epoch_size 65536 --batch_size 1024 --n_epochs 151 --checkpoint_epochs 5 --lr_decay 0.98 --bl_warmup_epochs 0 --pretrain_path pretrained/tsp_50 --hardness_adaptive_percent 100 --use_genome --run_name ablation_noewc_3
python run.py --problem tsp --graph_size 50 --baseline rollout --epoch_size 65536 --batch_size 1024 --n_epochs 151 --checkpoint_epochs 5 --lr_decay 0.98 --bl_warmup_epochs 0 --pretrain_path pretrained/tsp_50 --hardness_adaptive_percent 100 --ewc_lambda 1 --ewc_warmup_epochs 20 --run_name ablation_nogenome_3

echo "TRAINING TSP_50 MODEL 4"
python run.py --problem tsp --graph_size 50 --baseline rollout --epoch_size 65536 --batch_size 1024 --n_epochs 151 --checkpoint_epochs 5 --lr_decay 0.98 --bl_warmup_epochs 0 --pretrain_path pretrained/tsp_50 --hardness_adaptive_percent 100 --use_genome --ewc_lambda 1 --ewc_warmup_epochs 20 --run_name curriculum_ewc_genome_4
python run.py --problem tsp --graph_size 50 --baseline rollout --epoch_size 65536 --batch_size 1024 --n_epochs 151 --checkpoint_epochs 5 --lr_decay 0.98 --bl_warmup_epochs 0 --pretrain_path pretrained/tsp_50 --run_name baseline_unif_4
python run.py --problem tsp --graph_size 50 --baseline rollout --epoch_size 65536 --batch_size 1024 --n_epochs 151 --checkpoint_epochs 5 --lr_decay 0.98 --bl_warmup_epochs 0 --pretrain_path pretrained/tsp_50 --hardness_adaptive_percent 100 --run_name hac_fullbatch_4
python run.py --problem tsp --graph_size 50 --baseline rollout --epoch_size 65536 --batch_size 1024 --n_epochs 151 --checkpoint_epochs 5 --lr_decay 0.98 --bl_warmup_epochs 0 --pretrain_path pretrained/tsp_50 --hardness_adaptive_percent 100 --use_genome --run_name ablation_noewc_4
python run.py --problem tsp --graph_size 50 --baseline rollout --epoch_size 65536 --batch_size 1024 --n_epochs 151 --checkpoint_epochs 5 --lr_decay 0.98 --bl_warmup_epochs 0 --pretrain_path pretrained/tsp_50 --hardness_adaptive_percent 100 --ewc_lambda 1 --ewc_warmup_epochs 20 --run_name ablation_nogenome_4

echo "TRAINING TSP_50 MODEL 5"
python run.py --problem tsp --graph_size 50 --baseline rollout --epoch_size 65536 --batch_size 1024 --n_epochs 151 --checkpoint_epochs 5 --lr_decay 0.98 --bl_warmup_epochs 0 --pretrain_path pretrained/tsp_50 --hardness_adaptive_percent 100 --use_genome --ewc_lambda 1 --ewc_warmup_epochs 20 --run_name curriculum_ewc_genome_5
python run.py --problem tsp --graph_size 50 --baseline rollout --epoch_size 65536 --batch_size 1024 --n_epochs 151 --checkpoint_epochs 5 --lr_decay 0.98 --bl_warmup_epochs 0 --pretrain_path pretrained/tsp_50 --run_name baseline_unif_5
python run.py --problem tsp --graph_size 50 --baseline rollout --epoch_size 65536 --batch_size 1024 --n_epochs 151 --checkpoint_epochs 5 --lr_decay 0.98 --bl_warmup_epochs 0 --pretrain_path pretrained/tsp_50 --hardness_adaptive_percent 100 --run_name hac_fullbatch_5
python run.py --problem tsp --graph_size 50 --baseline rollout --epoch_size 65536 --batch_size 1024 --n_epochs 151 --checkpoint_epochs 5 --lr_decay 0.98 --bl_warmup_epochs 0 --pretrain_path pretrained/tsp_50 --hardness_adaptive_percent 100 --use_genome --run_name ablation_noewc_5
python run.py --problem tsp --graph_size 50 --baseline rollout --epoch_size 65536 --batch_size 1024 --n_epochs 151 --checkpoint_epochs 5 --lr_decay 0.98 --bl_warmup_epochs 0 --pretrain_path pretrained/tsp_50 --hardness_adaptive_percent 100 --ewc_lambda 1 --ewc_warmup_epochs 20 --run_name ablation_nogenome_5
