# human_microbiome

python train_pred.py --dataset_name human_microbiome --model_type MLP --layer_num 2 --hidden_dims "30" --activation ELU --dropout 0.1 --batch_size 16 --epochs 500 --lr 1e-3 --l1_weight 1e-3 --l2_weight 1e-3 --early_stopping_patience 50 --exptag debug