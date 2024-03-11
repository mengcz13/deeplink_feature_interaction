# human_microbiome
## best params from random search (vj6wyprv)
python train_pred.py --dataset_name human_microbiome --model_type MLP --layer_num 2 --hidden_dims 40 --activation ELU --dropout 0.0 --batch_size 64 --epochs 1200 --lr 0.0001 --l1_weight 0.01 --l2_weight 0.0 --early_stopping_patience 50 --exptag human_microbiome_best