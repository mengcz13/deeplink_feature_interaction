# human_microbiome
## best params from random search (vj6wyprv)
python train_pred.py --dataset_name human_microbiome --model_type MLP --layer_num 2 --hidden_dims 40 --activation ELU --dropout 0.0 --batch_size 64 --epochs 1200 --lr 0.0001 --l1_weight 0.01 --l2_weight 0.0 --early_stopping_patience 50 --exptag human_microbiome_best
## run archipelago
python explain_model.py --dataset_name human_microbiome --model_type MLP --layer_num 2 --hidden_dims 40 --activation ELU --dropout 0.0 --batch_size 64 --epochs 1200 --lr 0.0001 --l1_weight 0.01 --l2_weight 0.0 --early_stopping_patience 50 --exptag human_microbiome_best --load_model save/vj6wyprv/best_model.pth --save_explanation_dir explanations/human_microbiome_best

# murine_sc_RNAseq (oeu9oz8x)
python train_pred.py --dataset_name murine_sc_RNAseq --model_type MLP --layer_num 2 --hidden_dims 100 --activation LeakyReLU --dropout 0.3 --batch_size 1 --epochs 500 --lr 0.0001 --l1_weight 0.01 --l2_weight 0.0001 --early_stopping_patience 50 --exptag murine_sc_RNAseq_param_search
python explain_model.py --dataset_name murine_sc_RNAseq --model_type MLP --layer_num 2 --hidden_dims 100 --activation LeakyReLU --dropout 0.3 --batch_size 1 --epochs 500 --lr 0.0001 --l1_weight 0.01 --l2_weight 0.0001 --early_stopping_patience 50 --exptag murine_sc_RNAseq_best --load_model save/oeu9oz8x/best_model.pth --save_explanation_dir explanations/murine_sc_RNAseq_best

# human_sc_RNAseq (gf7igdwp)
python train_pred.py --dataset_name human_sc_RNAseq --model_type MLP --layer_num 2 --hidden_dims 100 --activation LeakyReLU --dropout 0.2 --batch_size 4 --epochs 500 --lr 0.001 --l1_weight 0.001 --l2_weight 0.0001 --early_stopping_patience 50 --exptag human_sc_RNAseq_param_search
python explain_model.py --dataset_name human_sc_RNAseq --model_type MLP --layer_num 2 --hidden_dims 100 --activation LeakyReLU --dropout 0.2 --batch_size 4 --epochs 500 --lr 0.001 --l1_weight 0.001 --l2_weight 0.0001 --early_stopping_patience 50 --exptag human_sc_RNAseq_best --load_model save/gf7igdwp/best_model.pth --save_explanation_dir explanations/human_sc_RNAseq_best
