from argparse import ArgumentParser
import torch


def construct_arg_parser():
    parser = ArgumentParser()
    ## data
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--nrep", type=int, default=0, help='used in filtered features')
    ## model
    parser.add_argument("--model_type", type=str, choices=["MLP"], required=True)
    temp_args, _ = parser.parse_known_args()
    if temp_args.model_type == "MLP":
        group = parser.add_argument_group("MLP", "MLP model arguments")
        group.add_argument("--layer_num", type=int, default=2, help="1 + hidden layers")
        group.add_argument("--hidden_dims", type=str, default='64', help='- separated or single number')
        group.add_argument("--activation", type=str, default="ELU")
        group.add_argument("--dropout", type=float, default=0.0)
    else:
        raise NotImplementedError(f"Model type {temp_args.model_type} not implemented")
    ## training
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--loss_type", type=str, default='BCEWithLogits')
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--l1_weight", type=float, default=0.0)
    parser.add_argument("--l2_weight", type=float, default=0.0)
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=-1,
        help="-1 for no early stopping",
    )
    parser.add_argument("--early_stopping_metric", type=str, default="val_loss")
    ## logging
    parser.add_argument("--wandb_project", type=str, default="deeplink_feature_interaction")
    parser.add_argument("--expname", type=str, default="")
    parser.add_argument("--exptag", type=str, default="")
    ## envs
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")

    return parser
