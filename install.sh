conda create -n deeplink_feature_interaction "python=3.11"
conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -y numpy scipy pandas matplotlib scikit-learn tqdm jupyter jupyterlab pytables autopep8 pylint wandb

conda create -n deeplink_feature_interaction_r r-essentials r-base
conda install -c r r-energy
