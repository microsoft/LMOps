export NCCL_DEBUG=""
# pip3 install transformers
bash blob_yuxian.sh
git submodule update --init
cd transformers
git checkout data_selection
cd ..
pip3 install -e transformers
pip3 install torch
pip3 install deepspeed
pip3 install nltk
pip3 install numerize
pip3 install torchtyping
pip3 install rich
pip3 install accelerate
pip3 install datasets
pip3 install sentencepiece
pip3 install peft
pip3 install matplotlib
pip3 install wandb
pip3 install cvxpy
pip3 install h5py
pip3 install scikit-learn
pip3 install Levenshtein
pip3 install xformers==0.0.26.post1
