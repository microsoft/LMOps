set -ex

# =================== analyze 1.3b GPT ===================

model=1_3b

task=cb
python icl_ft/compute_sim.py $task all $model

task=sst2
python icl_ft/compute_sim.py $task all $model

task=sst5
python icl_ft/compute_sim.py $task all $model

task=subj
python icl_ft/compute_sim.py $task all $model

task=mr
python icl_ft/compute_sim.py $task all $model

task=agnews
python icl_ft/compute_sim.py $task all $model

# =================== analyze 2.7b GPT ===================

model=2_7b

task=cb
python icl_ft/compute_sim.py $task all $model

task=sst2
python icl_ft/compute_sim.py $task all $model

task=sst5
python icl_ft/compute_sim.py $task all $model

task=subj
python icl_ft/compute_sim.py $task all $model

task=mr
python icl_ft/compute_sim.py $task all $model

task=agnews
python icl_ft/compute_sim.py $task all $model

# =================== analyze 1.3b GPT training ===================

model=1_3b

task=cb
python icl_ft/compute_training_example_attn.py $task $model

task=sst2
python icl_ft/compute_training_example_attn.py $task $model

task=sst5
python icl_ft/compute_training_example_attn.py $task $model

task=subj
python icl_ft/compute_training_example_attn.py $task $model

task=mr
python icl_ft/compute_training_example_attn.py $task $model

task=agnews
python icl_ft/compute_training_example_attn.py $task $model

# =================== analyze 2.7b GPT training ===================

model=2_7b

task=cb
python icl_ft/compute_training_example_attn.py $task $model

task=sst2
python icl_ft/compute_training_example_attn.py $task $model

task=sst5
python icl_ft/compute_training_example_attn.py $task $model

task=subj
python icl_ft/compute_training_example_attn.py $task $model

task=mr
python icl_ft/compute_training_example_attn.py $task $model

task=agnews
python icl_ft/compute_training_example_attn.py $task $model