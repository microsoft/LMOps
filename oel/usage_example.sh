# Sokoban, Qwen3-4B-Instruct-2507 (non-thinking model)
#   Round 1
#   1. Experiential knowledge extraction.
bash scripts/textgame_extract_inturn.sh "oel-sokoban-q3-4b-ins-ext-v4-selwp-round1,50,500,50,Qwen/Qwen3-4B-Instruct-2507,,v4,100,True,8192,Sokoban-v0,1024,5,True,1," ; python tools/make_exp_list.py "oel-sokoban-q3-4b-ins-ext-v4-selwp-round1,50,500,50,100,50"
#   2. Collection of user trajectories, to construct partial rollouts later.
bash scripts/textgame_generate_deploy.sh --model Qwen/Qwen3-4B-Instruct-2507 --exp_name oel-sokoban-q3-4b-ins-round1-deploy --nnodes 1 --oel_round 1 --experience_max_length 8192 --textgame_name Sokoban-v0 --max_response_length 1024 --textgame_max_steps 5 --textgame_no_think True --total_training_steps 100
#   3. Experiential knowledge consolidation.
bash scripts/textgame_consolidate.sh --model Qwen/Qwen3-4B-Instruct-2507 --exp_name oel-sokoban-q3-4b-ins-v4-selwp-lr1e-6-round1 --nnodes 2 --oel_round 1 --kl_loss_type full --kl_topk 256 --actor_lr 1e-6 --experience_max_length 8192 --textgame_name Sokoban-v0 --max_response_length 1024 --textgame_max_steps 5 --textgame_no_think True --deploy_save_dir /tmp/oel-sokoban-q3-4b-ins-round1-deploy/deploy_data --exp_path /tmp/oel-sokoban-q3-4b-ins-ext-v4-selwp-round1/experience_list.txt --total_training_steps 100 --save_freq 2
#   (Optional) Evaluate checkpoints during consolidation.
bash scripts/textgame_eval_inturn.sh "oel-sokoban-q3-4b-ins-v4-selwp-lr1e-6-round1,2,100,2,Qwen/Qwen3-4B-Instruct-2507,false,1024,Sokoban-v0,5,true"

#   Round 2
#   1. Experiential knowledge extraction.
bash scripts/textgame_extract_inturn.sh "oel-sokoban-q3-4b-ins-ext-v4-selwp-round2,50,500,50,oel-sokoban-q3-4b-ins-v4-selwp-lr1e-6-round1,100,v4,100,True,8192,Sokoban-v0,1024,5,True,2," ; python tools/make_exp_list.py "oel-sokoban-q3-4b-ins-ext-v4-selwp-round2,50,500,50,100,50"
#   2. Collection of user trajectories, to construct partial rollouts later.
bash scripts/textgame_generate_deploy.sh --resume_policy_name oel-sokoban-q3-4b-ins-v4-selwp-lr1e-6-round1 --resume_policy_ckpt 100 --exp_name oel-sokoban-q3-4b-ins-round2-deploy --nnodes 1 --oel_round 2 --experience_max_length 8192 --textgame_name Sokoban-v0 --max_response_length 1024 --textgame_max_steps 5 --textgame_no_think True --total_training_steps 100
#   3. Experiential knowledge consolidation.
bash scripts/textgame_consolidate.sh --resume_policy_name oel-sokoban-q3-4b-ins-v4-selwp-lr1e-6-round1 --resume_policy_ckpt 100 --exp_name oel-sokoban-q3-4b-ins-v4-selwp-lr1e-6-round2 --nnodes 2 --oel_round 2 --kl_loss_type full --kl_topk 256 --actor_lr 1e-6 --experience_max_length 8192 --textgame_name Sokoban-v0 --max_response_length 1024 --textgame_max_steps 5 --textgame_no_think True --deploy_save_dir /tmp/oel-sokoban-q3-4b-ins-round2-deploy/deploy_data --exp_path /tmp/oel-sokoban-q3-4b-ins-ext-v4-selwp-round2/experience_list.txt --total_training_steps 100 --save_freq 2

#   Round 3
#   1. Experiential knowledge extraction.
bash scripts/textgame_extract_inturn.sh "oel-sokoban-q3-4b-ins-ext-v4-selwp-round3,50,500,50,oel-sokoban-q3-4b-ins-v4-selwp-lr1e-6-round2,100,v4,100,True,8192,Sokoban-v0,1024,5,True,3," ; python tools/make_exp_list.py "oel-sokoban-q3-4b-ins-ext-v4-selwp-round3,50,500,50,100,50"
#   2. Collection of user trajectories, to construct partial rollouts later.
bash scripts/textgame_generate_deploy.sh --resume_policy_name oel-sokoban-q3-4b-ins-v4-selwp-lr1e-6-round2 --resume_policy_ckpt 100 --exp_name oel-sokoban-q3-4b-ins-round3-deploy --nnodes 1 --oel_round 3 --experience_max_length 8192 --textgame_name Sokoban-v0 --max_response_length 1024 --textgame_max_steps 5 --textgame_no_think True --total_training_steps 100
#   3. Experiential knowledge consolidation.
bash scripts/textgame_consolidate.sh --resume_policy_name oel-sokoban-q3-4b-ins-v4-selwp-lr1e-6-round2 --resume_policy_ckpt 100 --exp_name oel-sokoban-q3-4b-ins-v4-selwp-lr1e-6-round3 --nnodes 2 --oel_round 3 --kl_loss_type full --kl_topk 256 --actor_lr 1e-6 --experience_max_length 8192 --textgame_name Sokoban-v0 --max_response_length 1024 --textgame_max_steps 5 --textgame_no_think True --deploy_save_dir /tmp/oel-sokoban-q3-4b-ins-round3-deploy/deploy_data --exp_path /tmp/oel-sokoban-q3-4b-ins-ext-v4-selwp-round3/experience_list.txt --total_training_steps 100 --save_freq 2





# Frozen Lake, Qwen3-1.7B (thinking model)
#   Round 1
#   1. Experiential knowledge extraction.
bash scripts/textgame_extract_inturn.sh "oel-frozenlake-q3-1b7-ext-v3-selwop-round1,50,500,50,Qwen/Qwen3-1.7B,,v3,30,False,2048,FrozenLake-v0-raw,1024,5,False,1," ; python tools/make_exp_list.py "oel-frozenlake-q3-1b7-ext-v3-selwop-round1,50,500,50,30,15"
#   2. Collection of user trajectories, to construct partial rollouts later.
bash scripts/textgame_generate_deploy.sh --model Qwen/Qwen3-1.7B --exp_name oel-frozenlake-q3-1b7-round1-deploy --nnodes 1 --oel_round 1 --experience_max_length 2048 --textgame_name FrozenLake-v0-raw --max_response_length 1024 --textgame_max_steps 5 --textgame_no_think False --total_training_steps 20
#   3. Experiential knowledge consolidation.
bash scripts/textgame_consolidate.sh --model Qwen/Qwen3-1.7B --exp_name oel-frozenlake-q3-1b7-v3-selwop-lr5e-6-round1 --nnodes 2 --oel_round 1 --kl_loss_type full --kl_topk 256 --actor_lr 5e-6 --experience_max_length 2048 --textgame_name FrozenLake-v0-raw --max_response_length 1024 --textgame_max_steps 5 --textgame_no_think False --deploy_save_dir /tmp/oel-frozenlake-q3-1b7-round1-deploy/deploy_data --exp_path /tmp/oel-frozenlake-q3-1b7-ext-v3-selwop-round1/experience_list.txt --total_training_steps 20 --save_freq 2
#   (Optional) Evaluate checkpoints during consolidation.
bash scripts/textgame_eval_inturn.sh "oel-frozenlake-q3-1b7-v3-selwop-lr5e-6-round1,2,20,2,Qwen/Qwen3-1.7B,false,1024,FrozenLake-v0-raw,5,false"

#   Round 2
#   1. Experiential knowledge extraction.
bash scripts/textgame_extract_inturn.sh "oel-frozenlake-q3-1b7-ext-v3-selwop-round2,50,500,50,oel-frozenlake-q3-1b7-v3-selwop-lr5e-6-round1,20,v3,30,False,2048,FrozenLake-v0-raw,1024,5,False,2," ; python tools/make_exp_list.py "oel-frozenlake-q3-1b7-ext-v3-selwop-round2,50,500,50,30,15"
#   2. Collection of user trajectories, to construct partial rollouts later.
bash scripts/textgame_generate_deploy.sh --resume_policy_name oel-frozenlake-q3-1b7-v3-selwop-lr5e-6-round1 --resume_policy_ckpt 20 --exp_name oel-frozenlake-q3-1b7-round2-deploy --nnodes 1 --oel_round 2 --experience_max_length 2048 --textgame_name FrozenLake-v0-raw --max_response_length 1024 --textgame_max_steps 5 --textgame_no_think False --total_training_steps 20
#   3. Experiential knowledge consolidation.
bash scripts/textgame_consolidate.sh --resume_policy_name oel-frozenlake-q3-1b7-v3-selwop-lr5e-6-round1 --resume_policy_ckpt 20 --exp_name oel-frozenlake-q3-1b7-v3-selwop-lr5e-6-round2 --nnodes 2 --oel_round 2 --kl_loss_type full --kl_topk 256 --actor_lr 5e-6 --experience_max_length 2048 --textgame_name FrozenLake-v0-raw --max_response_length 1024 --textgame_max_steps 5 --textgame_no_think False --deploy_save_dir /tmp/oel-frozenlake-q3-1b7-round2-deploy/deploy_data --exp_path /tmp/oel-frozenlake-q3-1b7-ext-v3-selwop-round2/experience_list.txt --total_training_steps 20 --save_freq 2





# Frozen Lake, Qwen3-4B (thinking model)
#   Round 1
#   1. Experiential knowledge extraction.
bash scripts/textgame_extract_inturn.sh "oel-frozenlake-q3-4b-ext-v4-selwp-round1,50,500,50,Qwen/Qwen3-4B,,v4,50,True,8192,FrozenLake-v0-raw,1024,5,False,1," ; python tools/make_exp_list.py "oel-frozenlake-q3-4b-ext-v4-selwp-round1,50,500,50,50,25"
#   2. Collection of user trajectories, to construct partial rollouts later.
bash scripts/textgame_generate_deploy.sh --model Qwen/Qwen3-4B --exp_name oel-frozenlake-q3-4b-round1-deploy --nnodes 1 --oel_round 1 --experience_max_length 8192 --textgame_name FrozenLake-v0-raw --max_response_length 1024 --textgame_max_steps 5 --textgame_no_think False --total_training_steps 20
#   3. Experiential knowledge consolidation.
bash scripts/textgame_consolidate.sh --model Qwen/Qwen3-4B --exp_name oel-frozenlake-q3-4b-v4-selwp-lr1e-6-round1 --nnodes 2 --oel_round 1 --kl_loss_type full --kl_topk 256 --actor_lr 1e-6 --experience_max_length 8192 --textgame_name FrozenLake-v0-raw --max_response_length 1024 --textgame_max_steps 5 --textgame_no_think False --deploy_save_dir /tmp/oel-frozenlake-q3-4b-round1-deploy/deploy_data --exp_path /tmp/oel-frozenlake-q3-4b-ext-v4-selwp-round1/experience_list.txt --total_training_steps 20 --save_freq 2
#   (Optional) Evaluate checkpoints during consolidation.
bash scripts/textgame_eval_inturn.sh "oel-frozenlake-q3-4b-v4-selwp-lr1e-6-round1,2,20,2,Qwen/Qwen3-4B,false,1024,FrozenLake-v0-raw,5,false"

#   Round 2
#   1. Experiential knowledge extraction.
bash scripts/textgame_extract_inturn.sh "oel-frozenlake-q3-4b-ext-v4-selwp-round2,50,500,50,oel-frozenlake-q3-4b-v4-selwp-lr1e-6-round1,20,v4,50,True,8192,FrozenLake-v0-raw,1024,5,False,2," ; python tools/make_exp_list.py "oel-frozenlake-q3-4b-ext-v4-selwp-round2,50,500,50,50,25"
#   2. Collection of user trajectories, to construct partial rollouts later.
bash scripts/textgame_generate_deploy.sh --resume_policy_name oel-frozenlake-q3-4b-v4-selwp-lr1e-6-round1 --resume_policy_ckpt 20 --exp_name oel-frozenlake-q3-4b-round2-deploy --nnodes 1 --oel_round 2 --experience_max_length 8192 --textgame_name FrozenLake-v0-raw --max_response_length 1024 --textgame_max_steps 5 --textgame_no_think False --total_training_steps 20
#   3. Experiential knowledge consolidation.
bash scripts/textgame_consolidate.sh --resume_policy_name oel-frozenlake-q3-4b-v4-selwp-lr1e-6-round1 --resume_policy_ckpt 20 --exp_name oel-frozenlake-q3-4b-v4-selwp-lr1e-6-round2 --nnodes 2 --oel_round 2 --kl_loss_type full --kl_topk 256 --actor_lr 1e-6 --experience_max_length 8192 --textgame_name FrozenLake-v0-raw --max_response_length 1024 --textgame_max_steps 5 --textgame_no_think False --deploy_save_dir /tmp/oel-frozenlake-q3-4b-round2-deploy/deploy_data --exp_path /tmp/oel-frozenlake-q3-4b-ext-v4-selwp-round2/experience_list.txt --total_training_steps 20 --save_freq 2





# Frozen Lake, Qwen3-8B (thinking model)
#   Round 1
#   1. Experiential knowledge extraction.
bash scripts/textgame_extract_inturn.sh "oel-frozenlake-q3-8b-ext-v4-selwp-round1,50,500,50,Qwen/Qwen3-8B,,v4,100,True,8192,FrozenLake-v0-raw,1024,5,False,1," ; python tools/make_exp_list.py "oel-frozenlake-q3-8b-ext-v4-selwp-round1,50,500,50,100,50"
#   2. Collection of user trajectories, to construct partial rollouts later.
bash scripts/textgame_generate_deploy.sh --model Qwen/Qwen3-8B --exp_name oel-frozenlake-q3-8b-round1-deploy --nnodes 1 --oel_round 1 --experience_max_length 8192 --textgame_name FrozenLake-v0-raw --max_response_length 1024 --textgame_max_steps 5 --textgame_no_think False --total_training_steps 100
#   3. Experiential knowledge consolidation.
bash scripts/textgame_consolidate.sh --model Qwen/Qwen3-8B --exp_name oel-frozenlake-q3-8b-v4-selwp-lr1e-6-round1 --nnodes 2 --oel_round 1 --kl_loss_type full --kl_topk 256 --actor_lr 1e-6 --experience_max_length 8192 --textgame_name FrozenLake-v0-raw --max_response_length 1024 --textgame_max_steps 5 --textgame_no_think False --deploy_save_dir /tmp/oel-frozenlake-q3-8b-round1-deploy/deploy_data --exp_path /tmp/oel-frozenlake-q3-8b-ext-v4-selwp-round1/experience_list.txt --total_training_steps 100 --save_freq 2
#   (Optional) Evaluate checkpoints during consolidation.
bash scripts/textgame_eval_inturn.sh "oel-frozenlake-q3-8b-v4-selwp-lr1e-6-round1,2,100,2,Qwen/Qwen3-8B,false,1024,FrozenLake-v0-raw,5,false"

#   Round 2
#   1. Experiential knowledge extraction.
bash scripts/textgame_extract_inturn.sh "oel-frozenlake-q3-8b-ext-v4-selwp-round2,50,500,50,oel-frozenlake-q3-8b-v4-selwp-lr1e-6-round1,100,v4,100,True,8192,FrozenLake-v0-raw,1024,5,False,2," ; python tools/make_exp_list.py "oel-frozenlake-q3-8b-ext-v4-selwp-round2,50,500,50,100,50"
#   2. Collection of user trajectories, to construct partial rollouts later.
bash scripts/textgame_generate_deploy.sh --resume_policy_name oel-frozenlake-q3-8b-v4-selwp-lr1e-6-round1 --resume_policy_ckpt 100 --exp_name oel-frozenlake-q3-8b-round2-deploy --nnodes 1 --oel_round 2 --experience_max_length 8192 --textgame_name FrozenLake-v0-raw --max_response_length 1024 --textgame_max_steps 5 --textgame_no_think False --total_training_steps 100
#   3. Experiential knowledge consolidation.
bash scripts/textgame_consolidate.sh --resume_policy_name oel-frozenlake-q3-8b-v4-selwp-lr1e-6-round1 --resume_policy_ckpt 100 --exp_name oel-frozenlake-q3-8b-v4-selwp-lr1e-6-round2 --nnodes 2 --oel_round 2 --kl_loss_type full --kl_topk 256 --actor_lr 1e-6 --experience_max_length 8192 --textgame_name FrozenLake-v0-raw --max_response_length 1024 --textgame_max_steps 5 --textgame_no_think False --deploy_save_dir /tmp/oel-frozenlake-q3-8b-round2-deploy/deploy_data --exp_path /tmp/oel-frozenlake-q3-8b-ext-v4-selwp-round2/experience_list.txt --total_training_steps 100 --save_freq 2