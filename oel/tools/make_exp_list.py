import sys

def make_exp_list(config_str):
    fields = config_str.split(',')
    assert len(fields) == 6
    exp_name = fields[0]
    ckpt_start = int(fields[1])
    ckpt_end = int(fields[2])
    ckpt_step = int(fields[3])
    val_samples_limit = fields[4]
    val_samples_use = fields[5]

    base_path = f"/tmp/{exp_name}"
    exp_paths = []
    for step in range(ckpt_start, ckpt_end + ckpt_step, ckpt_step):
        path = f"{base_path}/global_step_{step}/extract_{val_samples_limit}_samples/experiences/experience_{val_samples_use}.txt"
        exp_paths.append(path)

    experience_list_file = f"{base_path}/experience_list.txt"
    print(f"Writing paths to {experience_list_file}")
    with open(experience_list_file, 'w') as f:
        for path in exp_paths:
            f.write(path + '\n')

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python make_exp_list.py <config_string>")
        sys.exit(1)
    
    config_string = sys.argv[1]
    make_exp_list(config_string)
