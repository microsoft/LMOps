import hydra
import tqdm
import json
import random
import os

@hydra.main(config_path="configs", config_name="merge_data")
def main(cfg):
    print(cfg)


    with open(cfg.step1_train, "r") as step1_train:
        step1_train_data = json.load(step1_train)
    
    with open(cfg.step1_valid, "r") as step1_valid:
        step1_valid_data = json.load(step1_valid)
    
    with open(cfg.step2_train, "r") as step2_train:
        multistep_data = json.load(step2_train)

    with open(cfg.step2_valid, "r") as step2_valid:
        multistep_data += json.load(step2_valid)

    with open(cfg.step3_train, "r") as step3_train:
        multistep_data += json.load(step3_train)

    with open(cfg.step3_valid, "r") as step3_valid:
        multistep_data += json.load(step3_valid)

    train_qid = set()
    valid_qid = set()

    train_data = []
    valid_data = []

    for data in step1_train_data:
        train_qid.add(data["id"])
        d = {}
        d["choosen"] = []
        d.update(data)
        train_data.append(d)

    for data in step1_valid_data:
        valid_qid.add(data["id"])
        d = {}
        d["choosen"] = []
        d.update(data)
        valid_data.append(d)
    
    print(len(train_data), len(valid_data))

    for data in multistep_data:
        if data["id"] in train_qid:
            train_data.append(data)
        else:
            valid_data.append(data)

    random.seed(42)
    random.shuffle(train_data)
    random.shuffle(valid_data)
    print(len(train_data), len(valid_data))
    
    with open(cfg.train_output_path, "w") as w1:
        w1.write(json.dumps(train_data, indent=4) + "\n")

    with open(cfg.valid_output_path, "w") as w2:
        w2.write(json.dumps(valid_data, indent=4) + "\n")

if __name__ == "__main__":
    main()