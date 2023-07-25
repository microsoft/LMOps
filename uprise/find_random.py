import hydra
import tqdm
import json
import random
import os
from DPR.dpr.utils.tasks import task_map


class RandomFinder:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        os.makedirs(os.path.dirname(cfg.prompt_pool_path), exist_ok=True)
        os.makedirs(os.path.dirname(cfg.output_path), exist_ok=True)
        task = task_map.cls_dic[self.cfg.task_name]()
        self.train_dataset = task.get_dataset(
            split="train",
            ds_size=None if "ds_size" not in cfg else cfg.ds_size,
            cache_dir=cfg.cache_dir,
        )

        print("started creating the prompt pool")
        self.get_prompt_pool()
        print("finished creating the prompt pool")

    # sample and save prompt pool
    def get_prompt_pool(self):
        self.prompt_pool = self.train_dataset
        for i, entry in enumerate(self.prompt_pool):
            entry["id"] = i
            entry["task_name"] = self.cfg.task_name
        print("prompt_pool size", len(self.prompt_pool))
        with open(self.cfg.prompt_pool_path, "w") as f:
            json.dump(self.prompt_pool, f)

# for each task input, sample L prompts for scoring from the prompt pool (i.e., task training data)
def find(cfg):
    random_finder = RandomFinder(cfg)
    data_list = random_finder.train_dataset
    idx_list = list(range(len(random_finder.prompt_pool)))

    for i, element in tqdm.tqdm(enumerate(data_list)):
        random.seed(i)
        element["id"] = i
        # `ctxs` stores the sampled prompt ids 
        element["ctxs"] = [
            {"id": int(a)}
            for a in random.sample([idx for idx in idx_list if idx != i], k=min(cfg.L, len(data_list)-1)) # avoid selecting the task input itself
        ]
    return data_list


@hydra.main(config_path="configs", config_name="random_finder")
def main(cfg):
    print(cfg)

    data_list = find(cfg)
    with open(cfg.output_path, "w") as writer:
        writer.write(json.dumps(data_list, indent=4) + "\n")

if __name__ == "__main__":
    main()
