import hydra
import tqdm
import json
import random
import os
import numpy
from scipy.stats import expon
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


def find_step3(cfg):
    with open(cfg.train_path, "r") as f1:
        datalist = json.load(f1)
    print(len(datalist))
    with open(cfg.valid_path, "r") as f2:
        datalist += (json.load(f2))
    print(len(datalist))

    random_finder = RandomFinder(cfg)
    data_list = random_finder.train_dataset
    idx_list = list(range(len(random_finder.prompt_pool)))

    x = range(0, cfg.L) # 50 is the ctxs number that scored in last step
    p = expon.pdf(x)
    p  /= sum(p)
    index = list(range(cfg.L))

    for i, element in tqdm.tqdm(enumerate(datalist)):
        random.seed(i)
        numpy.random.seed(i)
        qid = element["id"]
        
        choosen_ids = numpy.random.choice(index, size=cfg.choosen_num, replace=False, p=p)
        # print("***rank:", idx)
        element["step_3_have_choosen"] = []
        element["step_3_ctxs"] = []
        for choosen_id in choosen_ids:
            eid = element["ctxs"][choosen_id]["id"]
            choosen = [eid] + element["choosen"]
            element["step_3_have_choosen"].append(choosen)
            element["step_3_ctxs"].append([
                int(a)
                for a in random.sample([idx for idx in idx_list if idx != qid and idx not in choosen], k=min(cfg.L, len(data_list)-1)) # avoid selecting the task input itself
            ])
        element.pop("choosen") # add
        element.pop("ctxs")
    return datalist

@hydra.main(config_path="configs", config_name="random_finder_multi_step")
def main(cfg):
    print(cfg)

    data_list = find_step3(cfg)
    with open(cfg.output_path, "w") as writer:
        writer.write(json.dumps(data_list, indent=4) + "\n")


if __name__ == "__main__":
    main()

