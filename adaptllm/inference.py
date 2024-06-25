import warnings
import torch
import tqdm
from torch.utils.data import DataLoader
from src.data.collators import DataCollatorWithPaddingAndCuda
import hydra.utils as hu
import hydra
import json
import os
from src.utils.cache_util import BufferedJsonWriter, BufferedJsonReader
from accelerate import Accelerator

from src.utils.metric import compute_scores
import glob
import logging
from transformers import StoppingCriteriaList, StoppingCriteria

logger = logging.getLogger(__name__)

class StopStrCriteria(StoppingCriteria):
    def __init__(self, tokenizer, stop_str, answer_start):
        self.stop_str = stop_str
        self.tokenizer = tokenizer
        self.answer_start = answer_start

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        sequences = self.tokenizer.batch_decode(input_ids[:, self.answer_start:])
        return all(self.stop_str in seq for seq in sequences)

class Inferencer:
    def __init__(self, cfg, accelerator, model) -> None:
        self.dataset_reader = hu.instantiate(cfg.dataset_reader)
        self.dataset_reader.shard(accelerator)
        if self.dataset_reader.tokenizer.pad_token_id is None or self.dataset_reader.tokenizer.pad_token_id <0:
            self.dataset_reader.tokenizer.pad_token_id = self.dataset_reader.tokenizer.eos_token_id

        self.accelerator = accelerator
        co = DataCollatorWithPaddingAndCuda(tokenizer=self.dataset_reader.tokenizer, device=accelerator.device)

        self.dataloader = DataLoader(self.dataset_reader, batch_size=self.dataset_reader.task.inf_bsz, collate_fn=co)
        self.model=model

        self.output_file = os.path.join(cfg.output_dir, f'{os.path.basename(cfg.model_name)}_{cfg.task_name}.json')
        self.res_file = os.path.join(cfg.res_dir, f'{cfg.task_name}.txt')
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        os.makedirs(os.path.dirname(self.res_file), exist_ok=True)
        
        self.cfg = cfg
        self.option_num = self.dataset_reader.task.class_num
        self.max_length = cfg.max_length  # used for text completion task,
        self.generate_max_len = cfg.generate_max_len  # max seq len to be generated

    def choice_losses(self, input_ids, input_atten_mask, loss_mask, labels):
        bsz, option_num, seq_len = input_ids.shape
        if self.option_num is not None:
            assert option_num == self.option_num
        with torch.no_grad():
            output = self.model(
                input_ids=input_ids.reshape(bsz * option_num, seq_len),
                attention_mask=input_atten_mask.reshape(bsz * option_num, seq_len),
            )

        # (bsz, option_num, seq_len, vocab_size)
        logits = output.logits.reshape(bsz, option_num, seq_len, -1)

        # (bsz, option_num, seq_len-1, vocab_size)
        logits = logits[:, :, :-1, :]

        # (bsz, option_num, seq_len-1, 1)
        targets = input_ids[:, :, 1:].unsqueeze(-1)

        # (bsz, option_num, seq_len-1, vocab_size)
        logit_probs = torch.nn.functional.log_softmax(logits.float(), dim=-1)

        # (bsz, option_num, seq_len-1)
        loss_mask = loss_mask[:, :, 1:]

        # (bsz, option_num, seq_len-1,1), squeeze to (bsz, option_num, seq_len-1), loss mask to (bsz, option_num, answer_len)
        loss = - torch.gather(logit_probs, -1, targets).squeeze(-1) * loss_mask

        # (bsz, option_num)
        loss = loss.sum(-1) / loss_mask.sum(-1)

        # (bsz,)
        preds = torch.argmin(loss, dim=-1).tolist()
        normed_loss = torch.nn.functional.normalize(loss, p=1, dim=-1)
        
        # roughly get pred_probs for all classes
        pred_probs = 2/option_num - normed_loss
        pred_probs = pred_probs.tolist()
        labels = labels.squeeze(-1).tolist()
        assert len(labels) == len(preds)
        return {"preds": preds, "labels": labels, "pred_probs": pred_probs}

    def completion_losses(self, input_ids, input_atten_mask, labels):
        bsz, option_num, seq_len = input_ids.shape
        answer_start = int(input_atten_mask.shape[-1])
        stopping_criteria = StoppingCriteriaList([StopStrCriteria(tokenizer=self.dataset_reader.tokenizer, 
                                                                stop_str='\n', answer_start=answer_start)])
        assert option_num == 1
        with torch.no_grad():
            res = self.model.generate(
                input_ids=input_ids.squeeze(1),  # remove the dim of option_num
                attention_mask=input_atten_mask.squeeze(1),
                pad_token_id=self.dataset_reader.tokenizer.pad_token_id,
                max_length=min(self.max_length, answer_start + self.generate_max_len),
                do_sample=False,
                stopping_criteria=stopping_criteria,
            )
        pred_ids = res[:, answer_start:]
        preds = []
        for i in range(len(pred_ids)):
            preds.append(self.dataset_reader.tokenizer.decode(pred_ids[i], skip_special_tokens=True))
        return {"preds": preds, "labels": labels, "pred_probs": [None] * len(preds)}

    def forward(self):
        if self.accelerator.is_main_process:
            dataloader = tqdm.tqdm(self.dataloader)
        else:
            dataloader = self.dataloader
        cached_file_path = f"{self.output_file}tmp_{self.accelerator.device}.bin"
        with BufferedJsonWriter(cached_file_path) as buffer:
            if os.path.isfile(cached_file_path):
                os.remove(cached_file_path)
                print("cached file: % s removed successfully" % cached_file_path) 
            for i, entry in enumerate(dataloader):
                if "stop" in self.cfg and i == self.cfg.stop:
                    break  # early stop for debug
                metadata = entry.pop("metadata")
                if self.dataset_reader.task.class_num == 1:
                    few_shot_res = self.completion_losses(
                        input_ids=entry.input_ids,
                        input_atten_mask=entry.input_atten_mask,
                        labels=[x["label"] for x in metadata],
                    )
                else:
                    few_shot_res = self.choice_losses(
                        input_ids=entry.input_ids,
                        input_atten_mask=entry.input_atten_mask,
                        loss_mask=entry.loss_mask,
                        labels=entry.labels,
                    )
                for i in range(len(metadata)):
                    metadata[i]["pred_prob"] = few_shot_res["pred_probs"][i]
                    metadata[i]["pred"] = few_shot_res["preds"][i]
                    metadata[i]["label"] = few_shot_res["labels"][i]
                buffer.write(metadata)

    def write_predictions(self):
        data = []
        for path in glob.glob(f"{self.output_file}tmp_*.bin"):
            with BufferedJsonReader(path) as f:
                for x in f.read():
                    data.extend(x)
        logger.info("num of saved preds: %s", str(len(data)))
        scores = compute_scores(self.dataset_reader.task.metric, data)

        with open(self.output_file, "w") as f:
            f.write(json.dumps(data, indent=4) + "\n")
        logger.info("scores: %s", str(scores))
        with open(self.res_file, "a") as f:
            info = f"model: {str(self.cfg.model_name)}; scores: {str(scores)}\n"
            f.write(info)
        
        print("saved pred to: ", self.output_file)
        print("saved eval res to: ", self.res_file)
        for path in glob.glob(f"{self.output_file}tmp_*.bin"):
            os.remove(path)
        return data


@hydra.main(config_path="configs", config_name="inference")
def main(cfg):
    logger.info(cfg)

    accelerator = Accelerator()
    # load model once and run for many tasks
    model = hu.instantiate(cfg.model).half()
    if not cfg.model_parallel:  # map whole LLM to each GPU
        model = model.to(accelerator.device)
        print('map whole LLM to each GPU')
    model = model.eval()
    if hasattr(model, "module"):
        model = model.module

    # loop for tasks
    tasks = cfg.task_name.split('+')
    for task in tasks: 
        print(f'infer on {task}...')
        cfg.task_name = task
        inferencer = Inferencer(cfg, accelerator, model)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            inferencer.forward()
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                inferencer.write_predictions()


if __name__ == "__main__":
    main()
