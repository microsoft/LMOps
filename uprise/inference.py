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
from omegaconf import OmegaConf
import glob
import logging
from transformers import AutoModelForCausalLM

logger = logging.getLogger(__name__)


class Inferencer:
    def __init__(self, cfg, accelerator) -> None:
        self.dataset_reader = hu.instantiate(cfg.dataset_reader)
        self.dataset_reader.shard(accelerator)
        self.dataset_reader.tokenizer.pad_token_id = (
            self.dataset_reader.tokenizer.eos_token_id
        )  # to avoid error
        self.accelerator = accelerator
        co = DataCollatorWithPaddingAndCuda(
            tokenizer=self.dataset_reader.tokenizer, device=accelerator.device
        )

        self.dataloader = DataLoader(
            self.dataset_reader, batch_size=cfg.batch_size, collate_fn=co
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            device_map="auto",
            pretrained_model_name_or_path=cfg.model_name,
            cache_dir=cfg.cache_dir,
        ).half()

        self.model = self.model.eval()
        if hasattr(self.model, "module"):
            self.model = self.model.module

        self.output_file = cfg.output_file
        self.res_file = cfg.res_file

        os.makedirs(os.path.dirname(cfg.output_file), exist_ok=True)
        os.makedirs(os.path.dirname(cfg.res_file), exist_ok=True)

        self.cfg = cfg
        self.option_num = self.dataset_reader.task.class_num
        self.num_prompts = cfg.num_prompts
        self.random_sample = cfg.random_sample
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

        logits = output.logits.reshape(
            bsz, option_num, seq_len, -1
        )  # (bsz, option_num, seq_len, vocab_size)
        logits = logits[:, :, :-1, :]  # (bsz, option_num, seq_len-1, vocab_size)
        targets = input_ids[:, :, 1:].unsqueeze(-1)  # (bsz,option_num, seq_len-1, 1)
        logit_probs = torch.nn.functional.log_softmax(
            logits.float(), dim=-1
        )  # (bsz, option_num, seq_len-1,vocab_size)
        loss_mask = loss_mask[:, :, 1:]  # (bsz, option_num, seq_len-1)
        loss = -torch.gather(logit_probs, -1, targets).squeeze(-1) * loss_mask
        # (bsz, option_num, seq_len-1,1), squeeze to (bsz, option_num, seq_len-1), loss mask to (bsz, option_num, answer_len)
        loss = loss.sum(-1) / loss_mask.sum(-1)  # (bsz, option_num)
        preds = torch.argmin(loss, dim=-1).tolist()  # (bsz,)
        labels = labels.squeeze(-1).tolist()
        assert len(labels) == len(preds)
        return {"preds": preds, "labels": labels}

    def completion_losses(self, input_ids, input_atten_mask, labels):
        bsz, option_num, seq_len = input_ids.shape
        assert option_num == 1
        with torch.no_grad():
            answer_start = int(input_atten_mask.shape[-1])
            res = self.model.generate(
                input_ids=input_ids.squeeze(1),  # remove the dim of option_num
                attention_mask=input_atten_mask.squeeze(1),
                eos_token_id=self.dataset_reader.tokenizer.encode("\n")[0],
                pad_token_id=self.dataset_reader.tokenizer.pad_token_id,
                max_length=min(self.max_length, answer_start + self.generate_max_len),
                do_sample=False,
            )
        pred_ids = res[:, answer_start:]
        preds = []
        for i in range(len(pred_ids)):
            preds.append(
                self.dataset_reader.tokenizer.decode(
                    pred_ids[i], skip_special_tokens=True
                )
            )
        return {"preds": preds, "labels": labels}

    def forward(self):
        if self.accelerator.is_main_process:
            dataloader = tqdm.tqdm(self.dataloader)
        else:
            dataloader = self.dataloader
        with BufferedJsonWriter(
            f"{self.output_file}tmp_{self.accelerator.device}.bin"
        ) as buffer:
            for i, entry in enumerate(dataloader):
                if "stop" in self.cfg and i == self.cfg.stop:
                    break  # pass stop for debug
                metadata = entry.pop("metadata")
                if self.dataset_reader.task.class_num == 1:
                    few_shot_res = self.completion_losses(
                        input_ids=entry.input_ids,
                        input_atten_mask=entry.input_atten_mask,
                        labels=[x.pop("temp_label") for x in metadata],
                    )
                else:
                    few_shot_res = self.choice_losses(
                        input_ids=entry.input_ids,
                        input_atten_mask=entry.input_atten_mask,
                        loss_mask=entry.input_loss_mask,
                        labels=entry.labels,
                    )
                for i in range(len(metadata)):
                    metadata[i]["pred"] = few_shot_res["preds"][i]
                    metadata[i]["label"] = few_shot_res["labels"][i]
                buffer.write(metadata)

    def write_predictions(self):
        data = []
        for path in glob.glob(f"{self.output_file}tmp_*.bin"):
            with BufferedJsonReader(path) as f:
                for x in f.read():
                    data.extend(x)
        scores = compute_scores(self.dataset_reader.task.metric, data)
        with open(self.output_file, "w") as f:
            f.write(json.dumps(data, indent=4) + "\n")
        logger.info("scores: %s", str(scores))
        if "bm25" in self.cfg.prompt_file:
            retriever = "bm25"
        elif "knn" in self.cfg.prompt_file:
            retriever = "knn"
        elif self.random_sample:
            retriever = "random"
        else:
            retriever = "uprise"
        with open(self.res_file, "a") as f:
            f.write(
                f"retriever: {retriever}; model: {str(self.cfg.model_name)}; task_name: {str(self.cfg.task_name)}; num_prompts: {str(self.num_prompts)}; scores: {str(scores)}"
            )
            if self.cfg.random_sample:
                f.write(
                    f" random_seed: {str(self.cfg.random_seed)}; prompt_pool: {str(self.cfg.prompt_pool_path)}"
                )
            f.write("\n")
        print("saved pred to: ", self.output_file)
        print("saved eval res to: ", self.res_file)

        for path in glob.glob(f"{self.output_file}tmp_*.bin"):
            os.remove(path)
        return data


@hydra.main(config_path="configs", config_name="inference")
def main(cfg):
    logger.info(cfg)
    accelerator = Accelerator()
    inferencer = Inferencer(cfg, accelerator)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        inferencer.forward()
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            inferencer.write_predictions()


if __name__ == "__main__":
    main()
