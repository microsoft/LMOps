import torch
import tqdm
from torch.utils.data import DataLoader
from src.data.collators import DataCollatorWithPaddingAndCuda
import hydra.utils as hu
import hydra
import json
import os
from omegaconf import OmegaConf
import random 
from src.utils.cache_util import BufferedJsonWriter, BufferedJsonReader
from src.utils.metric import metric_dict
from accelerate import Accelerator
import glob
import logging
from transformers import  AutoModelForCausalLM
logger = logging.getLogger(__name__)


class Scorer:
    def __init__(self,cfg, accelerator) -> None:
        self.cfg = cfg
        self.output_train_file = cfg.output_train_file
        self.output_valid_file = cfg.output_valid_file
        os.makedirs(os.path.dirname(cfg.output_train_file), exist_ok=True)

        self.accelerator = accelerator

        self.dataset_reader = hu.instantiate(cfg.dataset_reader)
        self.dataset_reader.shard(accelerator)
        self.dataset_reader.tokenizer.pad_token_id = self.dataset_reader.tokenizer.eos_token_id
        co = DataCollatorWithPaddingAndCuda(tokenizer=self.dataset_reader.tokenizer,device=accelerator.device)
        self.dataloader = DataLoader(self.dataset_reader,batch_size=cfg.batch_size,collate_fn=co)

        self.tokenizer=self.dataset_reader.tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=cfg.model_name, cache_dir=cfg.cache_dir)
        self.model = self.model.half().to(self.accelerator.device)
        self.model = self.model.eval()

        self.max_length=cfg.max_length #used for text completion task,
        self.generate_max_len=cfg.generate_max_len # max seq len to be generated        
        self.option_num=self.dataset_reader.task.class_num

    def choice_losses(self,input_ids, input_atten_mask, loss_mask, labels):
        bsz, option_num, seq_len = input_ids.shape
        if self.option_num is not None: assert option_num == self.option_num
        with torch.no_grad():
            output=self.model(input_ids=input_ids.reshape(bsz*option_num, seq_len), 
                              attention_mask=input_atten_mask.reshape(bsz*option_num, seq_len))
        
        targets=input_ids[:,:,1:].unsqueeze(-1)
        logits=output.logits.reshape(bsz, option_num, seq_len, -1)            
        logits=logits[:,:, :-1, :]
        logit_probs= torch.nn.functional.log_softmax(logits.float(), dim=-1)


        loss_mask=loss_mask[:,:,1:]
        loss= -torch.gather(logit_probs, -1, targets).squeeze(-1) * loss_mask
        loss = loss.sum(-1) / loss_mask.sum(-1)

        preds= torch.argmin(loss, dim=-1)
        normed_loss = torch.nn.functional.normalize(loss, p=1, dim=-1)
        labels_losses = torch.gather(normed_loss, -1, labels).squeeze(-1).tolist()
        accurate_list=(preds==labels.squeeze(-1)).int().tolist()

        return  {
                "labels_losses": labels_losses,
                "accurate_list": accurate_list,
                "preds": preds.tolist(),
                }

    def completion_losses(self,input_ids,input_atten_mask,labels):
        with torch.no_grad():
            answer_start = int(input_atten_mask.shape[-1]) 
            res = self.model.generate(input_ids=input_ids.squeeze(1), #remove the dim for option_num
                                        attention_mask=input_atten_mask.squeeze(1),
                                        eos_token_id=self.dataset_reader.tokenizer.encode("\n")[0],
                                        pad_token_id=self.dataset_reader.tokenizer.pad_token_id,
                                        max_length=min(self.max_length,answer_start+self.generate_max_len),
                                        do_sample=False)
                        
        pred_ids=res[:,answer_start:]
        preds=[]
        for i in range(len(pred_ids)):
            pred=self.dataset_reader.tokenizer.decode(pred_ids[i],skip_special_tokens=True)
            # avoid empty prediction to avoid errors when calculating Rouge metric scores
            if '\n' not in pred: pred+='\n' 
            preds.append(pred)
        compute_metric=metric_dict[self.dataset_reader.task.metric]
        scores=compute_metric(preds=preds, labels=labels, return_list=True)
        return  {
                "labels_losses": [1-score for score in scores],
                "accurate_list": scores,
                "preds": preds
                }

    def completion_logits_loss(self, input_ids, input_atten_mask, loss_mask, labels):
        input_ids=input_ids.squeeze(1)
        input_attention_mask=input_atten_mask.squeeze(1)
        with torch.no_grad():
            output=self.model(input_ids=input_ids, attention_mask=input_attention_mask)

        pad_mask = torch.nn.functional.pad(labels,(input_ids.shape[-1]-labels.shape[-1]-1,0),value=0)
        loss_list = sequence_cross_entropy_with_logits(logits=output.logits,
                                                        targets=entry.input_ids[:,1:],
                                                        weights=pad_mask,
                                                        average=None)
        if len(loss_list.shape)==0:
            loss_list = loss_list.unsqueeze(0)
        for mdata, loss in zip(metadata,loss_list):
            mdata['score'] = float(loss.item())


    def forward(self):
        
        if self.accelerator.is_main_process:
            dataloader = tqdm.tqdm(self.dataloader)
        else:
            dataloader = self.dataloader

        with BufferedJsonWriter(f"{self.output_train_file}tmp_{self.accelerator.device}.bin") as buffer:
            for i,entry in enumerate(dataloader): # dict
                if "stop" in self.cfg and self.cfg.stop==i: # pass stop for debug
                    break
                metadata = entry.pop("metadata")
                if self.dataset_reader.task.class_num==1:
                    one_shot_res=self.completion_losses(
                                                    input_ids=entry.input_ids,
                                                    input_atten_mask=entry.input_atten_mask,
                                                    labels=[x.pop('temp_label') for x in metadata],
                                                    )
                else:
                    one_shot_res=self.choice_losses(
                                                    input_ids=entry.input_ids,
                                                    input_atten_mask=entry.input_atten_mask,
                                                    loss_mask=entry.input_loss_mask,
                                                    labels=entry.labels,
                                                    )
                one_shot_losses=one_shot_res["labels_losses"]
                for i in range(len(metadata)): #bsz
                    metadata[i]['pred']=one_shot_res["preds"][i]
                    metadata[i]['loss']=one_shot_losses[i]
                    metadata[i]['one_shot_acc']=one_shot_res["accurate_list"][i]
                buffer.write(metadata)

    def write_results(self):
        def split_example(entry):
            test_example = {}
            prompt_example = {}
            for key,val in entry.items():
                if key.startswith("test_") or key.startswith("have_"):
                    test_example[key[len("test_"):]] = val
                else:
                    prompt_example[key] = val
            return test_example, prompt_example
        
        data = []
        for path in glob.glob(f"{self.output_train_file}tmp_*.bin"):
            with BufferedJsonReader(path) as f:
                for x in f.read():
                    data.extend(x) 
        logger.info('len(data): %d', len(data))

        qid2info = {}
        one_shot_true=0
        for entry in data:
            qid = ""
            if "have_choosen" in entry: 
                for id in entry['have_choosen']: qid += str(id) + "+"
            qid += str(entry['test_id'])
            if qid not in qid2info:
                test_example, prompt_example = split_example(entry)
                test_example['ctxs'] = [prompt_example]
                qid2info[qid] = test_example
            else:
                _, prompt_example = split_example(entry)
                qid2info[qid]['ctxs'].append(prompt_example)
            one_shot_true+=prompt_example["one_shot_acc"]

        logger.info('task name: %s', self.cfg.task_name)
        logger.info('one_shot_acc: %f', one_shot_true/len(data))

        first_rank_true=0
        example_list = list(qid2info.values())
        for entry in example_list:
            entry['task_name']=self.cfg.task_name
            # rank loss from low to high, the lower the loss, the higher the efficiency of prompt
            entry['ctxs'] = sorted(entry['ctxs'],key = lambda x: x['loss']) 

            # check whether the first-ranked prompt can lead to the gold prediction
            first_rank_true+=entry['ctxs'][0]["one_shot_acc"]

        logger.info('len(example_list): %d',len(example_list))
        logger.info('first_rank_acc: %f', first_rank_true/len(example_list))
        # split the scored data to 90% : 10% for training and validation respectively
        random.Random(42).shuffle(example_list)
        split_ratio=0.9
        n_train=int(len(example_list)*split_ratio)
        with open(self.output_train_file,"w") as writer:
            writer.write(json.dumps(example_list[:n_train], indent=4) + "\n")
        with open(self.output_valid_file,"w") as writer:
            writer.write(json.dumps(example_list[n_train:], indent=4) + "\n")
        for path in glob.glob(f"{self.output_train_file}tmp_*.bin"):
            os.remove(path)


@hydra.main(config_path="configs",config_name="scorer")
def main(cfg):
    logger.info(cfg)
    accelerator = Accelerator()
    scorer = Scorer(cfg, accelerator)
    scorer.forward()
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        scorer.write_results()

if __name__ == "__main__":
    main()

