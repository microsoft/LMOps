import logging
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from datasets import load_dataset
import fire
import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import json
import jsonlines
import time
import deepspeed
from rouge_score import rouge_scorer
from typing import List, Dict, Sequence
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from train import PROMPT_DICT
from dataclasses import dataclass
import cProfile
import pstats

prompt_input = PROMPT_DICT["prompt_input"]
prompt_no_input = PROMPT_DICT["prompt_no_input"]
prompt_no_input_for_jsonl = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{text}\n\n### Response:\n"
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)



def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer, ids: Sequence[str]) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = [tokenized.input_ids[0] for tokenized in tokenized_list]
    return dict(input_ids=input_ids, raw_strings=strings, ids=ids)

def tokenize_function(examples, tokenizer):
    prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
    if 'input' in examples:
        sources = [
            prompt_input.format_map(dict(instruction=instruction, input=input)) if input != ""
            else prompt_no_input.format_map(dict(instruction=instruction))
            for instruction, input in zip(examples['instruction'], examples['input'])
        ]
    else:
        sources = [
            prompt_no_input.format_map(dict(instruction=instruction))
            for instruction in examples['instruction']
        ]
    ids = [index for index in examples["id"]]
    data_dict = _tokenize_fn(sources, tokenizer, ids)
    return data_dict

@dataclass
class DataCollator(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        raw_strings = [instance["raw_strings"] for instance in instances]
        ids = [instance["ids"] for instance in instances]
        input_ids = [instance["input_ids"] for instance in instances]
        input_ids = [torch.tensor(x).flip(dims=(0,)) for x in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        input_ids = input_ids.flip(dims=(1,))
        return dict(
            input_ids=input_ids,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            raw_strings=raw_strings,
            ids=ids,
        )

def prepare_dataloader(dataset, batch_size, shuffle=False, local_rank=-1, collate_fn=None):
    sampler = DistributedSampler(dataset,
                                 shuffle=shuffle,
                                 num_replicas=dist.get_world_size(),
                                 rank=local_rank)
    assert local_rank == dist.get_rank(), f"ranks are not equal! local_rank: {local_rank}, dist.get_rank(): {dist.get_rank()}"
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn)
    return dataloader

def check_duplicity(data_str: List[str], threshold: float = 0.8):
    # Compute the rougeL score of each pair of strings in the list. If the score is greater than a threshold, then
    # abort this string and return a False flag.
    unique_enough = True
    rl = []
    for i in range(len(data_str)-1):
        rl.extend([scorer.score(data_str[i], data_str[j])["rougeL"].fmeasure for j in range(i+1, len(data_str))])
        if max(rl) >= threshold:
            unique_enough = False
    return unique_enough, sum(rl) / len(rl)

def generate(model, tokenizer, tokenized_batch, threshold, max_trials, temperature_increment ,**kwargs):

    bs = tokenized_batch["input_ids"].shape[0]
    responses = [[] for _ in range(bs)]
    rouge_scores = [1.0 for _ in range(bs)]

    num_return_sequences = kwargs["num_return_sequences"]
    temperature = kwargs.get("temperature", 1.0)
    indices = list(range(bs))
    finished = [False] * bs

    input_length = tokenized_batch["input_ids"].shape[1]
    while not all(finished) and max_trials > 0:
        max_trials -= 1

        with torch.no_grad():
            output = model.generate(
                input_ids=tokenized_batch["input_ids"][indices],
                attention_mask=tokenized_batch["attention_mask"][indices],
                do_sample=kwargs.get("do_sample", False),
                top_p=kwargs.get("top_p", 1.0),
                temperature=temperature,
                num_beams=kwargs.get("num_beams", 1),
                num_beam_groups=kwargs.get("num_beam_groups", 1),
                max_new_tokens=kwargs.get("max_new_tokens", 512),
                num_return_sequences=num_return_sequences,
                return_dict_in_generate=True,
            )
        # decode the output to strings
        decoded_strings = tokenizer.batch_decode(output.sequences[:, input_length:], skip_special_tokens=True)

        # check the duplicity of the generated strings, pop the indices of the finished strings
        for i in range(len(indices)):
            unique_enough, rl_score = check_duplicity(
                decoded_strings[i*num_return_sequences:(i+1)*num_return_sequences],
                threshold=threshold)

            if rl_score < rouge_scores[indices[i]] or not responses[indices[i]]:
                rouge_scores[indices[i]] = rl_score
                responses[indices[i]] = decoded_strings[i*num_return_sequences:(i+1)*num_return_sequences]
            if unique_enough:
                finished[indices[i]] = True

        indices = [i for i in indices if not finished[i]]
        temperature += temperature_increment

    return responses, rouge_scores


def main(model_name_or_path: str,
         data_path: str,
         sample: bool = False,
         top_p: float = 1.0,
         temperature: float = 1.0,
         num_beams: int = 1,
         num_beam_groups: int = 1,
         num_return_sequences: int = 1,
         batch_size: int = 1,
         max_new_tokens: int = 512,
         seed: int = 0,
         num_examples: int = 100,
         output_path: str = "",
         tp_size: int = 0,
         rouge_threshold: float = 0.8,
         max_trials: int = 3,
         temperature_increment: float = 0.1,
         local_rank: int = 0,
    ):
    # set random seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # deepspeed.init_distributed()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # make output directory
    os.makedirs(output_path, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    tp_size = tp_size if tp_size > 0 else torch.cuda.device_count()
    ds = deepspeed.init_inference(model,
                                  tp={"tp_size": tp_size},
                                  dtype=torch.half,
                                  checkpoint=None,
                                  replace_with_kernel_inject=True)
    model = ds.module
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.padding_side = "left"
    model.eval()

    raw_dataset = load_dataset("json", data_files=data_path, split="train")
    generation_dataset = raw_dataset.map(
        tokenize_function,
        fn_kwargs={"tokenizer": tokenizer},
        batched=True,
        batch_size=3000,
        num_proc=32,
        remove_columns=raw_dataset.column_names,
        desc="Running tokenizer on dataset",
    )
    collator = DataCollator(tokenizer)
    generate_dataloader = prepare_dataloader(generation_dataset,
                                             batch_size,
                                             False,
                                             local_rank=local_rank,
                                             collate_fn=collator)

    model_responses = []
    counter = 0

    model_name = os.path.basename(model_name_or_path) if not model_name_or_path.endswith("/") \
        else os.path.basename(os.path.dirname(model_name_or_path))
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_path = os.path.join(output_path, f"rank-{local_rank}.{model_name}.seed-{seed}.{timestamp}.json")

    with open(output_path, "w") as f:

        for batch in generate_dataloader:
            batch["input_ids"] = batch["input_ids"].to(device)
            batch["attention_mask"] = batch["attention_mask"].to(device)
            batch_responses, rs = generate(
                model,
                tokenizer,
                batch,
                rouge_threshold,
                max_trials=max_trials,
                temperature_increment=temperature_increment,
                do_sample=sample,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                num_beam_groups=num_beam_groups,
                max_new_tokens=max_new_tokens,
                num_return_sequences=num_return_sequences,
            )

            complete_instance = [{"instruction": s, "output": br, "rouge": rs[i], "id": index}
                                     for i, (s, br, index) in enumerate(zip(batch["raw_strings"], batch_responses, batch["ids"]))]
            model_responses.extend(complete_instance)

            counter += len(complete_instance)
            counter_tensor = torch.tensor([counter], dtype=torch.int, device=f"cuda:{local_rank}")

            dist.all_reduce(counter_tensor, op=dist.ReduceOp.SUM)
            if local_rank == 0:
                c = counter_tensor.item()
                logging.info(f"Processed {c} examples")

            for instance in complete_instance:
                json_str = json.dumps(instance)
                f.write(json_str + "\n")


if __name__ == "__main__":
    fire.Fire(main)
