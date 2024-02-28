import re
import json
import logging
import random
import os
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Union, Any, Tuple
import wandb
from accelerate.utils import save_fsdp_model
from peft import LoraConfig, TaskType, get_peft_model, AdaLoraConfig, LoHaConfig, LoKrConfig
from torch import nn
sys.path.append('../../../../')
from myloralib.reslora import ResLoraConfig, ResLoraModel
from myloralib.layers import ResLoraLinear
from preprocess import get_dataset, BaseProcessor
from tqdm import tqdm
import datasets
import numpy as np
from datasets import load_dataset, DatasetDict, Dataset, concatenate_datasets
import torch
import transformers
from transformers import (
    AutoConfig,
    LlamaConfig, GenerationConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    HfArgumentParser,
    MBartTokenizer,
    MBartTokenizerFast,
    LlamaTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
    AutoModelForCausalLM, DataCollatorWithPadding, DataCollatorForSeq2Seq, Trainer, TrainingArguments, TrainerState,
    TrainerControl,
)
from transformers.trainer_utils import IntervalStrategy
from transformers.trainer_callback import TrainerCallback


logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    method: Optional[str] = field(
        default="lora",
        metadata={
            "help": "you can choose follow method: ft, lora, adapter, adalora, reslora"
        }
    )
    lora_alpha: Optional[int] = field(
        default=16,
        metadata={"help": "LoRA alpha"},
    )
    lora_r: Optional[int] = field(
        default=8,
        metadata={"help": "LoRA r"},
    )
    res_flag: Optional[int] = field(
        default=0
    )
    merge_flag: Optional[int] = field(
        default=0
    )
    pre_num: Optional[int] = field(
        default=0
    )


@dataclass
class DataTrainingArguments:
    debug_flag: Optional[int] = field(default=0, metadata={"help": "whether or not use wandb"})
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={"help": "The maximum total input sequence length after tokenization for source text."},
    )
    max_target_length: Optional[int] = field(
        default=512,
        metadata={"help": "The maximum total input sequence length after tokenization for target text."},
    )
    result_path: Optional[str] = field(
        default="temp_result",
        metadata={"help": "The path to save result."},
    )
    task_name: Optional[str] = field(
        default="gsm8k",
        metadata={"help": "The name of the task to train"},
    )



class EvalEpochIntervalCallback(TrainerCallback):

    def on_epoch_end(self, args, state, control, **kwargs):
        global total_epoch
        total_epoch += 1
        epoch = round(state.epoch)

        if (epoch % 5 == 0):
            control.should_save = True
        else:
            control.should_save = False

        if (args.logging_strategy == IntervalStrategy.EPOCH):
            control.should_log = True

        control.should_evaluate = True

        return control


class ResTrainer(Seq2SeqTrainer):
    def __init__(self, *args, resmodel=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.resmodel = resmodel

    def save_model(self, *args, **kwargs):
        if self.resmodel.resconfig.res_flag > 0:
            self.resmodel.unconcat_loras()
            super().save_model(*args, **kwargs)
            self.resmodel.concat_loras()
        else:
            super().save_model(*args, **kwargs)

    def evaluate(self, *args, **kwargs) -> Dict[str, float]:
        if (self.resmodel.resconfig.res_flag == 1 or self.resmodel.resconfig.res_flag == 3) and self.resmodel.resconfig.merge_flag > 0:
            self.resmodel.calculate_froc()
        return super().evaluate(*args, **kwargs)
        # else:
        #     return super().evaluate(*args, **kwargs)



my_max_input_length = 0
total_epoch = 0

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = local_rank
    world_size = int(os.environ["WORLD_SIZE"])

    if global_rank == 0:
        if data_args.debug_flag:
            wandb.init(mode="disabled")

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)
    random.seed(training_args.seed)

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        max_length=data_args.max_source_length,
        # add_eos_token=True,
    )
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.pad_token_id = tokenizer.unk_token_id
    # tokenizer.pad_token = tokenizer.bos_token
    # tokenizer.pad_token_id = tokenizer.bos_token_id
    # tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    logger.info(config)
    logger.info(len(tokenizer))
    logger.info(tokenizer.pad_token)
    logger.info(tokenizer.pad_token_id)
    logger.info(tokenizer.bos_token)
    logger.info(tokenizer.eos_token)
    logger.info(tokenizer.padding_side)
    logger.info(tokenizer.truncation_side)

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    logger.info(model)



    # def process(data):
    #     global my_max_input_length
    #     prompt = demo + data['question']
    #     inputs = tokenizer(prompt)
    #     ans = data['answer']
    #     ans = ans.replace('####', 'Therefore, the answer is')
    #     labels = tokenizer(ans)
    #     labels['input_ids'] = labels['input_ids'][1:]
    #
    #     data['input_ids'] = inputs['input_ids'] + labels['input_ids']
    #     data['labels'] = [-100] * len(inputs['input_ids']) + labels['input_ids']
    #     data['attention_mask'] = [1] * len(data['input_ids'])
    #     assert (len(data['input_ids']) == len(data['labels'])), "Error: input_ids and labels have different length!"
    #     my_max_input_length = max(my_max_input_length, len(data['input_ids']))
    #
    #     return data
    # my_dataset = load_dataset("gsm8k", name='main')
    # train_dataset = my_dataset['train']
    # test_dataset = my_dataset['test']
    #
    # ids = random.sample([i for i in range(len(train_dataset))], 3)
    # demo = ''
    # # pattern = PROMPT + STOP_WORD
    # for idx in ids:
    #     data = train_dataset[idx]
    #     problem = data['question']
    #     solution = data['answer']
    #     answer = solution.split('####')[-1].strip()
    #     solution = solution.split('####')[0].strip()
    #     demo = demo + f'{problem}\n{solution} Therefore, the answer is {answer}'

    if global_rank == 0 and not os.path.exists(data_args.result_path):
            os.makedirs(data_args.result_path, exist_ok=True)
    train_dataset, test_dataset, processor = get_dataset(data_args.task_name, tokenizer, data_args.result_path, data_args.max_source_length)
    add_train_dataset, add_test_dataset, add_processor = get_dataset('gsm8k', tokenizer, data_args.result_path, data_args.max_source_length)
    add_train_dataset2, add_test_dataset2, add_processor2 = get_dataset('svamp', tokenizer, data_args.result_path, data_args.max_source_length)

    add_processor.max_input_length = 0
    add_train_dataset = add_train_dataset.map(add_processor.process_train, load_from_cache_file=False)
    add_train_dataset = add_processor.post_process(add_train_dataset)

    if test_dataset is None or len(test_dataset) == 0:
        add_test_dataset = add_test_dataset.map(add_processor.process_test, load_from_cache_file=False)
        test_dataset = add_test_dataset
    else:
        test_dataset = test_dataset.map(processor.process_test, load_from_cache_file=False)

    add_processor2.max_input_length = 0
    add_train_dataset2 = add_train_dataset2.map(add_processor2.process_train, load_from_cache_file=False)
    add_train_dataset2 = add_processor2.post_process(add_train_dataset2)

    # add_train_dataset = BaseProcessor.sort_by_length(add_train_dataset)
    logger.info(f"Max Add Train Input Length: {add_processor.max_input_length}")
    # add_processor.max_input_length = 0
    # add_test_dataset = add_test_dataset.map(add_processor.process_test, load_from_cache_file=False)
    # logger.info(f"Max Add Test Input Length: {add_processor.max_input_length}")

    # logger.info('--------------- Raw Dataset ---------------')
    # logger.info(train_dataset)
    # logger.info(test_dataset)

    if data_args.debug_flag:
        add_train_dataset = add_train_dataset.select(range(100))
        add_train_dataset2 = add_train_dataset2.select(range(100))
        train_dataset = train_dataset.select(range(100))
        test_dataset = test_dataset.select(range(100))
    processor.max_input_length = 0
    train_dataset = train_dataset.map(processor.process_train, load_from_cache_file=False)
    if len(train_dataset) > 40_000:
        final_train_num = 40_000
        print(f"Warning: Auto set final train num to 40_000, because of the length {len(train_dataset)} of train dataset.")
    else:
        final_train_num = -1
    train_dataset = processor.post_process(train_dataset, final_train_num)
    logger.info(f"Max Train Input Length: {processor.max_input_length}")

    logger.warning(f"before add: {len(train_dataset)}")
    final_train_dataset = concatenate_datasets([train_dataset, add_train_dataset])
    logger.warning(f"after add: {len(final_train_dataset)}")
    # exit(-1)

    processor.max_input_length = 0

    logger.info(f"Max Test Input Length: {processor.max_input_length}")
    logger.info('--------------- Processed Dataset ---------------')
    logger.info(train_dataset)
    logger.info(test_dataset)
    # test_dataset = processor.sort_by_length(test_dataset)
    processor.test_dataset = test_dataset

    demo = "Sample:\n"
    for i in range(3):
        demo += f"{train_dataset[i]['input_text']}\n{train_dataset[i]['label_text']}\n\n"
    logger.info(f'\n{demo}')
    demo = ''

    data_collator = DataCollatorForSeq2Seq(tokenizer, model, pad_to_multiple_of=8)

    logger.info(f"training dataset: {train_dataset}")
    logger.info(f"test dataset: {test_dataset}")
    logger.info(f"add training dataset: {add_train_dataset}")
    logger.info(f"add training dataset2: {add_train_dataset2}")
    logger.info(f"final training dataset: {final_train_dataset}")

    # best_em = 0.0
    # def compute_metrics(eval_preds):
    #     nonlocal best_em
    #     preds, labels = eval_preds
    #     num_correct, total_problem = 0, len(preds)
    #     assert len(preds) == len(labels)
    #
    #     for p, l in zip(preds, labels):
    #         p = np.where(p != -100, p, tokenizer.pad_token_id)
    #         p = tokenizer.decode(p, skip_special_tokens=True)
    #         l = np.where(l != -100, l, tokenizer.pad_token_id)
    #         l = tokenizer.decode(l, skip_special_tokens=True)
    #         print(f"p: {p}\n l: {l}")
    #         if 'Therefore, the answer is' in p:
    #             p = p.split('Therefore, the answer is')[1].strip()
    #         else:
    #             p = ""
    #         assert 'Therefore, the answer is' in l
    #         l = l.split('Therefore, the answer is')[1].strip()
    #         if p == l:
    #             num_correct += 1
    #
    #     result = round(num_correct / total_problem * 100, 2)
    #     best_em = max(best_em, result)
    #     logger.info(f'Best Exactly Match: {best_em}')
    #     return {'EM': result}

    if training_args.num_train_epochs == 0:
        if len(train_dataset) <= 2_000:
            training_args.num_train_epochs = 40
        elif len(train_dataset) <= 10_000:
            training_args.num_train_epochs = 20
        else:
            training_args.num_train_epochs = 8
        logger.info(f"Auto set training epochs to {training_args.num_train_epochs}, because of the length of {len(train_dataset)}.")

    if model_args.method == "ft":
        print("Info: This is full finetune method.")
    elif model_args.method == "lora":
        pconfig = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1,
        )
        model = get_peft_model(model, pconfig)
        model.print_trainable_parameters()
    elif model_args.method == "adalora":
        pconfig = AdaLoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1,
        )
        model = get_peft_model(model, pconfig)
        model.print_trainable_parameters()
    # elif model_args.method == "ptuning":
    #     pconfig = PromptEncoderConfig(
    #         task_type=TaskType.CAUSAL_LM,
    #         encoder_reparameterization_type="MLP",
    #         encoder_hidden_size = 768,
    #     )
    #     model = get_peft_model(model, pconfig)
    #     model.print_trainable_parameters()
    elif model_args.method == "loha":
        pconfig = LoHaConfig(
            task_type=TaskType.CAUSAL_LM,
            r=model_args.lora_r,
            alpha=model_args.lora_alpha,
            target_modules=["q_proj", "v_proj"],
            rank_dropout=0.1,
        )
        model = get_peft_model(model, pconfig)
        model.print_trainable_parameters()
    elif model_args.method == "lokr":
        pconfig = LoKrConfig(
            task_type=TaskType.CAUSAL_LM,
            r=model_args.lora_r,
            alpha=model_args.lora_alpha,
            target_modules=["q_proj", "v_proj"],
            rank_dropout=0.1,
        )
        model = get_peft_model(model, pconfig)
        model.print_trainable_parameters()
    elif model_args.method == "reslora":
        if model_args.merge_flag:
            assert model_args.res_flag, "Error: merge flag must be used with res flag 1."
        if model_args.res_flag == 2 or model_args.res_flag == 3:
            assert model_args.pre_num != 0, "Error: pre num must be used when res flag 2."
        resconfig = ResLoraConfig(rank=model_args.lora_r, lora_alpha=model_args.lora_alpha,
                                 target_modules="q.v", lora_dropout=0.1, res_flag=model_args.res_flag,
                                 merge_flag=model_args.merge_flag, pre_num=model_args.pre_num)
        resmodel = ResLoraModel(model, resconfig, epochs=training_args.num_train_epochs)
        logger.info(
            f"reslora config: {resconfig.to_json_string()}\n"
        )
        logger.info(
            f"Trainable params: {sum(p.numel() for p in resmodel.parameters() if p.requires_grad) / 1_000_000:.2f}M")
    else:
        print("Error: Wrong method!")
        raise NotImplementedError

    generation_config = GenerationConfig(temperate=0.95, max_length=data_args.max_target_length,
                                         eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
                                         min_new_tokens=10, remove_invalid_values=True)
    training_args.generation_config = generation_config

    if global_rank == 0 and not os.path.exists(data_args.result_path):
        os.makedirs(data_args.result_path)

    if global_rank == 0:
        Total_params = 0
        Trainable_params = 0
        NonTrainable_params = 0


        for param in model.parameters():
            mulValue = np.prod(param.size())  # 使用numpy prod接口计算参数数组所有元素之积
            Total_params += mulValue  # 总参数量
            if param.requires_grad:
                Trainable_params += mulValue  # 可训练参数量
            else:
                NonTrainable_params += mulValue  # 非可训练参数量
        logger.info(f"trainable params: {Trainable_params / 1_000_000:.2f}M, non-trainable params: {NonTrainable_params / 1_000_000:.2f}M, total params: {Total_params / 1_000_000:.2f}M")

    if model_args.method == "reslora":
        raw_output_dir = training_args.output_dir
        logger.info(training_args)
        if global_rank == 0:
            if data_args.debug_flag:
                wandb.init(mode="disabled")
            else:
                names = training_args.output_dir.split('/')
                tags = [names[-2], names[-3], model_args.method]
                wandb.init(project=f"reslora_nlg", tags=tags, name="_".join(tags))
        training_args.output_dir = f"{raw_output_dir}"
        logger.warning(f"Training epoch: {training_args.num_train_epochs}")
        print(f"now model is in lora {resmodel.new_epoch()}")
        logger.info(training_args)
        logger.info(test_dataset)
        trainer = ResTrainer(
            model=resmodel.wrapped_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset if training_args.do_eval else None,
            compute_metrics=processor.compute_metrics,
            tokenizer=tokenizer,
            data_collator=data_collator,
            resmodel=resmodel,
        )

        # Training
        if training_args.do_train:
            checkpoint = None
            train_result = trainer.train(resume_from_checkpoint=checkpoint)
            metrics = train_result.metrics
            metrics["train_samples"] = len(train_dataset)

            trainer.save_model()  # Saves the tokenizer too for easy upload

            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_state()

        # # Evaluation
        # if training_args.do_eval:
        #     logger.info("*** Evaluate ***")
        #
        #     # Loop to handle MNLI double evaluation (matched, mis-matched)
        #     tasks = [data_args.task_name]
        #     eval_datasets = [eval_dataset]
        #     if data_args.task_name == "mnli":
        #         tasks.append("mnli-mm")
        #         eval_datasets.append(datasets["validation_mismatched"])
        #
        #     for eval_dataset, task in zip(eval_datasets, tasks):
        #         # trainer.model.set_rank(rank=rank) # set the test rank
        #         metrics = trainer.evaluate(eval_dataset=eval_dataset)
        #         max_val_samples = data_args.max_val_samples if data_args.max_val_samples is not None else len(
        #             eval_dataset)
        #         metrics[f"eval_samples"] = min(max_val_samples, len(eval_dataset))
        #
        #         trainer.log_metrics(f"eval", metrics)
        #         trainer.save_metrics(f"eval", metrics)
        wandb.finish()
    else:
        class TrainerAdapterCallback(TrainerCallback):

            def __init__(self):
                self.global_step = 0

            # offload original_modules to cpu, to save memory
            def on_train_begin(self, _args, state, control, **kwargs):
                # if hasattr(model, 'set_active_adapters'):
                #     model.set_active_adapters(model.adapters.keys(), offload='cpu')
                if model_args.method == 'adalora':
                    # model.peft_config['default'].total_step = state.max_steps

                    def zero_grad(_self, *args, **kwargs):
                        _self.update_and_allocate(self.global_step + 1)
                        _self._zero_grad(*args, **kwargs)

                    model._zero_grad = model.zero_grad
                    model.zero_grad = zero_grad
                    # model.zero_grad = types.MethodType(zero_grad, model)

            def on_step_end(self, _args, state, control, **kwargs):
                if model_args.method == 'adalora':
                    self.global_step = state.global_step

        if global_rank == 0:
            if data_args.debug_flag:
                wandb.init(mode="disabled")
            else:
                names = training_args.output_dir.split('/')
                tags = [names[-2], names[-3], model_args.method, f"lora_{0}"]
                wandb.init(project=f"reslora_nlg", tags=tags, name="_".join(tags))

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=processor.compute_metrics,
            callbacks=[],
        )

        if model_args.method == "adalora":
            trainer.add_callback(TrainerAdapterCallback())


        # Training
        if training_args.do_train:
            checkpoint = None
            # if last_checkpoint is not None:
            #     checkpoint = last_checkpoint
            # elif os.path.isdir(model_args.model_name_or_path):
            #     # Check the config from that potential checkpoint has the right number of labels before using it as a
            #     # checkpoint.
            #     if AutoConfig.from_pretrained(model_args.model_name_or_path).num_labels == num_labels:
            #         checkpoint = model_args.model_name_or_path

            train_result = trainer.train(resume_from_checkpoint=checkpoint)
            metrics = train_result.metrics
            metrics["train_samples"] = len(train_dataset)

            trainer.save_model()  # Saves the tokenizer too for easy upload

            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_state()

    # if training_args.do_eval:
    #     trainer.evaluate()

    # # Evaluation
    # if training_args.do_eval:
    #     logger.info("*** Evaluate ***")
    #
    #     # Loop to handle MNLI double evaluation (matched, mis-matched)
    #     tasks = [data_args.task_name]
    #     eval_datasets = [eval_dataset]
    #     if data_args.task_name == "mnli":
    #         tasks.append("mnli-mm")
    #         eval_datasets.append(datasets["validation_mismatched"])
    #
    #     for eval_dataset, task in zip(eval_datasets, tasks):
    #         for rank in range(0, model_args.lora_r):
    #             print(f'--> eval rank={rank}')
    #             # trainer.model.set_rank(rank=rank) # set the test rank
    #             metrics = trainer.evaluate(eval_dataset=eval_dataset)
    #
    #             max_val_samples = data_args.max_val_samples if data_args.max_val_samples is not None else len(
    #                 eval_dataset)
    #             metrics[f"eval_samples_r{rank}"] = min(max_val_samples, len(eval_dataset))
    #
    #             trainer.log_metrics(f"eval_r{rank}", metrics)
    #             trainer.save_metrics(f"eval_r{rank}", metrics)

    # # Predict
    # predict_results = trainer.predict(test_dataset, metric_key_prefix="predict")
    # logger.info(predict_results)


if __name__ == "__main__":
    main()