import os
import torch
import json
import time
import wandb
import datasets
import lm_eval

from utils import print_rank, save_rank, BOS_MODELS
from train_eval_utils.base_evaluator import BaseEvaluator


class LMHarnessEvaluator(BaseEvaluator):
    def __init__(self, args, ds_config, device):
        super().__init__(args, ds_config, device)

    def setup(self):
        super().setup()
        datasets.utils.logging.set_verbosity_error()
        self.print_and_save(f"Evaluate on {self.args.eval_data_names}")
        self.tasks = self.args.eval_data_names.split(",")
        self.model_args = {
            "add_bos_token": (self.args.model_type in BOS_MODELS),
            "dtype": "float" if self.args.fp32 else "half",
        }

    def extract_results(self, results):
        for k in results:
            if "acc_norm,none" in results[k]:
                results[k] = results[k]["acc_norm,none"]
            elif "acc,none" in results[k]:
                results[k] = results[k]["acc,none"]
            else:
                raise ValueError(f"Metric for {k} must contain acc_norm or acc")
        results["avg"] = sum(results.values()) / len(results)
        return results

    def _evaluate(self):
        self.print_and_save(f"Evaluating {self.model_path}")
        self.print_and_save(f"Results will be saved to {self.output_path}")
        time_tag = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

        model_args = {
            "pretrained": self.model_path,
            **self.model_args
        }
        results = lm_eval.simple_evaluate(  # call simple_evaluate
            model="hf",
            model_args=model_args,
            tasks=self.tasks,
            batch_size=self.args.eval_batch_size,
            bootstrap_iters=0 if self.args.eval_no_calc_stderr else 100000,
            num_fewshot=self.args.eval_shot
        )

        if self.dp_rank == 0:
            res = self.extract_results(results["results"])
            if wandb.run is not None:
                wandb.log(res, step=self.global_steps)
            print_rank(json.dumps(res, indent=4))
            save_rank("eval | {} | {}".format(time_tag, str(self.tasks)),
                      os.path.join(self.output_path, "log.txt"))
            save_rank(json.dumps(res, indent=4),
                      os.path.join(self.output_path, "log.txt"))
            save_rank("\n\n\n", os.path.join(self.output_path, "log.txt"))
            torch.save(results, os.path.join(self.output_path, f"results_{time_tag}.pt"))

        return results
