import asyncio
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.lm_openai_gpt3 import (
    GPT3Client,
    IncrementalOpenAIGPT3,
)
import json
import more_itertools
import tqdm
import hydra
import os
from DPR.dpr.utils.tasks import task_map
from src.utils.metric import compute_scores
import logging

logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="inference_openai")
def main(cfg):
    print(cfg)
    client = GPT3Client(api_key=os.environ["OPENAI_TOKEN"])
    lm = IncrementalOpenAIGPT3(
        client=client, engine=cfg.engine, cache_dir=cfg.cache_dir
    )

    async def get_pred_completion(entry_list, cfg):
        """
        for text completion
        """
        prompt = [x["enc_text"][0].strip() for x in entry_list]
        args = {
            "prompt": prompt,
            "max_tokens": min(cfg.generate_max_len, cfg.max_length - cfg.n_tokens),
            "stop": ["\n"],
            "echo": False,
            "logprobs": 1,
        }
        results = (
            await client.completions_rate_limited(cfg.engine, args)  # type: ignore
        ).json()
        for i, x in enumerate(entry_list):
            x["pred"] = results["choices"][i]["text"]
        return entry_list

    async def get_pred_choice(entry_list):
        """
        for multiple choice
        """
        assert len(entry_list) == 1  # because bsz=1
        entry = entry_list[0]
        res_list = []
        for i in range(len(entry["enc_text"])):
            enc_text = entry["enc_text"][i].strip()
            enc_answer = entry["enc_answer"][i].strip()

            prefix_tokens = lm.tokenizer.encode(enc_text)
            tokenized_labels = lm.tokenizer.encode(enc_answer)

            summed_logprob = await lm.logprob_of_completion(
                prefix_tokens, tokenized_labels
            )  # likelihood
            nll = -summed_logprob  # negative likelihood
            loss = nll / len(tokenized_labels)  # average by length of tokenized_labels
            res_list.append(loss)
        sum_loss = sum(res_list)
        normed_loss = [loss / sum_loss for loss in res_list]
        entry["pred"] = normed_loss.index(min(normed_loss))
        return [entry]

    async def run(data_list):
        task_list = []
        for i, prompt in enumerate(more_itertools.chunked(data_list, cfg.batch_size)):
            if len(data_list[0]["enc_text"]) > 1:  # multiple choice
                assert cfg.batch_size == 1
                task = asyncio.create_task(get_pred_choice(prompt))
            else:  # text completion
                task = asyncio.create_task(get_pred_completion(prompt, cfg))
            task_list.append(task)
        responses = [
            await f
            for f in tqdm.tqdm(asyncio.as_completed(task_list), total=len(task_list))
        ]
        return responses

    def run_main(cfg):
        with open(cfg.prompt_file) as f:
            data_list = json.load(f)
        res = asyncio.run(run(data_list))
        res = list(more_itertools.collapse(res, levels=1))
        os.makedirs(os.path.dirname(cfg.output_file), exist_ok=True)
        os.makedirs(os.path.dirname(cfg.res_file), exist_ok=True)

        with open(cfg.output_file, "w") as f:
            json.dump(res, f)
        task = task_map.cls_dic[cfg.task_name]()
        scores = compute_scores(task.metric, res)
        method = "UPRISE" if int(res[0]["n_prompts"]) > 0 else "0-SHOT"
        logger.info("method: %s", method)
        logger.info("scores: %s", str(scores))
        with open(cfg.res_file, "a") as f:
            f.write(
                f"LLM: {str(cfg.engine)}; task_name: {str(cfg.task_name)}; Method: {method}; scores: {str(scores)}\n"
            )
    run_main(cfg)


if __name__ == "__main__":
    main()
