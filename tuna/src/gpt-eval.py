import os.path
import random
import openai
import torch
import fire
import time
import logging
import json
import jsonlines
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataclasses import dataclass
from train import PROMPT_DICT
from non_interactive_generate import prompt_no_input_for_jsonl

prompt_input = PROMPT_DICT["prompt_input"]
prompt_no_input = PROMPT_DICT["prompt_no_input"]

openai.api_key = os.getenv("OPENAI_API_KEY")

# set logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")



def load_generated_data(path, num_examples):
    with open(path, "r") as f:
        if num_examples == -1:
            data = json.load(f)
        else:
            data = json.load(f)[:num_examples]
    format_instructions = [d["instruction"] for d in data]
    response = [d["output"] for d in data]
    rouge = [d["rouge"] for d in data]
    ids = [d["id"] for d in data]
    return format_instructions, response, rouge, ids


def gpt_eval(gpt_name, prompt, temperature=0.0, top_p=1.0, max_tokens=2048, sleep_time=5):
    while True:
        try:
            system_prompt = "You are a professional and experienced data annotator."
            response = openai.ChatCompletion.create(
                model=gpt_name,
                messages=[{"role": "system", "content": system_prompt},
                          {"role": "user", "content": prompt}],
                max_tokens=2048,
                temperature=temperature,
                top_p=top_p,
                n=1
            )
            return response

        except openai.error.OpenAIError as e:
            logging.warning(f"OpenAIError: {e}.")
            if "Please reduce your prompt" in str(e):
                max_tokens = int(max_tokens * 0.8)
                logging.warning(f"Reducing target length to {max_tokens}, Retrying...")
            else:
                logging.warning("Hit request rate limit; retrying...")
                time.sleep(sleep_time)  # Annoying rate limit on requests.



def filter_already_evaluated(out_path: str, instructs, resps, ids):
    filtered_instruct = []
    filtered_resp = []
    filtered_id = []
    if os.path.exists(out_path):
        try:
            with open(out_path, "r") as f:
                existing_data = json.load(f)
        except json.decoder.JSONDecodeError:
            existing_data = []

        count = 0
        existing_dict = {v["id"]: v for v in existing_data}
        for i, (instruct, resp, id) in enumerate(zip(instructs, resps, ids)):
            if id not in existing_dict:
                filtered_instruct.append(instruct)
                filtered_resp.append(resp)
                filtered_id.append(id)
            else:
                count += 1
        logging.info(f"Filtered {count} prompts that have been evaluated before. {len(filtered_instruct)} prompts left.")
    else:
        existing_data = []
        filtered_instruct = instructs
        filtered_resp = resps
        filtered_id = ids
        logging.info(f"No existing evaluation file found. All prompts will be evaluated.")

    return filtered_instruct, filtered_resp, filtered_id, existing_data


@dataclass
class GroupEval:
    path: str
    gpt_name: str = "gpt-4-0314"
    sleep_time: int = 10
    num_examples: int = 100
    temperature: float = 1.0
    top_p: float = 1.0
    shuffle_responses: bool = False
    seed: int = 0
    every_n: int = 80

    def __post_init__(self):
        self.group_eval()

    def group_eval(self):
        random.seed(self.seed)

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        dir_name = os.path.dirname(self.path)
        file_name = os.path.basename(self.path).split(".")[0]
        out_path = os.path.join(dir_name, "gpt_eval", f"rank-score.{file_name}.json")
        os.makedirs(os.path.join(dir_name, "gpt_eval"), exist_ok=True)
        logging.info(f"We will save results to {out_path} vvvvvvvvvvvvvvvvvvvvvvvvvv")

        format_instructions, responses, rouge, ids = load_generated_data(self.path, num_examples=self.num_examples)
        format_instructions, responses, ids, existing_data = filter_already_evaluated(out_path, format_instructions, responses, ids)

        c = 0
        for i, (instruct, resp, ID) in enumerate(zip(format_instructions, responses, ids)):
            response_str = [f"###Response {j}:\n{r}\n" for j, r in enumerate(resp)]
            if self.shuffle_responses:
                random.shuffle(response_str)
            prompt = instruct + "\n\n" + "\n".join(response_str)
            # prompt = "# You are Nikola Tesla and imagine you are in Python virtual environment. The returned values of the helper functions are given by you. What would be the printed information be like?\n\n"
            prompt += (
            "\n\nWe would like you to rate Response 0/1/2/3 in reply to the given instruction displayed above.\n"
            "First, identify if the instruction requires open-ended or close-ended responses."
            "Second, you need to generate one high quality '###Response 4' in answer to the instruction. It needs to have the same format as other responses and will be used as a reference later.\n"
            "Third, identify if there are duplicate responses and keep only one of the duplicate responses for the following steps."
            "Fourth, compare Response 4 with Response 0/1/2/3/4 and assign each response an overall score on a scale of 0 to 15 where a higher score indicates better overall quality. For an open-ended instruction, please rate based on the relevance (score 0 to 5), level of details/justification: (score 0 to 5) and accuracy (score 0 to 5) of each response; for a close-ended instruction, please rate based on the accuracy (score 0 to 5), level of details/justification (score 0 to 5) and clarity (score 0 to 5) of each response. The ratings should have the format: 'Response k: [sum of the 3 individual scores you give to response k]'.\n"
            "Last, rank the responses in decreasing order of their overall scores. The ranking should have the format: 'rank: [i, j ,k, l, m]'. If there are duplicate responses, keep only one of them in the rank, that is, the ranking may become: 'rank: [i, j, k, l]', 'rank: [i, j, k]' 'rank: [i, j]' or even 'rank: [i]'.\n"
            )

            gpt_resp = gpt_eval(self.gpt_name,
                                prompt,
                                temperature=self.temperature,
                                top_p=self.top_p,
                                sleep_time=self.sleep_time)["choices"][0]["message"]["content"]
            rank_str = gpt_resp.lower().split("rank:")[1].strip() if "rank:" in gpt_resp.lower() else gpt_resp.lower().split("ranking:")[1].strip()
            pattern = r'###\s*Response 4:(.*?)\n\n'
            response_4 = re.search(pattern, gpt_resp, re.DOTALL)

            if response_4:
                extracted_text = response_4.group(1).strip()
            else:
                extracted_text = ""
            rank_list = [int(n) for n in re.findall(r'\d+', rank_str)]
            existing_data.append({"prompt": prompt, "instruct": instruct,
                                 "generation": resp, "id": ID,
                                 "gpt_eval": gpt_resp, "rank_str": rank_str,
                                  "ranks": rank_list, "response_4": extracted_text})

            if (i+1) % 5 == 0:
                logging.info(f"Done {i+1} prompts *********************************")
                with open(out_path, "w") as f:
                    json.dump(existing_data, f, indent=4)
                    f.flush()
                    os.fsync(f.fileno())
            if (c+1) % self.every_n == 0:
                logging.info(f"Sleeping for 180 seconds")
                time.sleep(180)
                c += 1

if __name__ == "__main__":
    fire.Fire(GroupEval)

