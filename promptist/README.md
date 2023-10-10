# Promptist: reinforcement learning for automatic prompt optimization

## News
- Sept, 2023: Promptist was accepted by NeurIPS 2023 as Spotlight
- [Demo Release] Dec, 2022: [Demo at Hugging Face Space](https://aka.ms/promptist-demo)
- [Model Release] Dec, 2022: [link](#load-pretrained-model-for-stable-diffusion-v14)
- [Paper Release] Dec, 2022: [Optimizing Prompts for Text-to-Image Generation](https://aka.ms/promptist-paper)

> - Language models serve as a prompt interface that optimizes user input into model-preferred prompts.

> - Learn a language model for automatic prompt optimization via reinforcement learning.

![image](https://user-images.githubusercontent.com/1070872/207856962-02f08d92-f2bf-441a-b1c3-efff1a4b6187.png)


## Load Pretrained Model for [Stable Diffusion v1.4](https://huggingface.co/CompVis/stable-diffusion-v1-4)

You can try the online demo at [https://huggingface.co/spaces/microsoft/Promptist](https://huggingface.co/spaces/microsoft/Promptist).

`[Note]` The online demo at Hugging Face Space is using CPU, so slow generation speed would be expected. Please load the model locally with GPUs for faster generation.

```python
import gradio as grad
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_prompter():
  prompter_model = AutoModelForCausalLM.from_pretrained("microsoft/Promptist")
  tokenizer = AutoTokenizer.from_pretrained("gpt2")
  tokenizer.pad_token = tokenizer.eos_token
  tokenizer.padding_side = "left"
  return prompter_model, tokenizer

prompter_model, prompter_tokenizer = load_prompter()

def generate(plain_text):
    input_ids = prompter_tokenizer(plain_text.strip()+" Rephrase:", return_tensors="pt").input_ids
    eos_id = prompter_tokenizer.eos_token_id
    outputs = prompter_model.generate(input_ids, do_sample=False, max_new_tokens=75, num_beams=8, num_return_sequences=8, eos_token_id=eos_id, pad_token_id=eos_id, length_penalty=-1.0)
    output_texts = prompter_tokenizer.batch_decode(outputs, skip_special_tokens=True)
    res = output_texts[0].replace(plain_text+" Rephrase:", "").strip()
    return res

txt = grad.Textbox(lines=1, label="Initial Text", placeholder="Input Prompt")
out = grad.Textbox(lines=1, label="Optimized Prompt")
examples = ["A rabbit is wearing a space suit", "Several railroad tracks with one train passing by", "The roof is wet from the rain", "Cats dancing in a space club"]

grad.Interface(fn=generate,
               inputs=txt,
               outputs=out,
               title="Promptist Demo",
               description="Promptist is a prompt interface for Stable Diffusion v1-4 (https://huggingface.co/CompVis/stable-diffusion-v1-4) that optimizes user input into model-preferred prompts.",
               examples=examples,
               allow_flagging='never',
               cache_examples=False,
               theme="default").launch(enable_queue=True, debug=True)
```

## Environment Setup

```
alias=`whoami | cut -d'.' -f2`; docker run -it --rm --runtime=nvidia --ipc=host --privileged -v /home/${alias}:/home/${alias} chizewen/pytorch:1.12.1-mpi bash
```

```bash
pip install git+https://github.com/CZWin32768/accelerate.git
pip install pytorch_lightning==1.7.7
pip install transformers==4.23.1
pip install ftfy regex tqdm scipy
pip install git+https://github.com/openai/CLIP.git
pip install --editable ./diffusers
cd trlx
pip install --editable .
cd ..
# please provide the access token of huggingface and your wandb key
```

## Data
We release the data for SFT and RL at [Google Drive](https://drive.google.com/file/d/1EsuYEb9BuinJCdzvQ_gqa_Gu_sTyLWbf/view?usp=drive_link).

## Train Promptist

```
python ./diffusers_examples/quick-start.py

accelerate launch --multi_gpu --machine_rank ${OMPI_COMM_WORLD_RANK} --main_process_ip ${MASTER_ADDR} --main_process_port ${MASTER_PORT} --num_machines 4 --num_processes 32 ./diff_prompter/ppo_prompter.py --data /data_path --gpt_path /supervised_finetuned_gpt_path --trl_config ./diff_prompter/configs/ppo_config_a100_coco_bsz256_kl0.2.yml --checkpoint_dir /ckpt_dir
```

## Reference

If you find this repository useful, please consider citing our work:
```
@inproceedings{promptist,
  title={Optimizing Prompts for Text-to-Image Generation},
  author={Yaru Hao and Zewen Chi and Li Dong and Furu Wei},
  booktitle={Neural Information Processing Systems},
  year={2023}
}
```
