Inference code and sample data for LLMA paper.

## Installation

```bash
pip install torch tensorflow transformers sentencepiece tqdm
```
Additionally, you need to get LLaMA model weights and convert to Huggingface format.

## Usage
One Nvidia V100 32GB GPU or better is recommended.

For retrieval-augmented experiments in the paper, run the following:
```bash
# baseline decoding
python decode.py --model_path /path/to/llama_model --input_data_fn ./data/rag.jsonl --type base --forced_decoding --append_docs
# llma decoding
python decode.py --model_path /path/to/llama_model --input_data_fn ./data/rag.jsonl --n 1 --k 20 --type llma --forced_decoding --append_docs
```
Here we run "forced_decoding" which forces the output to be the same as the pre-generated output from davinci-003. The reason, as mentioned in the paper (section 3.2), is that the existing LLaMA models cannot generate high-quality output for RAG.

For experiments without forced decoding, we suggest to run summarization on CNNDM dataset using Alpaca 7B model:
```bash
# baseline decoding
python decode.py --model_path /path/to/alpaca_model --input_data_fn ./data/cnndm.jsonl --type base
# llma decoding
python decode.py --model_path /path/to/alpaca_model --input_data_fn ./data/cnndm.jsonl --n 1 --k 20 --type llma
```
