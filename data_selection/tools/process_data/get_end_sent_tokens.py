from transformers import AutoTokenizer
import json
from tqdm import tqdm

# characters that will appear at the end of a sentence
ent_sent_chars = ".!?;\n"
branket_chars = ["[]", "()", "\{\}", "<>"]
name = "fairseq"
tokenizer = AutoTokenizer.from_pretrained("checkpoints/fairseq/125M/")

# name = "mistral"
# tokenizer = AutoTokenizer.from_pretrained("checkpoints/mistral/160M/")

# name = "gpt2"
# tokenizer = AutoTokenizer.from_pretrained("checkpoints/gpt2/base/")

# name = "llama3_1"
# tokenizer = AutoTokenizer.from_pretrained("checkpoints/llama3.1-8B/")

# name = "qwen"
# tokenizer = AutoTokenizer.from_pretrained("checkpoints/qwen/")

print(len(tokenizer))

ent_sent_tokens = []
for i in tqdm(range(len(tokenizer))):
    for ent_sent_char in ent_sent_chars:
        token = tokenizer.decode([i])
        if ent_sent_char in token:
            valid = True
            for bracket_char in branket_chars:
                if (bracket_char[0] in token) and (bracket_char[1] not in token):
                    valid = False
                    break
            if valid:
                ent_sent_tokens.append(i)
                break

ent_sent_tokens.append(tokenizer.eos_token_id)

with open(f"tools/process_data/end_sent_token_{name}.json", "w") as f:
    json.dump(ent_sent_tokens, f, indent=4)

for ent_sent_token in ent_sent_tokens:
    print(tokenizer.convert_ids_to_tokens(ent_sent_token), ent_sent_token)
