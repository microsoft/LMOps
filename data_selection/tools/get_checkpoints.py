from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel

name = "KoboldAI/fairseq-dense-125M"
save_name = "faireq/125M"

tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
tokenizer.save_pretrained(f"checkpoints/{save_name}/")

model = AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True)
model.save_pretrained(f"checkpoints/{save_name}/", safe_serialization=False)