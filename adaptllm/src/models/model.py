from transformers import AutoModelForCausalLM

def get_model(**kwargs):
    if kwargs.pop("model_parallel")==True:
        model = AutoModelForCausalLM.from_pretrained(device_map="auto", **kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(**kwargs)
    return model