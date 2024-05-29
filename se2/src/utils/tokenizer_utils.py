

def get_length(tokenizer, text):
        tokenized_example = tokenizer.encode_plus(text,truncation=False,return_tensors='pt')
        return int(tokenized_example.input_ids.squeeze().shape[0])