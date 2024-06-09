from transformers import AutoTokenizer
def get_tokenizer(configs):
    if 'facebook/esm2' in configs.encoder_name:
        tokenizer = AutoTokenizer.from_pretrained(configs.encoder_name)
    else:
        raise ValueError("Wrong tokenizer specified.")
    return tokenizer