from transformers import AutoTokenizer, AutoModel
import torch

def get_embedding(text, cfg_proj):
    # AutoTokenizer will automatically load the appropriate tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg_proj.embedding_model)

    # Load pre-trained BERT model
    model = AutoModel.from_pretrained(cfg_proj.embedding_model)

    # Tokenize the input text
    tokens = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)

    # Forward pass through the BERT model
    outputs = model(**tokens)

    if cfg_proj.embedding_layer == "last_hidden_state":
        last_hidden_state = torch.mean(outputs.last_hidden_state, dim = 1).detach().cpu().numpy()
        return last_hidden_state
    else:
        pooler_output = outputs.pooler_output.detach().cpu().numpy()
        return pooler_output

