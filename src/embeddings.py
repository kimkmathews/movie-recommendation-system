import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch

def generate_and_save_embeddings(movies, output_path='outputs/embeddings.npy', model_name='sentence-transformers/all-MiniLM-L6-v2'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    embeddings = []
    for tags in movies['tag']:
        if not tags:
            embeddings.append(np.zeros(384))  # Dimension of all-MiniLM-L6-v2
            continue
        inputs = tokenizer(tags, return_tensors='pt', padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
        embedding = mean_pooling(outputs, inputs['attention_mask']).squeeze().numpy()
        embeddings.append(embedding)
    
    embeddings = np.array(embeddings)
    np.save(output_path, embeddings)
    return embeddings