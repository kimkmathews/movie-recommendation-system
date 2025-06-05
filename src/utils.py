import pandas as pd
import json
import numpy as np

def load_movies(movies_path='data/movies.dat'):
    movies = pd.read_csv(movies_path, sep="::", engine="python", header=None, 
                         names=["movieId", "title", "genres"])
    return movies['title'].tolist()

def load_embeddings(embeddings_path='outputs/embeddings.npy'):
    return np.load(embeddings_path)

def load_similarities(similarities_path='outputs/similarities.json'):
    with open(similarities_path, 'r') as f:
        return json.load(f)