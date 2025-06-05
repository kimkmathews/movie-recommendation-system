import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def precompute_similarities(movies, embeddings, output_path='outputs/similarities.json', top_k=50):
    # Compute cosine similarities
    sim_scores = cosine_similarity(embeddings)
    
    # Get top-k similar movies for each movie
    similarities = {}
    for idx, movie_id in enumerate(movies['movieId']):
        sim_scores_idx = sim_scores[idx]
        top_indices = np.argsort(sim_scores_idx)[::-1][1:top_k+1]  # Exclude self
        top_scores = sim_scores_idx[top_indices]
        similarities[movie_id] = {
            'similar_movies': movies.iloc[top_indices]['movieId'].tolist(),
            'scores': top_scores.tolist()
        }
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(similarities, f)
    
    return similarities

def load_recommendations(movie_title, movies, similarities, top_n=10, min_rating=3.0, year_range=(1900, 2025)):
    # Find movie ID
    print(f"Searching for movie: '{movie_title}'")
    # Try exact match first
    idx = movies[movies['title'] == movie_title].index
    if idx.empty:
        # Fallback to case-insensitive substring match with escaped special characters
        escaped_title = re.escape(movie_title)
        idx = movies[movies['title'].str.contains(escaped_title, case=False, na=False, regex=True)].index
        if idx.empty:
            print(f"No match found for movie: '{movie_title}'")
            return pd.DataFrame()
    movie_id = movies.iloc[idx[0]]['movieId']
    print(f"Found movie ID: {movie_id} for title: '{movies.iloc[idx[0]]['title']}'")
    
    # Load precomputed similar movies
    if str(movie_id) not in similarities:
        print(f"No precomputed similarities found for movie ID: {movie_id}")
        return pd.DataFrame()
    
    similar_movies = similarities[str(movie_id)]['similar_movies']
    scores = similarities[str(movie_id)]['scores']
    
    # Create recommendations DataFrame
    recommendations = movies[movies['movieId'].isin(similar_movies)].copy()
    recommendations['sim_score'] = [scores[similar_movies.index(mid)] for mid in recommendations['movieId']]
    
    # Apply filters
    recommendations = recommendations[
        (recommendations['avg_rating'] >= min_rating) &
        (recommendations['year'].between(year_range[0], year_range[1]))
    ]
    
    # Sort and return top_n
    recommendations = recommendations.sort_values(['sim_score', 'avg_rating'], ascending=False).head(top_n)
    print(f"Returning {len(recommendations)} recommendations for '{movie_title}'")
    return recommendations[['title', 'genres', 'avg_rating', 'year']]