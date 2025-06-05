import pandas as pd
import re

def load_and_preprocess_data(movies_path='data/movies.dat', ratings_path='data/ratings.dat', tags_path='data/tags.dat'):
    print("Starting data loading and preprocessing...")
    
    # Load data
    print("Loading movies data...")
    movies = pd.read_csv(movies_path, sep="::", engine="python", header=None, 
                         names=["movieId", "title", "genres"])
    print(f"Loaded {len(movies)} movies.")

    print("Loading ratings data...")
    ratings = pd.read_csv(ratings_path, sep="::", engine='python', header=None,
                          names=["userId", "movieId", "rating", "timestamp"])
    print(f"Loaded {len(ratings)} ratings.")

    print("Loading tags data...")
    tags = pd.read_csv(tags_path, sep="::", engine="python", header=None, 
                       names=["userId", "movieId", "tag", "timestamp"])
    print(f"Loaded {len(tags)} tags.")

    # Extract year from title
    print("Extracting release years from movie titles...")
    movies['year'] = movies['title'].str.extract(r'\((\d{4})\)')
    movies['year'] = pd.to_numeric(movies['year'], errors='coerce').fillna(0).astype(int)
    print(f"Extracted years for {len(movies)} movies.")

    # Compute average ratings
    print("Computing average ratings per movie...")
    avg_ratings = ratings.groupby('movieId')['rating'].mean().reset_index(name='avg_rating')
    movies = movies.merge(avg_ratings, on='movieId', how='left').fillna(3.0)
    print(f"Computed average ratings for {len(movies)} movies.")

    # Clean and aggregate tags
    print("Cleaning and aggregating tags...")
    # Convert tags to string and handle NaN
    tags['tag'] = tags['tag'].astype(str).str.lower().str.strip().replace('nan', '')
    tags = tags[tags['tag'] != ''].groupby('movieId')['tag'].apply(lambda x: ' '.join(set(x))).reset_index()
    movies = movies.merge(tags, on='movieId', how='left').fillna({'tag': ''})
    print(f"Aggregated tags for {len(movies)} movies.")

    print("Data preprocessing complete.")
    return movies