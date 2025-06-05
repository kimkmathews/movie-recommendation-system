from data_preprocessing import load_and_preprocess_data
from embeddings import generate_and_save_embeddings
from recommender import precompute_similarities

def main():
    # Step 1: Load and preprocess data
    print("Starting data loading and preprocessing...")
    movies = load_and_preprocess_data()
    print(f"Data loading complete. Loaded {len(movies)} movies.")

    # Step 2: Generate and save embeddings
    print("Starting tag embeddings generation...")
    embeddings = generate_and_save_embeddings(movies)
    print(f"Embeddings generation complete for {len(embeddings)} movies. Saved to 'outputs/embeddings.npy'.")

    # Step 3: Precompute and save similarities
    print("Starting similarity computation...")
    similarities = precompute_similarities(movies, embeddings)
    print(f"Similarity computation complete for {len(similarities)} movies. Saved to 'outputs/similarities.json'.")

if __name__ == '__main__':
    main()