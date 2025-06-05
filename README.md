# Movie Recommendation System
A modular movie recommendation system built with Python, Streamlit, and Hugging Face embeddings. It uses the MovieLens dataset to recommend movies based on tag similarity and user ratings, with a user-friendly Streamlit frontend.

## Features
- **Content-Based Recommendations**: Uses sentence-transformers/all-MiniLM-L6-v2 to generate tag embeddings for movie similarity.
- **Modular Design**: Separate modules for data preprocessing, embeddings, recommendation, and UI.
- **Precomputed Pipeline**: Generates embeddings and similarities offline for a lightweight Streamlit app.
- **Filtering**: Supports filters for minimum rating and release year range.

## Setup Instructions

### 1. Clone the Repository:
  ```bash
	git clone https://github.com/your-username/movie-recommender.git
	cd movie-recommendation-system
	```
### 2. Download MovieLens Dataset:
- Download the MovieLens 10M dataset from https://grouplens.org/datasets/movielens/10m/.
- Place movies.dat, ratings.dat, and tags.dat in the data/ directory.

### 3. Install Dependencies:
  ```bash
	pip install -r requirements.txt
	```

### 4. Run Precomputation:
	```bash
	python src\precompute.py
	```
	
### 5. Run the Streamlit App:
	```bash
	streamlit run src\app.py
	```

