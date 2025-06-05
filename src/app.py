import streamlit as st
from data_preprocessing import load_and_preprocess_data
from recommender import load_recommendations
from utils import load_movies, load_similarities

def main():
    st.title('Movie Recommendation System')
    
    # Load data and precomputed similarities
    @st.cache_data
    def load_cached_data():
        movies = load_and_preprocess_data()
        similarities = load_similarities()
        return movies, similarities
    
    movies, similarities = load_cached_data()
    
    # Streamlit interface
    movie_title = st.selectbox('Select a movie:', load_movies())
    min_rating = st.slider('Minimum average rating:', 0.0, 5.0, 3.0, step=0.1)
    year_min = st.slider('Minimum year:', 1900, 2025, 1990)
    year_max = st.slider('Maximum year:', 1900, 2025, 2025)
    print(movie_title)
    if st.button('Recommend'):
        recommendations = load_recommendations(movie_title, movies, similarities, min_rating=min_rating, year_range=(year_min, year_max))
        if not recommendations.empty:
            st.write('### Recommended Movies')
            st.dataframe(recommendations)
        else:
            st.write('No recommendations found. Try adjusting filters.')

if __name__ == '__main__':
    main()