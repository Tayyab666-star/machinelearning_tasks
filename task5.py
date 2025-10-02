import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the dataset (Replace with the correct file path)
df = pd.read_csv('tmdb_5000_movies.csv')  # Replace with your file path

# Preprocessing: Fill missing descriptions with an empty string (if any)
df['overview'] = df['overview'].fillna('')

# Use TF-IDF Vectorizer on the 'overview' column
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['overview'])

# Check the shape of the tf-idf matrix (it will be num of movies x num of features)
print(tfidf_matrix.shape)

from sklearn.metrics.pairwise import cosine_similarity

# Calculate cosine similarity between all movies
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Check the similarity for the first movie (index 0)
print(cosine_sim[0])

def recommend_movie(movie_id, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the movie_id
    idx = df.index[df['movieId'] == movie_id].tolist()[0]
    
    # Get the similarity scores for the selected movie
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort the movies by similarity score (highest first)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get the top 5 most similar movies (excluding the movie itself)
    top_movies = sim_scores[1:6]
    
    # Get the indices of the top 5 movies
    movie_indices = [i[0] for i in top_movies]
    
    return df['title'].iloc[movie_indices]

# Example usage
recommended_movies = recommend_movie(1)  # Replace '1' with the movieId you want recommendations for
print(recommended_movies)


