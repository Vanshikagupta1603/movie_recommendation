import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_data
def load_movies():
    movies_dict = pickle.load(open('movie_dict.pkl','rb'))
    return pd.DataFrame(movies_dict)

@st.cache_data
def build_similarity(df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['tags'].astype('str'))
    return cosine_similarity(tfidf_matrix)

movies = load_movies()
similarity = build_similarity(movies)

st.title('Movie Recommender System')

selected_movie_name = st.selectbox(
    'Choose a movie',
    movies['title'].values)

if st.button('Recommend'):
    movie_index = movies[movies['title'] == selected_movie_name].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended_movies = [movies.iloc[i[0]].title for i in movies_list]
    for recommended_movie in recommended_movies:
        st.write(recommended_movie)

