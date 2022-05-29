# import the important libraries

import streamlit as st
import pickle
import pandas as pd
import requests

# adding important files
movies = pickle.load(open('movies.pkl', 'rb'))
movie_list = movies['title'].values
cosine_sim2 = pickle.load(open('cosine_sim2.pkl', 'rb'))

st.header('Movie Recommendation System')
selected_movie = st.selectbox("choose the movie you have already watched : ", movie_list)

with st.sidebar:
    st.subheader('About:')
    st.text('It is a Content Based Movie ')
    st.text('recommendation system.')

    st.text('Choose the movie you have already')
    st.text('watched and it will show ')
    st.text('recommendation based on cast,')
    st.text('keywords and overview.')

# defining function for recommendation and fetching posters
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()


# fetching poster through api of tmdb site
def fetch_poster(i):
    data = (requests.get(f"https://api.themoviedb.org/3/movie/{i}?api_key=0874a959adc0544d6ae5cccf55ef5951&language=en-US".format(id))).json()

    print(data)
    # print(data )   # The poster path is taken from the generated Json File, and then image is given as the output
    return "https://image.tmdb.org/t/p/w500/" + data['poster_path']


def recommend(movie, cosine_sim_2=cosine_sim2):
    # Get the index of the movie that matches the title
    idx = indices[movie]

    # Get the pairwsie similarity scores of all movies with that movie

    sim_scores = list(enumerate(cosine_sim2[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    recommended_movie_names = []
    recommended_movie_posters = []

    # Get the movie indices
    for i in sim_scores:
        recommended_movie_posters.append(fetch_poster(movies.iloc[i[0]].id))
        recommended_movie_names.append(movies.iloc[i[0]].title)

    # Return the top 10 most similar movies
    return recommended_movie_names, recommended_movie_posters


# coloumn for displaying poster and recommendation
if st.button('Show Recommendation'):
    recommended_movie_names, recommended_movie_posters = recommend(selected_movie)
    print(recommended_movie_posters[0])
    # Each row is set with 3 columns, first poster then the text is given as the output
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(recommended_movie_posters[0])
        st.text(recommended_movie_names[0])
    with col2:
        st.image(recommended_movie_posters[1])
        st.text(recommended_movie_names[1])
    with col3:
        st.image(recommended_movie_posters[2])
        st.text(recommended_movie_names[2])

    col4, col5, col6 = st.columns(3)
    with col4:
        st.image(recommended_movie_posters[3])
        st.text(recommended_movie_names[3])
    with col5:
        st.image(recommended_movie_posters[4])
        st.text(recommended_movie_names[4])
    with col6:
        st.image(recommended_movie_posters[5])
        st.text(recommended_movie_names[5])

    col7, col8, col9 = st.columns(3)
    with col7:
        st.image(recommended_movie_posters[6])
        st.text(recommended_movie_names[6])
    with col8:
        st.image(recommended_movie_posters[7])
        st.text(recommended_movie_names[7])
    with col9:
        st.image(recommended_movie_posters[8])
        st.text(recommended_movie_names[8])

st.title('Project Details')

with st.sidebar:
    st.header('Project Details :')

    st.subheader('What is a Recommendation system?')
    st.subheader('What is Cosine Similarity and TF-IDF Matrix?')
    st.subheader('Why I am using Cosine Similarity?')
    st.subheader('Why I am using the TF-IDF Matrix?')
    st.subheader('What Sorting Algorithm I am using and why?')
    st.text('Graph Comparison:')
    st.text('Time Complexity')
    st.text('Space Complexity')

st.header('What is a Recommendation system?')
st.text('Recommender systems are a type of machine learning algorithm that provides consumer')
st.text('with "relevant" recommendations.')

st.header('What is Cosine Similarity and TF-IDF Matrix?')
st.text('Cosine Similarity : The metric cosine similarity is used to determine how similar ')
st.text('two objects are. It calculates the cosine of the angle formed by two vectors')
st.text('projected in three dimensions.')

st.text('TF-IDF : TF-IDF stands for Term Frequency Inverse Document Frequency of records. It')
st.text('be defined as the calculation of how relevant a word in a series is to a text.')

st.header('Why I am using Cosine Similarity?')
st.text('As we can use it to define text similarity between any two documents or paragraphs ')
st.text('by representing the word in vector form.')

st.header('Why I am using the TF-IDF Matrix?')
st.text('So I can create vectors for the cosine similarity.')

st.header('What Sorting Algorithm I am using and why?')
st.text('I am using sorted() function of python which uses TIM SORT algorithm.')
st.text('Following are the reason for choosing TIM SORT:')

from PIL import Image

image = Image.open('sorttc.jpg')
st.image(image, caption='Comparison by Time Complexity ')

image = Image.open('sortsc.jpg')
st.image(image, caption='Comparison by Space Complexity ')

st.text('Since tim sort is a faster and stable version of merge sort and perform well with')
st.text('ordered data, I choose to use it.')

with st.sidebar:
    st.header('Improvements :')

    st.subheader('Sorting Algorithm')
    st.subheader('Why I did not use QUAD SORT?')
    st.subheader('Increase Efficiency by Caching?')
    st.subheader('Use of Knn Algorithm')

st.title("Improvements")
st.header('Sorting Algorithm:')
st.text('I have used sorted() function of python which uses TIM SORT but if you want to ')
st.text('increases the performance slightly then you can use QUAD SORT.')

st.subheader('Why I did not use QUAD SORT?')
st.text('As speed of both the algorithms are similar and QUAD SORT performs slightly better')
st.text('than TIM SORT, you can use TIM SORT can be used directly by the function sorted()')
st.text('but you have to write a decent code for QUAD SORT. If you want that slightly high ')
st.text('performance, you can go for QUAD SORT.')

st.header('Increase Efficiency by Caching :')
st.text('If you have the knowledge and experience with caching you can use various caching ')
st.text('to increase the efficiency. Few examples are Using network cache, HTTP caching,')
st.text('Reverse proxy server caching, Using database cache. And you can also improve ')
st.text('your hardware for better results. ')

st.header('Use of Knn Algorithm:')
st.text('If you have the user rating dataset that correspond to your movie dataset then you ')
st.text('can use Knn algorithm to show results based on user liking.  ')

