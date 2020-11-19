import pickle
import numpy as np
from sklearn.decomposition import NMF
from fuzzywuzzy import process
import pandas as pd
import random
from dictionaries import R

# load dictionaries 

with open('./EDA/data/pickle_dictionaries.p', "rb") as file1:
        dictionaries = pickle.load(file1)

reference_movies =dictionaries[0]
dict_id_location = dictionaries[1]
dict_location_id = dictionaries[2]

def nmf_model(n, df):
    """ NMF model for movies recommender"""
    m = NMF(n_components=n, init='random')
    model = m.fit(df)
    Q = m.components_  # mvoies matrix
    P = m.transform(df)  # user matrix

    return P, Q, model

P, Q, trained_model=nmf_model(20,R)

def nmf_recommender(movies):   
    """NMF recommender:
    Function that provides 3 movie recommendations 
    based on the user input of 3 favorite movies
    Parameters: user input-list"""

    #assigning random ratings from a new user
    new_ratings=[]
    for i in range(1,len(R.columns)+1):
        new=round(random.uniform(0, 5), 1)
        new_ratings.append(new)
    new_ratings=np.array(new_ratings).reshape(1,9719)
     
    #finding movies and their indexes in dictionaries    
    index_location = []
    input_data = []
    for m in movies:
        fuzzy = process.extractOne(m, reference_movies.keys()) #matches the input data with the movies dictionary     
        input_data.append(fuzzy[0])                              #collects input data                         
        loc = dict_id_location[fuzzy[0]]             
        new_ratings[0, loc] = 5     #assigns rating 5 to the input data 
        index_location.append(loc)  #collects data location in table R

    #transforming by NMF model and reconstructing R for new user
    user = trained_model.transform(new_ratings)
    user_R = np.dot(user, Q)  

    # filter out already seen movies
    for i in index_location:
        user_R[0, i] = 0 # assigns 0 rating to the watched movies, 
                         # so when sorted they are not in the recommendation list

    user_R_ind = list(zip(user_R[0], range(len(user_R[0]))))  #creating a list with indexes from user_R
    user_R_ind.sort(key=lambda x: x[0], reverse = True)       #sorting in ascending order  

    recommendations = []
    for j in user_R_ind[0:10]: 
        movie = dict_location_id[j[1]]
        recommendations.append(movie)

    return recommendations, input_data