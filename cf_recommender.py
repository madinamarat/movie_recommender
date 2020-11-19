import pandas as pd
import numpy as np
import pickle
from scipy.spatial import distance
from dictionaries import R
from fuzzywuzzy import process

# load dictionaries 

with open('./EDA/data/pickle_dictionaries.p', "rb") as file1:
        dictionaries = pickle.load(file1)

reference_movies =dictionaries[0]
dict_id_location = dictionaries[1]
dict_location_id = dictionaries[2]

#prepare collaborative filtering matrix

def cf_model(df):
    """Collaborative Filtering Model"""
    df=df.T
    # cf_array = distance.squareform(distance.pdist(df, metric='cosine'))
    # cf_matrix=pd.DataFrame(cf_array)  
    cf_array = distance.squareform(distance.pdist(df, metric='cosine'))
    smaller_array = cf_array.astype('float32')   #smaller float
    cf_matrix=pd.DataFrame(np.tril(smaller_array))    #np.tril makes a triangle from rect., to reduce memory 
    
    return cf_matrix

cf_matrix = cf_model(R)

def cf_recommender(movies):
    """Collaborative filtering recommender
    Parameter: user input - list of 3 movies"""

    for m in movies:
        fuzzy = process.extractOne(m, reference_movies.keys()) #matches the input data with the movies dictionary                 
        loc_sim = dict_id_location[fuzzy[0]]  

        # get  neighbour-movies:
        nb = cf_matrix.loc[loc_sim].sort_values(ascending=False)  #takes interceptions of the favorite movies with all the movies in cf matrix

        top3 = nb.sort_values(ascending=False)[:3] # returns 3 best recommendations
        top3 = top3.to_frame()
    
    recommendations = []
    for i in range(0,3):
        movie_title = dict_location_id[int(top3.index[i])]
        recommendations.append(movie_title)

    return recommendations