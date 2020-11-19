
import pandas as pd

ratings=pd.read_csv('./EDA/data/ratings.csv',sep=',')
movies=pd.read_csv('./EDA/data/movies.csv',sep=',')

#list of duplicated movie titles
dup_movies=movies[movies.duplicated(subset=['title'])]['title'].tolist()
print(dup_movies)

#list of duplicated movies sorted by title
dup_sorted = movies[movies['title'].isin(dup_movies)].sort_values(by='title', ascending=True)
dup_sorted

# create a list of duplicated movieIds:
dup_ids = dup_sorted['movieId'].to_list()
print(dup_ids)

# create a dictionary with movieIds for the films:
movie_id_dict = dict(zip(dup_ids[::2],dup_ids[1::2]))
movie_id_dict

# create a unique_id column 
ratings['unique_id'] = ratings['movieId'] 
movies['unique_id'] = movies['movieId']

# replace the values of the movieId from the dictionary
ratings = ratings.replace({"unique_id": movie_id_dict}) 
movies = movies.replace({"unique_id": movie_id_dict}) 
print('duplicated ids removed')

# merge movies and ratings dataframes
mov_rat=ratings.merge(movies, on=['unique_id'])
print('movies and ratings dataframes merged')

# export cleaned up combined dateframe
mov_rat.to_csv('./EDA/data/cleaned_df.csv')
print('cleaned df saved')