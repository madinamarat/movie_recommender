import pandas as pd
import pickle

#load cleaned dataframe 
df = pd.read_csv('./EDA/data/cleaned_df.csv', index_col=0)

#create R matrix
R =df.pivot_table(index='userId', columns='title', values='rating')
R.fillna(R.median().median(),inplace=True)
 
#create dictionaries
reference_movies = pd.Series(df["unique_id"].values,index=df["title"]).to_dict() # {'Toy Story (1995)': 1..}

dict_id_location = {}    # location in the table R, {"'71 (2014)": 0...} 
for num, i in enumerate(R.columns):
    dict_id_location[i] = num
    
dict_location_id = dict((i,j) for j,i in dict_id_location.items())

#create top 50 list movies based on the sum of ratings
R.loc['rating_sum'] = R.sum(axis=0)
top50=R.sort_values(by='rating_sum',axis=1, ascending=False).iloc[-1:, 0:50].columns.to_list()

# save dictionaries
dictionaries = [reference_movies, dict_id_location, dict_location_id, top50]

with open('./EDA/data/pickle_dictionaries.p', 'wb') as f1:
    pickle.dump(dictionaries, f1)