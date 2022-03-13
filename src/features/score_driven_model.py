#%%

import numpy as np 
import pandas as pd 
import pickle as pkl 
import matplotlib.pyplot as plt 
import os 
import sys 

def save_logbook(dic):
    with open('logbook.pkl', 'wb') as file:
        pkl.dump(dic, file )

def load_logbook():
    with open('logbook.pkl', 'rb') as file:
        logbook = pkl.load(file)
        return logbook

def check_logbook(logbook, data_name):
    data_name = data_name.split('.')[0]
    print(logbook[data_name])

logbook = load_logbook()

#get the latest matches data 

cur_dir = os.getcwd()

def load_data(data_name):
    df =     pd.read_pickle(data_name)
    print(f"loaded {data_name} succesfully")
    return df 


data_name = 'matches_fr_labeled89294.pkl'
df = load_data(data_name)
year_one = min(df['season'])
first_year = df[df['season'] == year_one]
latter_seasons = df[df['season']!=year_one]

from functools import reduce 
def get_playing_team(df, season):
    temp = df[df['season']==season].apply(lambda x: set(x.participants), axis = 'columns')
    participants_in_season = reduce(lambda x, y: x.union(y), temp.values)
    return participants_in_season

other_seasons = set(latter_seasons.season)
participants_per_season = {season:get_playing_team(df,season) for season in set(df.season) }

non_first_partic = set()
for season in other_seasons:
    non_first_partic.union(participants_per_season.get(season))

def get_f1(data_name):
    name = data_name.split('.')[0]
    if logbook[name].get('maher'):
        f1 = logbook[name].get('maher')
        return f1
    else: 
        logbook[name]['maher'] = False
        print('Maher initialised f1 not found for this dataset.')
        print('Running Maher_initialisation on ', name)
        first_partici = participants_per_season.get(year_one)
#        f1 = maher_initialisation.start(first_year, first_partici,
                # variables = ['goals','weighted_attempts_discretized'])



def get_first_year():
    return first_year, participants_per_season.get(year_one)






#initialize the using Maher 

#run score driven updates on all but first year (which is used for Maher)

#calculate likelihood of the score-driven paths given a,b, delta, l3

#run optimizer on above 

#plot and compare to Koopman/Lit

