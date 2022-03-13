#%%

import numpy as np 
import pandas as pd 
import pickle as pkl 
import matplotlib.pyplot as plt 
import os 
import sys 
from maher_initialisation import get_maher_estimate
#import maher_initialisation




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
    name = data_name
    if logbook[name].get('maher'):
        f1 = logbook[name].get('maher')
        return f1
    else: 
        logbook[name]['maher'] = False
        print('Maher initialised f1 not found for this dataset.')
        print('Running Maher_initialisation on ', name)
        #first_partici = participants_per_season.get(year_one)
    df = load_data(data_name)
    f1 = get_maher_estimate(df, variable_names='goals')

    return f1


f1 = get_f1(data_name)
#initialize the using Maher  <DONE>
#----------------------All data is now loaded--------------------------------------------------#
#f1 = maher initialised strength vector
#latter_seasons = dataFrame of all other seasons
#participants_per_season= dict    season : participants_in_season 



def initialise_f():
    #put all Maher values from f1 in the dict that tracks all teams
    initial_strengths = {}
    return initial_strengths



def update_all(params):
    #run score updating on all teams, to get their time-varying strength-paths 
    #for all strengths concerned
    return 0 

def get_lambdas(variable_names):
    #construct the intensities l1, l2 and home-advantage delta 
    #based on which variables are used in the model 
    #if only 'goals' as variable: go for 

    return 0







def game_likelihood():
    return 0

def total_likelihood():
    for index, game in df.iteritems():
        get_lambdas()
        game_likelihood()
        break

    return 0


#run score driven updates on all but first year (which is used for Maher)

#calculate likelihood of the score-driven paths given a,b, delta, l3

#run optimizer on above 

#plot and compare to Koopman/Lit

