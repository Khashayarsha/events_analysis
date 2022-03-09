#%%

import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
import os
import sys


#get the latest matches data

cur_dir = os.getcwd()


def load_data(data_name):
    df = pd.read_csv(data_name)
    print(f"loaded {data_name} succesfully")
    return df


data_path = '../../data/raw/papardello/'

matches = load_data('matches_fr_labeled360554.csv')
teams =  pd.read_json(data_path+'teams.json')
players = pd.read_json(data_path+'players.json')
print('all dataframes loaded succesfully ')

teams_in_matches =  set(matches['ht at'.split()].values.flatten())

