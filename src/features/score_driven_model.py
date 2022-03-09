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
    df =     pd.read_csv(data_name)
    print(f"loaded {data_name} succesfully")
    return df 


df = load_data('matches_fr_labeled360554.csv')


#add rounds labels <DONE>


#add "first time seen" labels 


#initialize the using Maher 

#run score driven updates on all but first year (which is used for Maher)

#calculate likelihood of the score-driven paths given a,b, delta, l3

#run optimizer on above 

#plot and compare to Koopman/Lit

