#%%
from decorators import CallCountDecorator
from copy import deepcopy
import numpy as np 
import pandas as pd 
import pickle as pkl 
import matplotlib.pyplot as plt 
import os 
import sys 
from maher_initialisation import get_maher_estimate
#from biv_poiss import link_function
import biv_poiss
#import maher_initialisation
from scipy import optimize
import math


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

#get sets of teams in first year and teams that don't play in first year: 
starting_teams = participants_per_season.get(year_one)
non_first_partic = set()
for season in other_seasons:
    non_first_partic = non_first_partic.union(participants_per_season.get(season))
non_first_partic = non_first_partic.difference(starting_teams)


def get_f1(data_name,variable_names):
    name = data_name
    f1 = None
    if logbook.get(name):
        print(f"Found previously Maher-initialised team-strengths for {data_name} for variable: {variable_names}")
        if 'maher_'+variable_names in logbook[data_name].keys():
            f1 = logbook[name].get('maher_'+variable_names)
            print('Loaded f1-vector from saved Maher initalisation in logbook')
            return f1
        else: 
            #logbook[name]['maher'] = False
            print('Maher initialised f1 not found for this dataset.')
            print('Running Maher_initialisation on ', name)
            #first_partici = participants_per_season.get(year_one)
            df = load_data(data_name)
            f1 = get_maher_estimate(df, variable_names=variable_names)
            logbook[name].update({'maher_'+variable_names:f1})
            
    else:
        print(
            f"No entries found in log for {data_name} for {variable_names}. \n running Maher initialisation on {data_name}")
        df = load_data(data_name)
        f1 = get_maher_estimate(df, variable_names)
        logbook.update({data_name:{'maher_'+variable_names:f1 }})
    save_logbook(logbook)

    return f1


f1_goals = get_f1(data_name, 'goals')
f1_weighted_attempts = get_f1(data_name, 'weighted_attempts_discrete')
#initialize the using Maher  <DONE>

#%%

#----------------------All data is now loaded--------------------------------------------------#






#f1 = maher initialised strength vector
#latter_seasons = dataFrame of all other seasons
#participants_per_season= dict    season : participants_in_season 

        #teams first_year and teams not in first year:
#starting_teams = participants_per_season.get(year_one)
#non_first_partic

all_teams = deepcopy(set(df.ht) )
all_seasons = deepcopy(df)

print("RETRIEVED ALL DATA. RUNNING MODEL:")

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





def get_strengths_dictionary(f1, variable_names = 'goals'):
    # if variable_names == 'goals':
    #     a1, a2,b1,b2,delta,l3 = params
    # elif variable_names == 'goals weighted_attempts':
    #     a1,a2,b1,b2,c1,c2,d1,c2,delta,l3 = params

    if variable_names == 'goals':    
        ft = {team: [f1_goals.get(team)] for team in f1_goals.keys()}    
        for team in non_first_partic:
            ft.update({team: [(0, 0)]})
    if variable_names == 'weighted_attempts_discrete':
        ft = {team: [f1_weighted_attempts.get(team)] for team in f1_weighted_attempts.keys()}
        for team in non_first_partic:
            ft.update({team: [(0, 0)]})

    return deepcopy(ft)

def get_w (f1):
    
    f1_values = get_strengths_dictionary(f1, variable_names = 'goals')
    ones_vec = np.ones((team_amount*2,1))
    diagonal_of_B = np.array(([b1]*team_amount,[b2]*team_amount)).reshape((team_amount*2,1))

    rhs = ones_vec - diagonal_of_B 
    result = np.multiply(f1_values, rhs)
    return result.reshape(team_amount*2,) #used to be (66,)
    return 0




#@CallCountDecorator
def run_model(params, *args):
    #init teams that aren't playing in first year to be 0 
    variable_names, return_strengths = args
    if variable_names == 'goals':
        a1, a2, b1, b2, delta, l3 = params
        ft_local = deepcopy(ft_goals)
    if variable_names == 'weighted_attempts_discrete':
        a1, a2, b1, b2, delta, l3 = params
        ft_local = deepcopy(ft_attempts)


    total_likelihood = 0 

     

    for round in set(latter_seasons.round_labels):
        round_matches = latter_seasons[latter_seasons.round_labels == round]       
        
        round_participants = set(round_matches.iloc[0].participants)

        for index, match in round_matches.iterrows():
            h_col_name, a_col_name = 'home_'+variable_names, 'away_'+variable_names
            home_team = match['ht']
            away_team = match['at']
            home_result = int(match[h_col_name])
            away_result = int(match[a_col_name])

            home_strengths = alpha_i, beta_i = ft_local[home_team][-1]
            away_strengths = alpha_j, beta_j = ft_local[away_team][-1]

            #delta = delta #+ get_exogenous(match)

            l1,l2 = biv_poiss.link_function(alpha_i, alpha_j, beta_i, beta_j, delta)

            # l1 = math.exp(home_adv + home_strengths[0] - away_strengths[1])
            # l2 = math.exp(away_strengths[0] - home_strengths[1])

            score = biv_poiss.score(home_result, away_result, l1, l2, l3)

            alpha_i_next = 0+ b1*home_strengths[0] + a1*score[0]
            alpha_j_next = 0+ b1*away_strengths[0] + a1*score[1]
            beta_i_next = 0+ b2* home_strengths[1] + a2*score[2]
            beta_j_next = 0+ b2*away_strengths[1] + a2*score[3]

            ft_local[home_team].append((alpha_i_next, beta_i_next))
            ft_local[away_team].append((alpha_j_next, beta_j_next))

            p = biv_poiss.prob_mass_func(home_result, away_result, l1, l2, l3)

            if p > 0:
                total_likelihood = total_likelihood + math.log(p)        

        for team in all_teams.difference(round_participants):
            alpha_i_next = b1*ft_local[team][-1][0]
            beta_i_next = b2*ft_local[team][-1][1]
            ft_local[team].append((alpha_i_next, beta_i_next))
            #ft_local[team].append(ft_local[team][-1])


                 
    if return_strengths:
        return ft_local, -total_likelihood
    return -total_likelihood



x0 = [0.1, 0.1,   0.95, 0.95, 0.9, 0.9, 0.9, 0.9]
x0_b = [0.2, 0.2, 0.09, 0.9, 0.2, 0.2]  # a1, a2, l3 
# options={'maxfev':100}


#variable_names = 'goals'
#return_strengths = False

print('running score-driven model on goals-data')
ft_goals = get_strengths_dictionary(f1_goals, variable_names='goals')
args = ('goals', False)
opt = optimize.minimize(run_model, x0_b,args = args, method='BFGS')
x_opt = opt.x


print('running score-driven model on weighted_attempts-data')
ft_attempts = get_strengths_dictionary(f1_weighted_attempts, variable_names='weighted_attempts_discrete')
args = ('weighted_attempts_discrete', False)
opt_attempts = optimize.minimize(run_model, x0_b, args = args)
x_opt_attempts = opt_attempts.x
#args_opt = (variable_names, True)


strengths_opt, LL = run_model(x_opt, 'goals', True)
attempt_strengths_opt, LL2 = run_model(x_opt_attempts, 'weighted_attempts_discrete', True)

def plot_dict_entry(ft,team, variable_names = 'goals'):
    if variable_names == 'goals':
        attack = [i[0] for i in ft[team]]
        defense = [i[1] for i in ft[team]]
        x = pd.DataFrame([attack, defense]).T
        x.columns = ['goal_attack','goal_defense']
        x.plot()

    if variable_names == 'weighted_attempts_discrete':
        attack = [i[0] for i in ft[team]]
        defense = [i[1] for i in ft[team]]
        x = pd.DataFrame([attack, defense]).T
        x.columns = ['home_attempt_intensity', 'home_defense_intensity']
        x.plot()
        
plot_dict_entry(strengths_opt,'Lyon')
plot_dict_entry(attempt_strengths_opt,'Lyon', 'weighted_attempts_discrete')

# def game_likelihood():
#     return 0

# def total_likelihood():
#     for index, game in df.iteritems():
#         get_lambdas()
#         game_likelihood()
#         break

#     return 0


#run score driven updates on all but first year (which is used for Maher)

#calculate likelihood of the score-driven paths given a,b, delta, l3

#run optimizer on above 

#plot and compare to Koopman/Lit


# %%
