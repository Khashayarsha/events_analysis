#%%
from functools import reduce
import pandas as pd 
import numpy as np 
import math
#import score_driven_model
import biv_poiss
import scipy
from scipy import optimize
import time

#from score_driven_model import get_playing_team

tic = time.time()

# def link_function(parameters, variables=['goals']):

#     if variables == 'goals':
#         alpha_i, alpha_j, beta_i, beta_j, delta = parameters 

#         try:
#             l1 = min(10, math.exp( delta + alpha_i - beta_j))
#             l1 = max(0.1, l1)
#         except OverflowError:
#             l1 = 10
                
#         try:
#             l2 = min(10, math.exp(alpha_j - beta_i))
#             l2 = max(0.1, l2)
#         except OverflowError:
#             l2 = 10
#         return l1,l2

#     elif variables == 'goals weighted_attempts_discrete':
#         alpha_i, alpha_j, beta_i, beta_j, gam_i, gam_j, nu_i, nu_j,delta,eta = parameters
#         try:
#             l1 = min(10, math.exp(delta + alpha_i - beta_j + eta*(gam_i - nu_j)))
#             l1 = max(0.1, l1)
#         except OverflowError:
#             l1 = 10

#         try:
#             l2 = min(10, math.exp(alpha_j - beta_i + eta*(gam_j - nu_i)))
#             l2 = max(0.1, l2)
#         except OverflowError:
#             l2 = 10

#         return l1, l2
#     else: 
#         print('unexpected variable type used')
#         return False

eps = 1e-14
thresh = 30


def link_function(parameters, variables=['goals']):  #
    #exponent1= np.clip(delta + ai - bj, a_min =-5, a_max = 20 )  # bestond eerst niet
    #exponent2 = np.clip(aj - bi, a_min= -5, a_max = 20)
    if variables == 'goals':
        ai, aj, bi, bj, delta = parameters
        exponent1 = delta + ai - bj
        try:
            l1 = min(thresh, math.exp(exponent1))  # delta + ai - bj))
            l1 = max(eps, l1)
        except OverflowError:
            print('OVERFLOW ERROR IN MAHER INITIALISATION')
            l1 = thresh  # was eerst 50, daarvoor 10

        try:
            l2 = min(thresh, math.exp(aj - bi))
            l2 = max(eps, l2)
        except OverflowError:
            print('OVERFLOW ERROR IN MAHER INITIALISATION')
            l2 = thresh

        return l1, l2



def maher_estimation(x, *args):
    #x = strengths vector
    first_year_data, teams, variable_names = args

    df = first_year_data
    total_log_likelihood = 0
    
    #if variable_names == 'attempts':
    for index, match in df.iterrows():
        h_col_name, a_col_name = 'home_'+variable_names, 'away_'+variable_names

        home = match['ht']
        away = match['at']
        home_result = match[h_col_name]
        away_result = match[a_col_name]

        alpha_i, alpha_j = x[teams[home][0]], x[teams[away][0]]
        beta_i, beta_j = x[teams[home][1]], x[teams[away][1]]
        delta, l3 = x[teams['delta'][0]], x[teams['l3'][0]]
        parameters = (alpha_i, alpha_j, beta_i, beta_j, delta)
        # we actually estimate it the same way as the normal goals-based analysis, at first.
        l1, l2 = link_function(parameters, 'goals')

        game_log_likelihood = math.log(
            biv_poiss.prob_mass_func(home_result, away_result, l1, l2, l3))
        total_log_likelihood += game_log_likelihood

#print('done calculating total likelihood once')
    return -1*total_log_likelihood

def get_playing_team(df, season):
    temp = df[df['season'] == season].apply(
        lambda x: set(x.participants), axis='columns')
    participants_in_season = reduce(lambda x, y: x.union(y), temp.values)
    return participants_in_season

def constraint_alphas(x):
    # hardcoded for convenience.
    n = len(x)-2
    alpha_indices = [i for i in range(n) if i%2 ==0]
    alphas = [x[i] for i in alpha_indices]
    """ Sum of alphas (attack strengths) should be 0 for identification purposes. """
    
    return -1*sum(alphas)


def constraint_delta(x):
    delta = x[-2]

    return delta


def constraint_gamma(x):
    gamma = x[-1]
    return gamma








    # if variable_names == 'goals weighted_attempts':
    #     alpha_i, alpha_j, beta_i, beta_j, gam_i, gam_j, nu_i, nu_j,delta,eta = parameters
    #     print('weighted_attempts variables initialisation not yet implemented')

def start(first_year_data, first_year_participants, variable_names = 'goals'):
    #initialises the dictionary that will contain the strengths per team:
    n = len(first_year_participants)
    participants = sorted(list(first_year_participants))
    
    teams = {team:(2*index, (2*index)+1) for index,team in enumerate(participants)}
    teams.update({'delta':[2*n]})
    teams.update({'l3': [2*n+1]})

    
     
    # if variable_names == 'goals':
    print(f"starting Maher-optimization for {variable_names} ")
    x_ini = tuple([0]*(2*n+2))

    con1 = {'type': 'eq', 'fun': constraint_alphas}
    #con2 = {'type': 'ineq', 'fun': constraint_delta}
    #con3 = {'type': 'ineq', 'fun': constraint_gamma}
    cons = [con1]#, con2, con3]


    boundaries = [(-np.inf, np.inf) if i < (2*n) else (0, 1) for i in range(2*n+2)]
    options = {'eps': 1e-09,  # was 1e-09
            'disp': True,
            'maxiter': 500}

    results = scipy.optimize.minimize(maher_estimation, x_ini,options=options, args=(first_year_data, teams, variable_names),
                                    method='SLSQP', 
                                    constraints=cons,
                                    bounds=boundaries)

    # if variable_names == 'weighted_attempts_discretized': 
    #     print(f"starting Maher-optimization for {variable_names} ")

        
    return results, teams

#first_year, participants = score_driven_model.get_first_year()
# results, teams = start(first_year, participants)

# x = results.x #strengths-vector
# strengths = {team: (x[v[0]], x[v[1]]) #strengths dict   w/ mapping  team_name : alpha_i, beta_i
#  for team, v in teams.items() if team not in ['delta', 'l3']}
# strengths.update({'delta':x[teams['delta']]})
# strengths.update({'l3': x[teams['l3']]})


#zorg dat de dict telkens naar zelfde soort datatype wijst (tuple)
#fix de get_maher_estimate functie zodat het maher runt, en de year1 strengths dict returnt. 

def load_data(data_name):
    df = pd.read_pickle(data_name)
    print(f"loaded {data_name} succesfully")
    return df

def get_maher_estimate(df, variable_names = "goals"):
    print(f"MAHER: RUNNING MAHER ESTIMATION")
      

    year_one = min(df['season'])
    print(f"MAHER: first year detected as {year_one}, using variables {variable_names}")
    first_year_data = df[df['season'] == year_one]

    first_participants = get_playing_team(first_year_data, year_one)
    results, teams = start(first_year_data, first_participants, variable_names )

    x = results.x  # strengths-vector


    strengths_dict = {team: (x[v[0]], x[v[1]])  # strengths dict   w/ mapping  team_name : alpha_i, beta_i
             for team, v in teams.items() if team not in ['delta', 'l3']}
    strengths_dict.update({'delta': x[teams['delta']]})
    strengths_dict.update({'l3': x[teams['l3']]})



    return strengths_dict


# df = load_data('matches_fr_labeled89294.pkl')
# strengths_debug = get_maher_estimate(df, 'weighted_attempts_discrete')

toc = time.time()

print(f"elapsed time = {toc-tic} seconds.. for Maher initialisation")




#if variable_names == 'goals':
    #     for index, match in df.iterrows():
    #         #print(index)
    #         home = match['ht']
    #         away = match['at']
    #         home_goals = match.home_goals
    #         away_goals = match.away_goals

    #         alpha_i, alpha_j = x[teams[home][0]], x[teams[away][0]]
    #         beta_i, beta_j = x[teams[home][1]], x[teams[away][1]]
    #         delta, l3 = x[teams['delta'][0]], x[teams['l3'][0]]
    #         parameters = (alpha_i, alpha_j, beta_i, beta_j, delta)
    #         l1, l2 = link_function(parameters, 'goals')

    #         game_log_likelihood = math.log(
    #             biv_poiss.prob_mass_func(home_goals, away_goals, l1, l2, l3))
    #         total_log_likelihood +=game_log_likelihood

    #     #print('done calculating total likelihood once')
    #     return -1*total_log_likelihood

    # if variable_names == 'weighted_attempts_discrete':
    #     for index, match in df.iterrows():
    #         'home_weighted_attempts_discrete'
    #         'away_weighted_attempts_discrete'

    #         home = match['ht']
    #         away = match['at']
    #         home_result = match['home_weighted_attempts_discrete']
    #         away_result = match['away_weighted_attempts_discrete']

    #         alpha_i, alpha_j = x[teams[home][0]], x[teams[away][0]]
    #         beta_i, beta_j = x[teams[home][1]], x[teams[away][1]]
    #         delta, l3 = x[teams['delta'][0]], x[teams['l3'][0]]
    #         parameters = (alpha_i, alpha_j, beta_i, beta_j, delta)
    #         l1, l2 = link_function(parameters, 'goals') #we actually estimate it the same way as the normal goals-based analysis, at first.

    #         game_log_likelihood = math.log(
    #             biv_poiss.prob_mass_func(home_result, away_result, l1, l2, l3))
    #         total_log_likelihood += game_log_likelihood

    #     #print('done calculating total likelihood once')
    #     return -1*total_log_likelihood
