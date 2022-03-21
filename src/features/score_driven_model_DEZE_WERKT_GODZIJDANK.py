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


def save_logbook(dic, name = 'logbook.pkl'):
    with open(name, 'wb') as file:
        pkl.dump(dic, file )


def load_logbook(name='logbook.pkl'):
    with open(name, 'rb') as file:
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


# previous: 'matches_fr_labeled89294.pkl'
data_name = 'matches_fr_labeled_latest3.pkl'
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
        print(f"Found previous entries for {data_name}: {logbook[name].keys()}")
        if 'maher_'+variable_names in logbook[data_name].keys():
            f1 = logbook[name].get('maher_'+variable_names)
            print(f"Loaded f1-vector '  {'maher_'+variable_names}  ' from logbook")
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
f1_attempts = get_f1(data_name, 'attempts')
f1_weighted_attempts = get_f1(data_name, 'weighted_attempts_discrete')

#%%

f1_attempt_count_013 = get_f1(data_name, 'counted_attempts_0.13')
f1_attempt_count_0219 = get_f1(data_name, 'counted_attempts_0.219')


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
    if variable_names == 'attempts':
        ft = {team: [f1_goals.get(team)] for team in f1_goals.keys()}
        for team in non_first_partic:
            ft.update({team: [(0, 0)]})
    if variable_names == 'weighted_attempts_discrete':
        ft = {team: [f1_weighted_attempts.get(team)] for team in f1_weighted_attempts.keys()}
        for team in non_first_partic:
            ft.update({team: [(0, 0)]})

    if variable_names =='goals_and_wa':
        #estimating goal-intensities using the weighted-attempts strength as exogenous variables
        ft = {team: [f1_goals.get(team)] for team in f1_goals.keys()}
        for team in non_first_partic:
            ft.update({team: [(0, 0)]})

    if variable_names == 'counted_attempts_0.219':
        ft = {team: [f1_attempt_count_0219.get(
            team)] for team in f1_attempt_count_0219.keys()}
        for team in non_first_partic:
            ft.update({team: [(0, 0)]})
    #if variable_names not in ['goals', 'attempts', 'weighted_attempts_discrete']:
     #   print(f"SDM get_strengths_dict says: Variable name {variable_names} not recognized.")

    return deepcopy(ft)

def get_w (f1):
    
    f1_values = get_strengths_dictionary(f1, variable_names = 'goals')
    ones_vec = np.ones((team_amount*2,1))
    diagonal_of_B = np.array(([b1]*team_amount,[b2]*team_amount)).reshape((team_amount*2,1))

    rhs = ones_vec - diagonal_of_B 
    result = np.multiply(f1_values, rhs)
    return result.reshape(team_amount*2,) #used to be (66,)
    return 0

# results_dict = {}
# for match_id in df.id_odsp:
# results_dict = {match_id: {'h_est_'+variable_names: [],
#                            'a_est_'+variable_names: []} for match_id in df.id_odsp}

#@CallCountDecorator


def run_model_extended3(params, *args):

    # ('goals weighted_attempts', False, ft_goals, ft_weighted_attempts, 1) = args
    variable_names, return_strengths, ft_local_goals, ft_local_other, extension_type = args
    var_name1, var_name2 = variable_names.split()    #'goals', 'weighted_attempts'
    if extension_type == 1:
        # a1, a2, b1, b2, delta, l3, eta = params
        a1,  a2, a3, a4,  delta1, delta2, l3, theta3, eta1 = params
    if extension_type == 2:
        #a1, a2, b1, b2, delta,l3, eta1, eta2  = params
        a1,  a2, a3, a4,  delta1, delta2, l3, theta3, eta1, eta2 = params
    # if return_strengths:
    #     match_id_dict = {match_id: {'home': [], 'away': []}
    #                      for match_id in df.id_odsp}
    b1 = 1
    b2 = 1  # set to 1
    b3 = 1
    b4 = 1

    ft_local_goals = deepcopy(ft_local_goals)
    ft_local_other = deepcopy(ft_local_other)


    total_likelihood = 0

    for round in set(latter_seasons.round_labels):
        round_matches = latter_seasons[latter_seasons.round_labels == round]

        round_participants = set(round_matches.iloc[0].participants)

        for index, match in round_matches.iterrows():
            home_team = match['ht']
            away_team = match['at']

            h_col_name, a_col_name = 'home_'+var_name1, 'away_'+var_name1            
            home_result = int(match[h_col_name])
            away_result = int(match[a_col_name])
            #home_weighted_attempts = match_id_dict[match.id_odsp]['home']
            #away_weighted_attempts = match_id_dict[match.id_odsp]['away']
            home_strengths = alpha_i, beta_i = ft_local_goals[home_team][-1][0:2]
            away_strengths = alpha_j, beta_j = ft_local_goals[away_team][-1][0:2]


            h_col_name_other, a_col_name_other = 'home_'+var_name2, 'away_'+var_name2            
            home_result_other = int(match[h_col_name_other])
            away_result_other = int(match[a_col_name_other])
            home_strengths_other = gamma_i, nu_i = ft_local_other[home_team][-1][0:2]
            away_strengths_other = gamma_j, nu_j = ft_local_other[away_team][-1][0:2]

            # if return_strengths:           This is used in the other run_model to construct the id-dict
            #     match_id_dict[match.id_odsp]['home'].append(
            #         ft_local[home_team][-1][0:2])
            #     match_id_dict[match.id_odsp]['away'].append(
            #         ft_local[away_team][-1][0:2])

            #delta = delta #+ get_exogenous(match)
            if extension_type == 1:
                l1, l2 = biv_poiss.link_function_ext1(
                    alpha_i, alpha_j, beta_i, beta_j, delta1, eta1, gamma_i, nu_j)
            if extension_type == 2:
                l1, l2 = biv_poiss.link_function_ext2(
                    alpha_i, alpha_j, beta_i, beta_j, delta1, eta1, eta2, gamma_i, gamma_j, nu_i, nu_j)

            if extension_type == 1 or extension_type == 2:
                theta1, theta2 = biv_poiss.link_function(gamma_i, gamma_j, nu_i, nu_j, delta2)
            else:
                print(f"extension type {extension_type} not recognized")


            # l1 = math.exp(home_adv + home_strengths[0] - away_strengths[1])
            # l2 = math.exp(away_strengths[0] - home_strengths[1])

            goals_score = biv_poiss.score(home_result, away_result, l1, l2, l3)

            alpha_i_next = 0 + b1*home_strengths[0] + a1*goals_score[0]
            alpha_j_next = 0 + b1*away_strengths[0] + a1*goals_score[1]
            beta_i_next = 0 + b2 * home_strengths[1] + a2*goals_score[2]
            beta_j_next = 0 + b2*away_strengths[1] + a2*goals_score[3]

            other_score = biv_poiss.score(home_result_other, away_result_other, theta1, theta2, theta3)

            gamma_i_next = 0+ b3*home_strengths_other[0] + a3* other_score[0]
            gamma_j_next = 0+ b3*home_strengths_other[0] + a3* other_score[1]
            nu_i_next =  0+ b4* home_strengths_other[1] + a4 *other_score[2]
            nu_j_next = 0 + b4*home_strengths_other[1] + a4 * other_score[3]


            ft_local_goals[home_team].append(
                (alpha_i_next, beta_i_next, round, match.id_odsp, 'home'))
            ft_local_goals[away_team].append(
                (alpha_j_next, beta_j_next, round, match.id_odsp, 'away'))

            ft_local_other[home_team].append(
                (gamma_i_next, nu_i_next, round, match.id_odsp, 'home'))
            ft_local_other[away_team].append(
                (gamma_j_next, nu_j_next, round, match.id_odsp, 'away'))


            #results_dict[match.id_odsp]['h_est_'+variable_names].append()

            p = biv_poiss.prob_mass_func(home_result, away_result, l1, l2, l3)

            if p > 0:
                total_likelihood = total_likelihood + math.log(p)

        for team in all_teams.difference(round_participants):
            alpha_i_next = b1*ft_local_goals[team][-1][0]
            beta_i_next = b2*ft_local_goals[team][-1][1]

            gamma_i_next = b3* ft_local_other[team][-1][0]
            nu_i_next = b4 * ft_local_other[team][-1][1]


            ft_local_goals[team].append((alpha_i_next, beta_i_next, round, 'absent'))
            ft_local_other[team].append( (gamma_i_next, nu_i_next, round, 'absent'))
            #ft_local[team].append(ft_local[team][-1])

    if return_strengths:
        return ft_local_goals, ft_local_other,  -total_likelihood
    return -total_likelihood



def run_model_extended(params, *args):

    variable_names, return_strengths, ft_local,  match_id_dict, extension_type = args # df, latter_seasons, all_teams, match_id_dict = args
    if extension_type == 1:
        # a1, a2, b1, b2, delta, l3, eta = params
         a1, a2, delta, l3, eta = params
    if extension_type == 2:
        #a1, a2, b1, b2, delta,l3, eta1, eta2  = params
        a1, a2, delta, l3, eta1, eta2 = params
    # if return_strengths:
    #     match_id_dict = {match_id: {'home': [], 'away': []}
    #                      for match_id in df.id_odsp}
    b1 = 1 
    b2 = 1  #set to 1 
    ft_local = deepcopy(ft_local)
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
            #home_weighted_attempts = match_id_dict[match.id_odsp]['home']
            #away_weighted_attempts = match_id_dict[match.id_odsp]['away']

            home_strengths = alpha_i, beta_i = ft_local[home_team][-1][0:2]
            away_strengths = alpha_j, beta_j = ft_local[away_team][-1][0:2]
            home_wa_strengths = gamma_i, nu_i = match_id_dict[match.id_odsp]['home'][0]
            away_wa_strengths = gamma_j, nu_j = match_id_dict[match.id_odsp]['away'][0]

            # if return_strengths:           This is used in the other run_model to construct the id-dict
            #     match_id_dict[match.id_odsp]['home'].append(
            #         ft_local[home_team][-1][0:2])
            #     match_id_dict[match.id_odsp]['away'].append(
            #         ft_local[away_team][-1][0:2])

            #delta = delta #+ get_exogenous(match)
            if extension_type == 1:
                l1, l2 = biv_poiss.link_function_ext1(
                    alpha_i, alpha_j, beta_i, beta_j, delta, eta, gamma_i, nu_j)
            if extension_type == 2:
                l1, l2 = biv_poiss.link_function_ext2(alpha_i, alpha_j, beta_i, beta_j, delta, eta1, eta2, gamma_i, gamma_j, nu_i, nu_j)


            # l1 = math.exp(home_adv + home_strengths[0] - away_strengths[1])
            # l2 = math.exp(away_strengths[0] - home_strengths[1])

            score = biv_poiss.score(home_result, away_result, l1, l2, l3)

            alpha_i_next = 0 + b1*home_strengths[0] + a1*score[0]
            alpha_j_next = 0 + b1*away_strengths[0] + a1*score[1]
            beta_i_next = 0 + b2 * home_strengths[1] + a2*score[2]
            beta_j_next = 0 + b2*away_strengths[1] + a2*score[3]

            ft_local[home_team].append(
                (alpha_i_next, beta_i_next, round, match.id_odsp, 'home'))
            ft_local[away_team].append(
                (alpha_j_next, beta_j_next, round, match.id_odsp, 'away'))

            #results_dict[match.id_odsp]['h_est_'+variable_names].append()

            p = biv_poiss.prob_mass_func(home_result, away_result, l1, l2, l3)

            if p > 0:
                total_likelihood = total_likelihood + math.log(p)

        for team in all_teams.difference(round_participants):
            alpha_i_next = b1*ft_local[team][-1][0]
            beta_i_next = b2*ft_local[team][-1][1]
            ft_local[team].append((alpha_i_next, beta_i_next, round, 'absent'))
            #ft_local[team].append(ft_local[team][-1])

    if return_strengths:
        return ft_local, match_id_dict, -total_likelihood
    return -total_likelihood


def run_model(params, *args):
    #init teams that aren't playing in first year to be 0 
    variable_names, return_strengths = args
    if variable_names == 'goals':
        a1, a2, delta, l3 = params #a1, a2, b1, b2, delta, l3 = params
        ft_local = deepcopy(ft_goals)
    if variable_names == 'attempts':
        a1, a2, delta, l3 = params      #a1, a2, b1, b2, delta, l3 = params
        ft_local = deepcopy(ft_attempts)
    if variable_names == 'weighted_attempts_discrete':
        a1, a2, b1, b2, delta, l3 = params
        ft_local = deepcopy(ft_weighted_attempts)
    if variable_names == 'goals_and_wa': 
        a1,a2,a3,a4,b1,b1,b3,b4,delta1,delta2,l3,theta3 = params
        ft_local = deepcopy(ft_goals)

    ft_local.update({'match_id':{}})
    if return_strengths: 
        match_id_dict = {match_id:{'home':[] , 'away': [] } for match_id in df.id_odsp}

    #if variable_names == 'goals':        
    b1 = 1
    b2 = 1

    total_likelihood = 0 

     

    for round in set(latter_seasons.round_labels):
        round_matches = latter_seasons[latter_seasons.round_labels == round]       
        
        round_participants = set(round_matches.iloc[0].participants)

        for index, match in round_matches.iterrows():
            #ft_local['match_id'].update({match.id_odsp: })
            h_col_name, a_col_name = 'home_'+variable_names, 'away_'+variable_names
            home_team = match['ht']
            away_team = match['at']
            home_result = int(match[h_col_name])
            away_result = int(match[a_col_name])

            home_strengths = alpha_i, beta_i = ft_local[home_team][-1][0:2]
            away_strengths = alpha_j, beta_j = ft_local[away_team][-1][0:2]

            if return_strengths:
                match_id_dict[match.id_odsp]['home'].append(ft_local[home_team][-1][0:2])
                match_id_dict[match.id_odsp]['away'].append(ft_local[away_team][-1][0:2])

            #delta = delta #+ get_exogenous(match)

            l1,l2 = biv_poiss.link_function(alpha_i, alpha_j, beta_i, beta_j, delta)

            # l1 = math.exp(home_adv + home_strengths[0] - away_strengths[1])
            # l2 = math.exp(away_strengths[0] - home_strengths[1])

            score = biv_poiss.score(home_result, away_result, l1, l2, l3)

            alpha_i_next = 0+ b1*home_strengths[0] + a1*score[0]
            alpha_j_next = 0+ b1*away_strengths[0] + a1*score[1]
            beta_i_next = 0+ b2* home_strengths[1] + a2*score[2]
            beta_j_next = 0+ b2*away_strengths[1] + a2*score[3]

            ft_local[home_team].append((alpha_i_next, beta_i_next, round, match.id_odsp,'home'))
            ft_local[away_team].append((alpha_j_next, beta_j_next, round, match.id_odsp,'away'))

            #results_dict[match.id_odsp]['h_est_'+variable_names].append()

            p = biv_poiss.prob_mass_func(home_result, away_result, l1, l2, l3)

            if p > 0:
                total_likelihood = total_likelihood + math.log(p)        

        for team in all_teams.difference(round_participants):
            alpha_i_next = b1*ft_local[team][-1][0]
            beta_i_next = b2*ft_local[team][-1][1]
            ft_local[team].append((alpha_i_next, beta_i_next, round, 'absent'))
            #ft_local[team].append(ft_local[team][-1])


                 
    if return_strengths:
        return ft_local, match_id_dict, -total_likelihood
    return -total_likelihood



x0 = [0.1, 0.1,   0.95, 0.95, 0.9, 0.9, 0.9, 0.9]
x0_b = [0.2, 0.2, 0.2, 0.2]  # a1, a2, delta, l3
# options={'maxfev':100}


#variable_names = 'goals'
#return_strengths = False

print(f"Estimating model for GOALS")
ft_goals = get_strengths_dictionary(f1_goals, variable_names='goals')
#args = ('goals', False)
#opt = optimize.minimize(run_model, x0_b,args = args, method='BFGS')
g_optimum_no_b = [0.02030879, 0.01928687, 0.32837571, 0.05405947]
strengths_opt, match_dict_goals, LL = run_model(g_optimum_no_b, 'goals', True)

#g_optimum = [0.02056461, 0.01779263, 0.99449382, 0.99792438, 0.34076973, 0.02790289]
#g_optimum = [0.02056461, 0.01779263, 0.34076973, 0.02790289]  # [0.02056461, 0.01779263, 0.99449382,             0.99792438, 0.34076973, 0.02790289]
#strengths_opt, match_dict_goals, LL = run_model(g_optimum, 'goals', True)



# print(f"Estimating model for ATTEMPTS-----------------------------------------------------------")
# ft_attempts = get_strengths_dictionary(f1_attempts,variable_names = 'attempts')
# args = ('attempts', False)
# opt_attempts = optimize.minimize(run_model, x0_b, args=args)
# attempt_strengths_opt, match_dict_attempts, LL2 = run_model(opt_attempts.x, 'attempts', True)

 
print(f"Estimating constrained model for ATTEMPTS-----------------------------------------------")
ft_attempts = get_strengths_dictionary(f1_attempts, variable_names='attempts')
args = ('attempts', False)


bnds = [(-1, 1), (-1, 1), (-6, 6),
        (0, 0.99) ]
opt_attempts = optimize.minimize(run_model, x0_b, args=args, method='SLSQP', bounds=bnds, tol=None, callback=None, options={
                                       'disp': None, 'maxcor': 10, 'ftol': 2.220446049250313e-09, 'gtol': 1e-05, 'eps': 1e-08, 'maxfun': 15000, 'maxiter': 15000, 'iprint': - 1, 'maxls': 20, 'finite_diff_rel_step': None})
attempt_strengths_opt, match_dict_attempts, LL2 = run_model(opt_attempts.x, 'attempts', True)



# print(f"Estimating model for WEIGHTED ATTEMPTS-----------------------------------------------------------")
ft_weighted_attempts = get_strengths_dictionary(f1_weighted_attempts, variable_names='weighted_attempts_discrete')
# wa_optimum = [0.05960302, 0.09505286, 0.98744643,
#               0.99708252, 0.66970482,  0.63086085]
# #print(f"Estimating model for WEIGHTED ATTEMPTS")
# weighted_attempt_strengths_opt, match_dict_weighted_attempts,  LL3 = run_model(
#     wa_optimum, 'weighted_attempts_discrete', True)



#args = ('weighted_attempts_discrete', False)
#opt_weighted_attempts = optimize.minimize(run_model, x0_b, args = args)
#weighted_attempt_strengths_opt, match_dict_weighted_attempts,  LL3 = run_model(opt_weighted_attempts.x, 'weighted_attempts_discrete', True)
#args_opt = (variable_names, True)

# variable_names, return_strengths, ft_local,  match_id_dict, extension_type

# print('estimating model extension 1 (goals, weighted_attempts eta1) -----------------------------------------------------------------------')
# args = ('goals', False, ft_goals,  match_dict_weighted_attempts, 1 ) #(var_names, return_strengths, ft_local,  match_id_dict )
# x0_extended = [0.0205, 0.0178, 0.3407, 0.02790,0.4]  # a1, a2, b1, b2, delta, l3, eta = params

# bnds = [(-1, 1), (-1, 1), (0, 5),    #a1 a2 delta 
#         (0, 5), (0, 5)]      # l3 eta
# extended_strengths_opt = optimize.minimize(run_model_extended, x0_extended, args=args, method='L-BFGS-B', bounds=bnds, tol=None, callback=None, options={
#     'disp': None, 'maxcor': 10, 'ftol': 2.220446049250313e-09, 'gtol': 1e-05, 'eps': 1e-08, 'maxfun': 15000, 'maxiter': 15000, 'iprint': - 1, 'maxls': 20, 'finite_diff_rel_step': None})
# #extended_strengths_opt = optimize.minimize(run_model_extended, x0_extended, args=args, method='BFGS')
# extended_strengths_opt_dict, match_dict_extended, LL4 = run_model_extended(
#     extended_strengths_opt.x, 'goals', True, ft_goals,  match_dict_weighted_attempts, 1)
# # extended_strengths_opt.x = [ 0.020,  0.0177, 0.9946, 0.9979,  0.3257, 0.01853, -0.01364  ]
#                             #   a1       a2      b1       b2     delta     l3       eta


# print('estimating model extension 2 (goals, weighted_attemts, eta1, eta2------------------------')
# args2 = ('goals', False, ft_goals,  match_dict_weighted_attempts, 2)  # (var_names, return_strengths, ft_local,  match_id_dict )
# x0_extended2 = [0.0205, 0.0178,  0.3407, 0.02790, 0.4,0.4]  # a1, a2, b1, b2, delta, l3, eta1,eta2 = params

# extended_strengths_opt2 = optimize.minimize(
#     run_model_extended, x0_extended2, args=args2, method='BFGS')
# extended_strengths_opt2_dict, match_dict_extended, LL5 = run_model_extended(
#     extended_strengths_opt2.x, 'goals', True, ft_goals,  match_dict_weighted_attempts ,2)


# print('estimating model extension 3  (simulataneous estimation) ------------------------')
# # (var_names, return_strengths, ft_local,  extension_type )
# args3 = ('goals weighted_attempts', False, ft_goals, ft_weighted_attempts, 1)
# # a1, a2, b1, b2, delta, l3, eta1,eta2 = params

# x0_extended3 = [0.02, 0.02, 0.02, 0.02,  0.3, 0.3,  0.05, 0.05, 0.05] #, 0.5]  # 9 params. 10 with eta2
#                 #a1,  a2 , a3, a4 ,delta1,delta2,l3,     theta3,eta1 (eta2)
# bnds = [(-1, 1), (-1, 1), (-1, 1), (-1, 1), (-10, 10), (-10, 10), (0, 1), (0,1), (-10,10)]
# import scipy
# #extended_strengths_opt3 =scipy.optimize.differential_evolution(run_model_extended3, bounds =bnds, args=args3, strategy='best1bin', maxiter=50, popsize=15, tol=0.01, mutation=(0.5, 1), recombination=0.7, seed=None, callback=None, disp=False, polish=True, init='latinhypercube', atol=0, updating='immediate', workers=1, constraints=())

# extended_strengths_opt3 = optimize.minimize(run_model_extended3, x0_extended3, args=args3, method='BFGS')


# goal_strengths_simultaan_dict, weighted_strengths_simultaan_dict, LL6 = run_model_extended3(
#     extended_strengths_opt3.x, 'goals weighted_attempts', True, ft_goals,  ft_weighted_attempts, args3[-1])


# # optimum (quasi) found for extended model 3 (simultaneous) with extension_nr = 2.
# #[ 0.02489183,  0.02296556, -0.01361026,  0.27460132,  0.34403237, 0.23702291,  0.05186831, -0.02307723,  0.02760914,  0.02124028

# names = 'goals weighted_attempts_discrete'.split()
# exo_dict = {name: {team_name:{} for team_name in all_teams} for name in names }


print('estimating model extension 4  (simulataneous estimation, using attempt_counts_0219) ------------------------')
ft_counted0219 = get_strengths_dictionary(
    f1_attempt_count_0219, variable_names='counted_attempts_0.219')
args4 = ('goals counted_attempts_0.219', False, ft_goals, ft_counted0219, 1)

x0_extended4 = [0.02, 0.02, 0.02, 0.02,  0.3, 0.3,  0.05, 0.05, 0.05]
extended_strengths_opt4 = optimize.minimize(
    run_model_extended3, x0_extended4, args=args4, method='BFGS')
#extended_strengths_opt4.x = [0.02035264,  0.01928753,  0.02955609,  0.0073123,  0.33332439,
                            # 1.23161871,  0.053819, -0.00170566,  0.00524042]
goal_strengths_simultaan_dict2, counted_strengths_simultaan_dict, LL7 = run_model_extended3(
    extended_strengths_opt4.x, 'goals counted_attempts_0.219', True, ft_goals,  ft_counted0219, args4[-1])


print(f'SUCCESFULLY (?) FINISHED ESTIMATING EXTENSION 4')


print('estimating model extension 5  (simulataneous estimation, using attempt_counts_0219) ------------------------')
ft_counted0219 = get_strengths_dictionary(
    f1_attempt_count_0219, variable_names='counted_attempts_0.219')
args5 = ('goals counted_attempts_0.219', False, ft_goals, ft_counted0219, 2)

x0_extended5 = [0.02, 0.02, 0.02, 0.02,  0.3, 0.3,  0.05, 0.05, 0.05,0.05]
extended_strengths_opt5 = optimize.minimize(
    run_model_extended3, x0_extended5, args=args5, method='BFGS')
#extended_strengths_opt4.x = [0.02035264,  0.01928753,  0.02955609,  0.0073123,  0.33332439,
                            # 1.23161871,  0.053819, -0.00170566,  0.00524042]
goal_strengths_simultaan_dict3, counted_strengths_simultaan_dict2, LL8 = run_model_extended3(
    extended_strengths_opt5.x, 'goals counted_attempts_0.219', True, ft_goals,  ft_counted0219, args5[-1])


print(f'SUCCESFULLY (?) FINISHED ESTIMATING EXTENSION 4')



#attempt_strengths_opt, LL2 = run_model(x_opt_attempts, 'weighted_attempts_discrete', True)

def plot_dict_entry(ft, team, variable_names='goals', save = False):
    # if variable_names == 'goals':
    #     attack = [i[0] for i in ft[team]]
    #     defense = [i[1] for i in ft[team]]
    #     x = pd.DataFrame([attack, defense]).T
    #     x.columns = ['attack_goal','defense_goal']
    #     x.plot(title = team+'__'+variable_names)

    # if variable_names == 'weighted_attempts_discrete':
    #     attack = [i[0] for i in ft[team]]
    #     defense = [i[1] for i in ft[team]]
    #     x = pd.DataFrame([attack, defense]).T
    #     x.columns = ['attack_weighted_attempt', 'defense_weighted_attempt']
    #     x.plot(title=team+'__'+variable_names)
    # else:
    attack = [i[0] for i in ft[team]]
    defense = [i[1] for i in ft[team]]
    x = pd.DataFrame([attack, defense]).T
    x.columns = ['attack_'+variable_names, 'defense_'+variable_names]
    x.plot(title=team+'__'+variable_names)
    fig = x.plot(title=team+'__'+variable_names).get_figure()
    if save != False:
        fig.savefig(team+'__'+variable_names+'.png', dpi=400)


plot_dict_entry(strengths_opt,'Lyon')
#plot_dict_entry(attempt_strengths_opt, 'Lyon', 'attempts')
# plot_dict_entry(weighted_attempt_strengths_opt, 'Lyon', 'weighted_attempts_discrete')
# plot_dict_entry(extended_strengths_opt_dict,
#                 'Lyon', 'extended_model_goals')
# plot_dict_entry(extended_strengths_opt2_dict, 'Lyon', 'extended_model_2_goals')

plot_dict_entry(goal_strengths_simultaan_dict, 'Lyon', 'goals_simultaneous extended')
plot_dict_entry(weighted_strengths_simultaan_dict, 'Lyon', 'weighted_attempts simultaneous extended')

plot_dict_entry(goal_strengths_simultaan_dict2, 'Lyon', 'goals_simultaneous_counts0219')

plot_dict_entry( counted_strengths_simultaan_dict, 'Lyon', 'counted_attempts_strengths' )


plot_dict_entry(goal_strengths_simultaan_dict3, 'Lyon',
                'goals_simultaneous_counts0219_ext2')

plot_dict_entry(counted_strengths_simultaan_dict2,
                'Lyon', 'counted_attempts_strengths_ext2')

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

save_dict2 = {'x0': x0_extended3, 'optimizer_results': extended_strengths_opt3,
              'goal_strengths': goal_strengths_simultaan_dict, 'wa_strengths': weighted_strengths_simultaan_dict, 'likelihood': LL6}

from scipy import stats
from scipy.stats.distributions import chi2
LL1 = -LL
LL2 = -LL6


def likelihood_ratio(llmin, llmax):

    return 2*(llmax-llmin)


LR = likelihood_ratio(LL1, LL2)
dof = 6  # Degrees of freedom
p = chi2.sf(LR, dof)

print('LR: ', LR, 'p: ', p)



# %%
