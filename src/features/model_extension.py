import biv_poiss
import numpy as np 
import pandas as pd 
import math 



def run_model(params, *args):
    variable_names, return_strengths, ft_local,df, latter_seasons,all_teams, match_id_dict = args
    a1, a2, b1, b2, delta, l3, eta = params
    # if return_strengths:
    #     match_id_dict = {match_id: {'home': [], 'away': []}
    #                      for match_id in df.id_odsp}

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
            home_wa_strengths = gamma_i, nu_i = match_id_dict[match.id_odsp]['home']
            away_wa_strengths = gamma_j, nu_j = match_id_dict[match.id_odsp]['away']

            if return_strengths:
                match_id_dict[match.id_odsp]['home'].append(ft_local[home_team][-1][0:2])
                match_id_dict[match.id_odsp]['away'].append(ft_local[away_team][-1][0:2])

            #delta = delta #+ get_exogenous(match)

            l1,l2 = biv_poiss.link_function_ext1(alpha_i, alpha_j, beta_i, beta_j, delta, eta, gamma_i, nu_j)

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
