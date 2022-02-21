
import os 
import sys 
import pandas as pd 
import numpy as np 


#this code returns relevant dictionaries to process the Kaggle csv

code_path = os.getcwd()
data_path = "C:/Users/XHK/Desktop/thesis_code/events_analysis/data/raw/kaggle"

#events = pd.read_csv('/'.join(data_path,"events.csv"))
#ginf = pd.read_csv('/'.join(data_path, "ginf.csv"))

file_contents = 0
with open('/'.join([data_path, 'dictionary.txt'])) as file:
    file_contents = file.read()

file_contents = [i.split('\t') for i in file_contents.split('\n')]
file_contents = [i for i in file_contents if len(i[0])>0]

event_type1 = [['0', 'Announcement'],
['1', 'Attempt'],
['2', 'Corner'],
['3', 'Foul'],
['4', 'Yellow card'],
['5', 'Second yellow card'],
['6', 'Red card'],
['7', 'Substitution'],
['8', 'Free kick won'],
['9', 'Offside'],
['10', 'Hand ball'],
['11', 'Penalty conceded']]

event_type2 = [['12', 'Key Pass'],
               ['13', 'Failed through ball'],
               ['14', 'Sending off'],
               ['15', 'Own goal']]

side = [        ['1', 'Home'],
        ['2', 'Away'] ]

shot_place = [['1', 'Bit too high'],
              ['2', 'Blocked'],
              ['3', 'Bottom left corner'],
              ['4', 'Bottom right corner'],
              ['5', 'Centre of the goal'],
              ['6', 'High and wide'],
              ['7', 'Hits the bar'],
              ['8', 'Misses to the left'],
              ['9', 'Misses to the right'],
              ['10', 'Too high'],
              ['11', 'Top centre of the goal'],
              ['12', 'Top left corner'],
              ['13', 'Top right corner'] ]

shot_outcome = [['1', 'On target'],
['2', 'Off target'],
['3', 'Blocked'],
['4', 'Hit the bar']]


location = [['1', 'Attacking half'],
 ['2', 'Defensive half'],
 ['3', 'Centre of the box'],
 ['4', 'Left wing'],
 ['5', 'Right wing'],
 ['6', 'Difficult angle and long range'],
 ['7', 'Difficult angle on the left'],
 ['8', 'Difficult angle on the right'],
 ['9', 'Left side of the box'],
 ['10', 'Left side of the six yard box'],
 ['11', 'Right side of the box'],
 ['12', 'Right side of the six yard box'],
 ['13', 'Very close range'],
 ['14', 'Penalty spot'],
 ['15', 'Outside the box'],
 ['16', 'Long range'],
 ['17', 'More than 35 yards'],
 ['18', 'More than 40 yards'],
 ['19', 'Not recorded']]

bodypart = [['1', 'right foot'],
             ['2', 'left foot'],
             ['3', 'head']]

assist_method = [['0', 'None'],
                 ['1', 'Pass'],
                 ['2', 'Cross'],
                 ['3', 'Headed pass'],
                 ['4', 'Through ball']]

situation = [['1', 'Open play'],
 ['2', 'Set piece'],
 ['3', 'Corner'],
 ['4', 'Free kick']]


lists = [event_type1, event_type2, side, shot_place,
         shot_outcome, location, bodypart, assist_method, situation]

dicts = []
for ind,array in enumerate(lists): 
    lists[ind] = {int(i):j for [i,j] in array}

print(lists)

event_type1, event_type2, side, shot_place, \
shot_outcome, location, bodypart, assist_method, situation = lists #[event_type1, event_type2, side, shot_place,
                                                              #shot_outcome, location, bodypart, assist_method, situation]



def get_dictionaries():
    print('Getting dictionaries to analyse events.csv')
    return [event_type1, event_type2, side, shot_place,  shot_outcome, location, bodypart, assist_method, situation]
#%%
