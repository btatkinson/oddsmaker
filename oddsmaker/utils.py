
import gc
import os
import faker
import random

import numpy as np
import pandas as pd

from copy import copy

def make_symmetrical(df, protag_id='protag_id', antag_id='antag_id'):
    
    antag_df = df.copy()
    antag_df['result'] = 1-antag_df['result'].copy()
    antag_df = antag_df.rename(columns={protag_id:antag_id, antag_id:protag_id})
    
    df = pd.concat([df, antag_df], axis=0).reset_index(drop=True)
    
    return df

def create_fake_Elo_ratings(num_players, mean=1500, std_dev=100):

    """Create fake Elo ratings for a given number of players."""

    fake = faker.Faker()
    rtg_dict = {}
    ratings = np.random.normal(mean, std_dev, num_players)
    for i in range(num_players):
        rtg_dict[fake.name()] = ratings[i]

    return rtg_dict


def create_Elo_random_walk(player_rating_dict, num_rounds=250, step_size=5, stat='chess'):

    """Create a random walk of Elo ratings, given some ratings."""

    assert(num_rounds>=1)
    names = list(player_rating_dict.keys())

    data_random_walk = []
    for rnd in range(1, num_rounds+1):
        rnd_set = set(copy(names))
        while len(rnd_set)>=2:
            player_a = random.choice(list(rnd_set))
            rnd_set.remove(player_a)
            player_b = random.choice(list(rnd_set))
            rnd_set.remove(player_b)
            data_random_walk.append([player_a, player_b, rnd, player_rating_dict[player_a], player_rating_dict[player_b]])
        
        for player in names:
            nudge = step_size*2*(np.random.random()-0.5)
            rtg  = player_rating_dict[player]
            rtg = np.max([1, rtg+nudge])
            player_rating_dict[player] = rtg
            
    data_random_walk = pd.DataFrame(data_random_walk, columns=['player_a','player_b','rating_period', 'player_a_true','player_b_true'])
    data_random_walk = data_random_walk.sort_values(by=['rating_period','player_a']).reset_index(drop=True)
    data_random_walk['true_prob'] = data_random_walk.apply(lambda x: 1 / (1 + 10 ** ((x.player_b_true - x.player_a_true) / 400)), axis=1)
    data_random_walk['result'] = data_random_walk['true_prob'].apply(lambda x: 1 if x > np.random.random() else 0).astype(int)
    data_random_walk['stat'] = stat

    ## need reverse
    reverse = data_random_walk.copy()

    reverse = reverse.rename(columns={
        'player_a':'player_b',
        'player_b':'player_a',
        'player_a_true':'player_b_true',
        'player_b_true':'player_a_true'
    })

    reverse['true_prob'] = 1-reverse['true_prob'].copy()
    reverse['result'] = 1-reverse['result'].copy().astype(int)

    data_random_walk = pd.concat([data_random_walk, reverse], axis=0).sort_values(by=['rating_period','player_a']).reset_index(drop=True)
    data_random_walk['rating_period'] = data_random_walk['rating_period'].astype(int)

    return data_random_walk





