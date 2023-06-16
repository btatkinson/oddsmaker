import gc
import os

import numpy as np
import pandas as pd

from copy import copy
from tqdm import tqdm

from sklearn.linear_model import Ridge as RidgeRegression
from bayes_opt import BayesianOptimization
from sklearn.preprocessing import OneHotEncoder
# from scipy.optimize import minimize, dual_annealing
from pandas.api.types import is_datetime64_any_dtype as is_datetime





class RidgeRating():

    """
    Implements ranking system that uses ridge regression to estimate mean player/team ratings.
    Can work for 1v1,1vMany,Manyv1,ManyvMany games.
    Can use either a date or rating period to decay ratings.
    Can use a custom decay function.
    Can use a custom prior rating for each player/team.
    Can use a custom home field advantage.

        Parameters
    ----------
    data : pandas dataframe
        Dataframe containing the following columns:
            - protag_id: id of protagonist (player or team)
            - antag_id: id of antagonist (player or team)
            - stat: name of stat
            - result: 0 for loss, 1 for win, 0.5 for tie
            - is_home (optional): 1 for home, -1 for away, 0 for neutral
            - date: date of game
            - rating_period (if no date): rating period of game 
    alpha : float
        Regularization parameter for ridge regression. The default is 1.0.
    decay_type : str
        Type of decay to use. Either 'date' or 'rating_period'. The default is 'date'.
    decay_function : function
        Function that takes a time delta and returns a decay factor. The default is lambda x: 1.0.
    protag_id : str, optional
        Name of protagonist id column in data. The default is 'team_name'.
    antag_id : str, optional
        Name of antagonist id column in data. The default is 'opp_name'.
    result_col : str, optional
        Name of result column in data. The default is 'result'.
    matchup_type : str, optional
        Type of matchup. Either '1v1', '1vMany', or 'ManyvMany'. The default is '1v1'.
    priors : dict, optional
        Dictionary of prior ratings to use for each stat. Keys are protag_ids, values are dicts of stat:rating pairs. The default is None.  
    
    """

    def __init__(
            self, 
            data, 
            alpha, 
            decay_type, 
            decay_function, 
            protag_id='player_a', 
            antag_id='player_b', 
            result_col='result', 
            matchup_type='1v1', 
            priors=None):

        data.columns=[x.lower().strip() for x in data.columns]

        self.data = data
        assert(isinstance(self.data, pd.DataFrame)), "data is not a pandas dataframe"
        self.alpha = alpha
        assert(isinstance(self.alpha, float)), "alpha is not a float"
        self.decay_type = decay_type
        assert(decay_type in ['date','rating_period']), "decay_type is not 'date' or 'rating_period'"
        if decay_type == 'date':
            assert('date' in self.data.columns), "data does not contain a date column"
            assert(is_datetime(self.data['date'])), "date column is not a datetime"
            self.period_col = 'date'
        elif decay_type == 'rating_period':
            assert('rating_period' in self.data.columns), "data does not contain a rating_period column"
            assert(self.data['rating_period'].dtype == int), "rating_period column is not an int"
            self.period_col = 'rating_period'
        self.decay_function = decay_function
        assert(callable(self.decay_function)), "decay_function is not a function"
        self.protag_id = protag_id
        assert(isinstance(self.protag_id, str)), "protag_id is not a string"
        self.antag_id = antag_id
        assert(isinstance(self.antag_id, str)), "antag_id is not a string"
        self.result_col = result_col
        assert(isinstance(self.result_col, str)), "result_col is not a string"
        self.matchup_type = matchup_type
        assert(matchup_type in ['1v1','1vMany','ManyvMany']), "matchup_type is not '1v1', '1vMany', or 'ManyvMany'"
        self.priors = priors
        assert(isinstance(self.priors, dict) or self.priors is None), "priors is not a dict or None"

        if (('is_home' not in self.data.columns) and ('hfa' not in self.data.columns)):
            print("Warning: no is_home or home field advantage ('hfa') column found in data. Assuming all games are neutral site.")
            self.data['is_home'] = 0

        assert('stat' in self.data.columns), "data does not contain a stat column"
        self.stats = list(self.data['stat'].unique())
        assert(len(self.stats)>0), "data does not contain any stats"
        assert([isinstance(x, str) for x in self.stats]), "stats are not strings"

        ### useful meta information
        self.num_stats = len(self.stats)
        self.protag_ids = list(self.data[self.protag_id].unique())
        self.num_protags = len(self.protag_ids)
        self.num_games = len(self.data)//2

        ### check if data is symmetrical, i.e., for every team a vs team b there is a team b vs team a
        protags = self.data.groupby(['rating_period','stat'])[self.protag_id].apply(set).reset_index().copy()
        antags =self.data.groupby(['rating_period','stat'])[self.antag_id].apply(set).reset_index().copy()
        sym_test = protags.merge(antags, how='left', on=['rating_period','stat'])
        sym_test['sym_diff'] = sym_test[[self.protag_id,self.antag_id]].apply(lambda x: len(x[self.protag_id].symmetric_difference(x[self.antag_id])), axis=1)
        sym_val = sym_test['sym_diff'].mean()
    
        if sym_val > 0.05:
            print("Warning: data is not symmetrical. There should be two rows per match for this class")
            raise ValueError(" At least {}% of games are missing.".format(round(100*sym_val,2)))
        elif sym_val > 0:
            print("Warning: a few games are likely missing their symmetrical partner.")


    def optimize(self):

        return
    

    
    
    def run_history(self):

        self.data = self.data.sort_values(by=[self.period_col,self.protag_id]).reset_index(drop=True)
        self.data['decay'] = self.data[self.period_col].apply(self.decay_function)

        null_decays = self.data.decay.isnull().sum()/len(self.data)
        if null_decays > 0:
            print("Warning: {}% of games have no decay factor.".format(round(100*null_decays,2)))
            print("         Setting decay factor to 0.")
        self.data['decay'] = self.data['decay'].fillna(0)

        unique_periods = self.data[self.period_col].unique()

        pregame_ratings = []
        for unique_period in tqdm(unique_periods[1:]):
            prev_history = self.data[self.data[self.period_col]<unique_period].copy()
            
            team_ohe = OneHotEncoder()
            opp_ohe = OneHotEncoder()
            team_ohe.fit(prev_history[self.protag_id].values.reshape(-1,1))
            opp_ohe.fit(prev_history[self.antag_id].values.reshape(-1,1))
            team_one_hot = team_ohe.transform(prev_history[self.protag_id].values.reshape(-1,1)).toarray()
            opp_one_hot = opp_ohe.transform(prev_history[self.antag_id].values.reshape(-1,1)).toarray()
            if 'hfa' not in prev_history.columns:
                is_home = prev_history['is_home'].values.reshape(-1,1)
            else:
                is_home = prev_history['hfa'].values.reshape(-1,1)

            X = np.concatenate([team_one_hot, opp_one_hot, is_home], axis=1)
            y = prev_history[self.result_col].values.reshape(-1,1)
            ridge = RidgeRegression(alpha=self.alpha)
            ridge.fit(X,y)

            protag_names = team_ohe.get_feature_names_out()
            ## strip x0_ from names
            protag_names = [x[3:] for x in protag_names]
            antag_names = opp_ohe.get_feature_names_out()
            ## strip x0_ from names
            antag_names = [x[3:] for x in antag_names]
            protag_map = {pn:ridge.coef_[0][i] for i,pn in enumerate(protag_names)}
            antag_map = {an:ridge.coef_[0][len(protag_names)+i] for i,an in enumerate(antag_names)}

            to_append = self.data[[self.protag_id,self.antag_id,'is_home',self.result_col,'stat',self.period_col]][self.data[self.period_col]==unique_period].copy()
            to_append['protag_rating'] = to_append[self.protag_id].apply(lambda x: protag_map.get(x))
            to_append['antag_rating'] = -1*to_append[self.antag_id].apply(lambda x: antag_map.get(x))

            pregame_ratings.append(to_append)
        
        pregame_ratings = pd.concat(pregame_ratings, axis=0)
        pregame_ratings = pregame_ratings.sort_values(by=[self.period_col,self.protag_id]).reset_index(drop=True)

        return pregame_ratings
    
    def update(self):

        return





