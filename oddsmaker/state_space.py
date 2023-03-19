import os

import numpy as np
import pandas as pd

from tqdm import tqdm
from pandas.api.types import is_datetime64_any_dtype as is_datetime

class EloClassic():
    
    """
    
    Implements classic Elo algorithm for multiple stats
    
    """
    
    
    def __init__(self, 
                 data, 
                 k=10, 
                 protag_id='team_name', 
                 antag_id='opp_name', 
                 hfa=True, 
                 result_col='result',
                 priors = None
                ):
        
        self.data = data
        self.k=k
        self.protag_id = protag_id
        self.antag_id = antag_id
        assert('stat' in list(self.data)), 'No stat column'
        self.stats = list(self.data.stat.unique())
        self.hfa = hfa
        self.result_col = result_col
        self.priors = priors
        
        results = list(self.data[self.result_col].unique())
        assert(all([(np.isclose(r,0)|(np.isclose(r,1)|(np.isclose(r,0.5)))) for r in results])), "Results must be zero (for loss) or one (for win) or 0.5 (for tie)"
        
        col_names = list(self.data)
        col_names = [cn.lower().strip() for cn in col_names]
        assert((('date' in col_names)|('rating_period' in col_names))), "Need either a date column or a rating period column"
        self.data.columns=col_names
        
        if 'date' in col_names:
            date_dtype = self.data.date.dtype
            assert(is_datetime(date_dtype)), "Date column must be of type datetime"
        else:
            date_dtype_check = True
        
        if (('date' in col_names) & ('rating_period' not in col_names)):
            self.data['rating_period'] = self.data.date.copy().rank(method='dense')

        self.protag_ids = set(self.data[self.protag_id].unique())
        self.antag_ids = set(self.data[self.antag_id].unique())
        
        assert(len(self.protag_ids.symmetric_difference(self.antag_ids))==0), "In SPR format, need a row for each team in dataframe (two rows per game)"
        
        self.num_stats = len(self.stats)
        self.num_protags = len(self.protag_ids)
        self.num_games = len(self.data)//2
        
        self.protag2index = {}
        for i,protag_id in enumerate(self.protag_ids):
            self.protag2index[protag_id] = i
            
        self.stat2index = {}
        for j,stat in enumerate(self.stats):
            self.stat2index[stat] = j
            
        self.data['protag_idx'] = self.data[self.protag_id].map(self.protag2index)
        self.data['antag_idx'] = self.data[self.antag_id].map(self.protag2index)
        self.data['stat_idx'] = self.data['stat'].map(self.stat2index)
        
        self.rating_matrix = np.ones((self.num_protags, self.num_stats))*1500
        
        return
    
    def _rating_period_update(self, protag_ratings, antag_ratings, results):
        probs = 1/(1+10**((antag_ratings-protag_ratings)/400))
        return protag_ratings + self.k*(results - probs)
    
    def info(self):
        
        print(f"There are {self.num_stats} stats: {self.stats}")
        print(f"There are {self.num_protags} unique players/teams.")
        print(f"There are {self.num_games} games.")
        
        return
    
    
    def history(self):
        
        history = []
        quick_iterator = self.data.groupby(['rating_period'])
        for rp_index, rating_period in tqdm(quick_iterator, total=len(quick_iterator)):
            
            ## append pregame ratings to history
            pregame_protag_ratings = self.rating_matrix[rating_period.protag_idx.values, rating_period.stat_idx.values]
            pregame_antag_ratings = self.rating_matrix[rating_period.antag_idx.values, rating_period.stat_idx.values]

            to_append = rating_period[['date',self.protag_id,self.antag_id,'stat',self.result_col]].copy()
            to_append['pregame_rating'] = pregame_protag_ratings
            to_append['pregame_opp_rating'] = pregame_antag_ratings
            
            new_ratings = self._rating_period_update(pregame_protag_ratings, pregame_antag_ratings, rating_period.result.values)
            
            history.append(to_append)
            
            self.rating_matrix[rating_period.protag_idx.values, rating_period.stat_idx.values] = new_ratings

        self.history = pd.concat(history, axis=0).reset_index(drop=True)
        return self.history
    
    
    def optimize(self):
        """
        optimizes k value and home field advantage
        """
        
        return

