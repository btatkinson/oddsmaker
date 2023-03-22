import gc
import os

import numpy as np
import pandas as pd

from copy import copy
from tqdm import tqdm
from bayes_opt import BayesianOptimization
# from scipy.optimize import minimize, dual_annealing
from pandas.api.types import is_datetime64_any_dtype as is_datetime

class Elo():
    
    """
    
    Implements classic Elo algorithm for a single stat or multiple stats simultaneously
    
    """
    
    
    def __init__(self, 
                 data, 
                 k,
                 hfa,
                 protag_id='team_name', 
                 antag_id='opp_name',
                 result_col='result',
                 priors = None
                ):
        
        self.data = data
        self.protag_id = protag_id
        self.antag_id = antag_id
        assert('stat' in list(self.data)), 'No stat column'
        self.stats = sorted(list(self.data.stat.unique()))
        self.result_col = result_col
        self.priors = priors
        
        ### determine format of provided k values
        if type(k)==dict:
            assert(key in self.stats for key in k.keys()), "each key in k value dict must be a stat"
            kval_not_provided = []
            for stat in self.stats:
                if stat not in k:
                    print(f"No k value provided for {stat}, using average of other kvalues...")
                    kval_not_provided.append(stat)
            self.k = k
            for stat in kval_not_provided:
                self.k[stat] = np.mean(self.k.values())
                
        elif ((isinstance(k, int))|(isinstance(k, float))):
            self.k = {}
            for stat in self.stats:
                self.k[stat] = k
        else:
            raise ValueError("K values must either be a numeric (to assign to all stats) or a dict (where keys are stat names, values to be applied individually)")
        
        ### determine format of provided home field advantages
        if type(hfa)==dict:
            assert(key in self.stats for key in hfa.keys()), "each key in home field advantage dict must be a stat"
            hfa_not_provided = []
            for stat in self.stats:
                if stat not in hfa:
                    print(f"No home field advantage provided for {stat}")
            self.hfa = hfa
        elif ((isinstance(hfa, int))|(isinstance(hfa, float))):
            self.hfa = {}
            for stat in self.stats:
                self.hfa[stat] = hfa
        else:
            raise ValueError("Home field advantage must either be a numeric (to assign to all stats) or a dict (where keys are stat names, values to be applied individually)")
        
        ### check if results are in binary format (or tie)
        results = list(self.data[self.result_col].unique())
        assert(all([(np.isclose(r,0)|(np.isclose(r,1)|(np.isclose(r,0.5)))) for r in results])), "Results must be zero (for loss) or one (for win) or 0.5 (for tie)"
        
        ### check for home field advantage
        if 'is_home' not in list(self.data):
            self.data['is_home']=0
            for stat in self.stats:
                self.hfa[stat] = 0
        locs = (self.data['is_home'].unique())
        assert(all([(np.isclose(l,0)|(np.isclose(l,1)|(np.isclose(l,-1)))) for l in locs])), "is_home col needs either 1 for home, -1 for away, or 0 for neutral"
        
        ### check for rating periods or dates
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
        
        ### initialize rating tracking and ids
        self.protag_ids = set(self.data[self.protag_id].unique())
        self.antag_ids = set(self.data[self.antag_id].unique())
        
        assert(len(self.protag_ids.symmetric_difference(self.antag_ids))==0), "In SPR format, need a row for each team in dataframe (two rows per game)"
        
        ### useful meta information
        self.num_stats = len(self.stats)
        self.num_protags = len(self.protag_ids)
        self.num_games = len(self.data)//2
        
        ### index maps
        self.protag2index = {}
        for i,protag_id in enumerate(self.protag_ids):
            self.protag2index[protag_id] = i
            
        self.stat2index = {}
        for j,stat in enumerate(self.stats):
            self.stat2index[stat] = j
            
        ### initialize ratings
        self.data['protag_idx'] = self.data[self.protag_id].map(self.protag2index)
        self.data['antag_idx'] = self.data[self.antag_id].map(self.protag2index)
        self.data['stat_idx'] = self.data['stat'].map(self.stat2index)
        self.data['hfa'] = self.data['stat'].map(self.hfa).copy()*self.data['is_home'].copy()
        assert(len(self.data.loc[self.data.hfa.isnull()])==0), f"{self.data.loc[self.data.hfa.is_null()].stat.unique()} do not have a home field advantage number"
        
        self.data['k'] = self.data['stat'].map(self.k).copy()
        assert(len(self.data.loc[self.data.k.isnull()])==0), f"{self.data.loc[self.data.k.is_null()].stat.unique()} do not have a k factor specified in the k factor dict"
        
        self.rating_matrix = np.ones((self.num_protags, self.num_stats))*1500
        self.history = "Use .run_history() to create history"
        
        ### add priors for those specified
        ### priors must be a dict in form of {protag_id:{stat:rating}}
        if priors is not None:
            assert(type(priors)==dict), "Priors must be a dict"
            for protag_id, stat_dict in priors.items():
                assert(type(stat_dict)==dict), "Each protag_id key in priors dict must be a dict of stat:rating pairs"
                for stat, rating in stat_dict.items():
                    assert(stat in self.stats), f"{stat} is not a stat in the dataset"
                    self.rating_matrix[self.protag2index[protag_id], self.stat2index[stat]] = rating

        return
    
    def _opt_helper(self, **kwargs):
        """
        needed to be compatible with the optimization library
        """
        k = {}
        hfa = {}
        
        for key,value in kwargs.items():
            if '_hfa' in key:
                key_name = copy(key).replace('_hfa','') ## appended numbers to distinguish
                hfa[key_name] = value
            elif '_kval' in key:
                key_name = copy(key).replace('_kval','')
                k[key_name] = value
            
        _, grade = self.run_history(k=k, hfa=hfa)
        return -grade # optimizer maximizes, so need to take negative
    
    def _rating_period_update(self, protag_ratings, antag_ratings, k, results):
        """
        performs classic Elo rating update calculation
        """
        
        probs = 1/(1+10**((antag_ratings-protag_ratings)/400))
        return k*(results - probs)
    
    def _reset_ratings(self, old_data=None, priors=None):
        """
        resets rating matrix to last known rating
        """
        self.rating_matrix = np.ones((self.num_protags, self.num_stats))*1500
        old_ratings = old_data.drop_duplicates(subset=['protag_idx','stat_idx'], keep='last').copy()
        for i,row in old_ratings.iterrows():
            self.rating_matrix[row['protag_idx'], row['stat_idx']] = row['pregame_rating'] + row['rating_adjustment']

        for key, value in priors.items():
            for stat, rating in value.items():
                self.rating_matrix[self.protag2index[key], self.stat2index[stat]] = rating

        ### reset history
        if 'date' in list(old_data):
            self.history = self.history.loc[self.history['date']<old_data['date'].max()].reset_index(drop=True)
        else:
            self.history = self.history.loc[self.history['rating_period']<old_data['rating_period'].max()].reset_index(drop=True)

        return
    
    def _update_rating_matrix(self, new_ids, priors=None):

        """
        After new ids are added, need to update the rating matrix and the protag2index map
        """

        ## update maps
        max_index = np.max(self.protag2index.values())
        for i,new_id in enumerate(new_ids):
            self.protag2index[new_id] = max_index + i + 1

        new_ratings = np.ones((len(new_ids), self.num_stats))*1500
        self.rating_matrix = np.vstack((self.rating_matrix, new_ratings))

        if priors is not None:
            assert(type(priors)==dict), "priors must be a dictionary"
            for protag_id, stat_dict in priors.items():
                assert(type(stat_dict)==dict), "Each protag_id key in priors dict must be a dict of stat:rating pairs"
                for stat, rating in stat_dict.items():
                    assert(stat in self.stats), f"{stat} is not a stat in the dataset"
                    self.rating_matrix[self.protag2index[protag_id], self.stat2index[stat]] = rating

        print("New teams have been added.")
        return
    
    def info(self):
        """
        returns meta info, usually used right after initialization
        """
        
        print(f"There are {self.num_stats} stats: {self.stats}")
        print(f"There are {self.num_protags:,} unique players/teams.")
        print(f"There are {self.num_games:,} games from {self.data.date.min()} to {self.data.date.max()}.")
        
        return
    
    def current_ratings(self):
        """
        returns current ratings
        """
        ratings_long = self.rating_matrix.reshape(-1)
        idx_long = np.repeat(sorted(self.protag2index.values()), self.num_stats)
        stat_idx_long = np.tile(sorted(self.stat2index.values()), self.num_protags)
        ratings = pd.DataFrame({
            'protag_idx':idx_long,
            'stat_idx':stat_idx_long,
            'rating':ratings_long
        })
        self.idx2protag = {v:k for k,v in self.protag2index.items()}
        self.idx2stat = {v:k for k,v in self.stat2index.items()}
        ratings['protag_id'] = ratings['protag_idx'].map(self.idx2protag)
        ratings['stat'] = ratings['stat_idx'].map(self.idx2stat)
        
        return ratings
    
    def run_history(self, data=None, k=None, hfa=None, metric='brier'):
        """
        calculates entire pre-game (non-leaky) ratings using current parameters
        
        returns: those ratings
        """
        
        assert(metric in ['brier','log_loss']), 'please use an implemented metric (brier, log_loss)' 
        if k is None:
            k = self.k
        if hfa is None:
            hfa = self.hfa
        if data is None:
            data = self.data.copy()
        
        history = []
        quick_iterator = data.groupby(['rating_period'])
        for rp_index, rating_period in tqdm(quick_iterator, total=len(quick_iterator)):
            
            ## append pregame ratings to history
            pregame_protag_ratings = self.rating_matrix[rating_period.protag_idx.values, rating_period.stat_idx.values]
            pregame_antag_ratings = self.rating_matrix[rating_period.antag_idx.values, rating_period.stat_idx.values]

            to_append = rating_period[['date',self.protag_id,self.antag_id,'is_home','hfa','stat',self.result_col]].copy()
            to_append['pregame_rating'] = pregame_protag_ratings
            to_append['pregame_opp_rating'] = pregame_antag_ratings
            
            ## account for hfa
            pregame_protag_ratings = pregame_protag_ratings+(rating_period.hfa.values/2)
            pregame_antag_ratings = pregame_antag_ratings-(rating_period.hfa.values/2)

            rating_adjustments = self._rating_period_update(pregame_protag_ratings, pregame_antag_ratings, rating_period.k.values, rating_period.result.values)
            
            ## reset ratings
            pregame_protag_ratings = pregame_protag_ratings-(rating_period.hfa.values/2)
            pregame_antag_ratings = pregame_antag_ratings+(rating_period.hfa.values/2)

            ## apply update
            new_ratings = pregame_protag_ratings+rating_adjustments
            to_append['rating_adjustments'] = rating_adjustments
            
            history.append(to_append)
            
            self.rating_matrix[rating_period.protag_idx.values, rating_period.stat_idx.values] = new_ratings

        if type(self.history) == pd.DataFrame:
            self.history = pd.concat([self.history, pd.concat(history, axis=0).reset_index(drop=True)], axis=0).reset_index(drop=True)
        else:
            self.history = pd.concat(history, axis=0).reset_index(drop=True)
        
        self.history['rtg_diff'] = self.history['pregame_opp_rating'].copy()-(self.history['pregame_rating'].copy()+(self.history['is_home'].copy()*self.history.hfa.values))
        self.history['probability'] = 1/(1+10**((self.history['rtg_diff'])/400))
        if metric == 'brier':
            ## allow some time for ratings to stabilize
            initial = int(0.15*len(self.history))
            score = ((self.history[initial:]['result'].copy()-self.history[initial:]['probability'].copy())**2).mean()
        self.history = self.history.drop(columns=['rtg_diff'])
        
        return self.history, score
    
    
    def optimize(self, 
                 init_points=20, 
                 n_iter=250, 
                 k_lower=5, 
                 k_upper=25, 
                 hfa_lower=0, 
                 hfa_upper=65, 
                 random_state=17,
                 update_params = True
                ):
        """
        optimizes k value and home field advantage
        """
        
        pbounds = {}
        for stat in self.stats:
            pbounds[stat+'_hfa'] = (hfa_lower, hfa_upper)
            pbounds[stat+'_kval'] = (k_lower, k_upper)
            
        self.optimizer = BayesianOptimization(
            f=self._opt_helper,
            pbounds=pbounds,
            random_state=random_state,
        )
        
        self.optimizer.maximize(
            init_points=init_points,
            n_iter=n_iter
        )
        
        print("\nBest Values:")
        print(self.optimizer.max)
        print("\n")
        
        if update_params:
            print("Updating params to optimized values...")
            self.k = {}
            self.hfa = {}
            for param, value in self.optimizer.max['params'].items():
                if '_hfa' in param:
                    stat_name = param.replace('_hfa','')
                    self.hfa[stat_name] = value
                elif '_kval' in param:
                    stat_name = param.replace('_kval','')
                    self.k[stat_name] = value
                    
        gc.collect()
        return 
    
    def _check_for_new_ids(self, new_data, priors):

        ### check for new ids
        new_ids = set(new_data[self.protag_id].unique()).union(set(new_data[self.antag_id].unique()))
        old_ids = self.protag_ids.union(self.antag_ids)
        new_ids = new_ids.difference(old_ids)
        if len(new_ids)>0:
            print(f"Found {len(new_ids)} new ids, adding them to the rating matrix...")
            self.protag_ids = self.protag_ids.union(new_ids)
            self.antag_ids = self.antag_ids.union(new_ids)
            self._update_rating_matrix(new_ids, priors)
        return
    
    def update(self, new_data, affirm_update=True, priors=None):


        ### check that no new stats have been added
        new_stats = set(new_data.stat.unique())
        old_stats = set(self.stats)
        new_stats = new_stats.difference(old_stats)
        assert(len(new_stats)==0), f"New stats found: {new_stats}. This class is not currently able to handle new stats added via update. Please re-initialize starting from beginning. Feel free to submit request to add this functionality"
        
        col_names= list(new_data)
        if 'date' in col_names:
            ### use date
            date_dtype = new_data.date.dtype
            assert(is_datetime(date_dtype)), "Date column must be of type datetime"
            oldest_date = new_data.date.min()
            assert('date' in list(self.date)), "Old data does not contain date column, new data does. Confused whether to use rating period or date."
            print(f"Oldest date in new data is {oldest_date}, starting update from that date...")

            if affirm_update:
                print("Proceed, Y/N?")
                proceed = input()
                if proceed.lower() not in ['y', 'yes']:
                    print("Update aborted, if affirming update, please type 'Y' or 'Yes")
                    return
            
            ### check for new ids
            self._check_for_new_ids(new_data, priors)
            self.data = pd.concat([self.data, new_data], axis=0)
            
            self.data = self.data.sort_values(by='date').reset_index(drop=True)
            self.data['rating_period'] = self.data.date.copy().rank(method='dense')
            old_data = self.data.copy().loc[self.data['date']<oldest_date].reset_index(drop=True) 
            new_data = self.data.copy().loc[self.data['date']>=oldest_date].reset_index(drop=True)

            
        else:
            ### use rating period instead
            oldest_rp = new_data.rating_period.min()
            assert('rating_period' in list(new_data)), "New data does not contain rating period or date column. Need one or the other to update."
            print(f"Oldest rating period in new data is {oldest_rp}, starting update from that rating period...")

            if affirm_update:
                print("Proceed, Y/N?")
                proceed = input()
                if proceed.lower() not in ['y', 'yes']:
                    print("Update aborted, if affirming update, please type 'Y' or 'Yes")
                    return
                
            ### check for new ids
            self._check_for_new_ids(new_data, priors)
            self.data = pd.concat([self.data, new_data], axis=0)
            self.data = self.data.sort_values(by='rating_period').reset_index(drop=True)
            old_data = self.data.copy().loc[self.data['rating_period']<oldest_rp].reset_index(drop=True)
            new_data = self.data.copy().loc[self.data['rating_period']>=oldest_rp].reset_index(drop=True)

        ### reset ratings matrix to prior to new data
        self._reset_ratings(old_data, priors)

        ### update ratings
        self.run_history(data=new_data)
        

        
        return
    
    def predict(self):
        
        return
    
    
