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

    Parameters
    ----------
    data : pandas dataframe
        Dataframe containing the following columns:
            - protag_id: id of protagonist (player or team)
            - antag_id: id of antagonist (player or team)
            - stat: name of stat
            - result: 0 for loss, 1 for win, 0.5 for tie
            - is_home: 1 for home, -1 for away, 0 for neutral
            - date: date of game
            - rating_period: rating period of game (if no date)
    k : int, float, or dict
        Elo k value(s) to use for each stat. If int or float, applies to all stats. If dict, keys must be stat names, values are k values for each stat
    hfa : int, float, or dict
        Home field advantage value(s) to use for each stat. If int or float, applies to all stats. If dict, keys must be stat names, values are hfa values for each stat
    protag_id : str, optional
        Name of protagonist id column in data. The default is 'team_name'.
    antag_id : str, optional
        Name of antagonist id column in data. The default is 'opp_name'.
    result_col : str, optional
        Name of result column in data. The default is 'result'.
    priors : dict, optional
        Dictionary of prior ratings to use for each stat. Keys are protag_ids, values are dicts of stat:rating pairs. The default is None.      
    
    """
    
    
    def __init__(self, 
                 data, 
                 k,
                 hfa=None,
                 protag_id='team_name', 
                 antag_id='opp_name',
                 result_col='result',
                 priors = None
                ):
        
        self.data = data.copy()
        self.protag_id = protag_id
        self.antag_id = antag_id

        assert(self.protag_id in list(data)), f"{self.protag_id} not in columns, feel free to specify team column name with protag_id argument"
        assert(self.antag_id in list(data)), f"{self.antag_id} not in columns, feel free to specify opponent column name with antag_id argument"

        assert('stat' in list(self.data)), 'No stat column, please add a stat name column to your data '
        self.stats = sorted(list(self.data.stat.unique()))
        self.result_col = result_col
        assert(self.result_col in list(self.data)), 'Please include an outcome/result column, can specify the name with result col argument'
        self.priors = priors
        
        ## initialize k and hfa as None to help _add methods know it is still initializing
        self.k = None
        self._add_k(k)

        self.hfa = None
        self._add_hfa(hfa)
        
        ### check if results are in binary format (or tie)
        results = list(self.data[self.result_col].unique())
        assert(all([(np.isclose(r,0)|(np.isclose(r,1)|(np.isclose(r,0.5)))) for r in results])), "Results must be zero (for loss) or one (for win) or 0.5 (for tie)"

        ## add rating period if only date is provided
        self._add_rating_period()
        
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
        self.data['protag_idx'] = self.data[self.protag_id].copy().map(self.protag2index)
        self.data['antag_idx'] = self.data[self.antag_id].copy().map(self.protag2index)
        self.data['stat_idx'] = self.data['stat'].copy().map(self.stat2index)
        self.data['hfa'] = self.data['stat'].copy().map(self.hfa).copy()*self.data['is_home'].copy()
        assert(len(self.data.loc[self.data.hfa.isnull()])==0), f"{self.data.loc[self.data.hfa.is_null()].stat.unique()} do not have a home field advantage number"
        
        self.history = "Use .run_history() to create history"
        
        self.reset_ratings_mat()

        return
    
    def reset_ratings_mat(self):

        self.rating_matrix = np.ones((self.num_protags, self.num_stats))*1500
        
        ### add priors for those specified
        ### priors must be a dict in form of {protag_id:{stat:rating}}
        if self.priors is not None:
            assert(type(self.priors)==dict), "Priors must be a dict"
            for protag_id, stat_dict in self.priors.items():
                assert(type(stat_dict)==dict), "Each protag_id key in priors dict must be a dict of stat:rating pairs"
                for stat, rating in stat_dict.items():
                    assert(stat in self.stats), f"{stat} is not a stat in the dataset"
                    self.rating_matrix[self.protag2index[protag_id], self.stat2index[stat]] = rating
        return
    
    def _add_k(self, k=None, data=None):

        """
        
        Adds k to self.data if none is passed, otherwise adds hfa to passed data
        
        """
        
        if self.k is None:
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
        if data is None:
            self.data['k'] = self.data['stat'].map(self.k).copy()
            assert(len(self.data.loc[self.data.k.isnull()])==0), f"{self.data.loc[self.data.k.is_null()].stat.unique()} do not have a k factor specified in the k factor dict"
        else:
            data['k'] = data['stat'].map(self.k).copy()
            assert(len(data.loc[data.k.isnull()])==0), f"{data.loc[data.k.is_null()].stat.unique()} do not have a k factor specified in the k factor dict"
            return data
        
        return
    
    def _add_hfa(self, hfa=None, data=None):

        """
        
        Adds home field advantage to self.data if none is passed, otherwise adds hfa to passed data and returns it
        
        """
        

        if data is None:
            ### apply to self.data
            ### check for home field advantage
            if 'is_home' not in list(self.data):
                ### if no home field advantage column, assume no home field advantage
                self.data['is_home']=0
                self.has_hfa = False
            else:
                self.has_hfa = True

            locs = (self.data['is_home'].unique())
            assert(all([(np.isclose(l,0)|(np.isclose(l,1)|(np.isclose(l,-1)))) for l in locs])), "is_home col needs either 1 for home, -1 for away, or 0 for neutral"
            
            if self.hfa is None:
                ### determine format of provided home field advantages
                if ((hfa is None)|(hfa==0)):
                    self.hfa = {}
                    for stat in self.stats:
                        self.hfa[stat] = 0
                elif type(hfa)==dict:
                    assert(key in self.stats for key in hfa.keys()), "each key in home field advantage dict must be a stat"
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
                
        else:
            ### apply to passed data
            if 'is_home' not in list(data):
                data['is_home']=0
            if data.is_home.isnull().any():
                data['is_home'].fillna(0, inplace=True)
            locs = (data['is_home'].unique())
            assert(all([(np.isclose(l,0)|(np.isclose(l,1)|(np.isclose(l,-1)))) for l in locs])), "is_home col needs either 1 for home, -1 for away, or 0 for neutral"

        if data is None:
            self.data['hfa'] = self.data['stat'].map(self.hfa).copy()*self.data['is_home'].copy()
            assert(len(self.data.loc[self.data.hfa.isnull()])==0), f"{self.data.loc[self.data.hfa.is_null()].stat.unique()} do not have a home field advantage number"
        else:
            data['hfa'] = data['stat'].map(self.hfa).copy()*data['is_home'].copy()
            assert(len(data.loc[data.hfa.isnull()])==0), f"{data.loc[data.hfa.isnull()].stat.unique()} do not have a home field advantage number"
            return data

        return
    
    def _add_rating_period(self, data=None):

        """
        
        Adds rating period to self.data if none is passed, otherwise adds rating period to passed data and returns it
        
        """

        if data is None:

            ### check for rating periods or dates
            col_names = list(self.data)
            col_names = [cn.lower().strip() for cn in col_names]
            assert((('date' in col_names)|('rating_period' in col_names))), "Need either a date column or a rating period column"

            self.data.columns=col_names
            if 'date' in col_names:
                self.has_date = True
                date_dtype = self.data.date.dtype
                assert(is_datetime(date_dtype)), "Date column must be of type datetime"
            else:
                self.has_date = False

            if (('date' in col_names) & ('rating_period' not in col_names)):
                self.data['rating_period'] = self.data.date.copy().rank(method='dense')
        else:  
            ### check for rating periods or dates
            col_names = list(data)
            col_names = [cn.lower().strip() for cn in col_names]
            assert((('date' in col_names)|('rating_period' in col_names))), "Need either a date column or a rating period column"

            data.columns=col_names
            if 'date' in col_names:
                assert(self.has_date), "Data has date column, but self.data does not"
                date_dtype = data.date.dtype
                assert(is_datetime(date_dtype)), "Date column must be of type datetime"

            if (('date' in col_names) & ('rating_period' not in col_names)):
                data['rating_period'] = data.date.copy().rank(method='dense')

            return data
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
            
        if self.has_hfa:
            _, grade = self.run_history(k=k, hfa=hfa)
        else:
            _, grade = self.run_history(k=k)
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
        old_data = old_data.drop_duplicates(subset=[self.protag_id,'stat'], keep='last').copy()

        old_data['protag_idx'] = old_data[self.protag_id].map(self.protag2index)
        old_data['stat_idx'] = old_data['stat'].map(self.stat2index)
        old_data[['protag_idx','stat_idx']] = old_data[['protag_idx','stat_idx']].astype(int)

        ## merge in history
        old_data = old_data.merge(self.history, on=['rating_period',self.protag_id,'stat'], how='left')

        assert(old_data['pregame_rating'].isnull().sum()==0)
        assert(old_data['rating_adjustment'].isnull().sum()==0)

        for i,row in old_data.iterrows():
            self.rating_matrix[row['protag_idx'], row['stat_idx']] = row['pregame_rating'] + row['rating_adjustment']
        if priors is not None:
            assert(type(priors)==dict), "priors must be a dictionary"
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
        max_index = np.max(list(self.protag2index.values()))
        for i,new_id in enumerate(new_ids):
            self.protag2index[new_id] = int(max_index + i + 1)
        if len(new_ids) > 0:
            new_ratings = np.ones((len(new_ids), self.num_stats))*1500
            # print(f"Old shape: {self.rating_matrix.shape}")
            self.rating_matrix = np.vstack((self.rating_matrix.copy(), new_ratings))
            # print(f"New shape: {self.rating_matrix.shape}")
            if priors is not None:
                assert(type(priors)==dict), "priors must be a dictionary"
                for protag_id, stat_dict in priors.items():
                    assert(type(stat_dict)==dict), "Each protag_id key in priors dict must be a dict of stat:rating pairs"
                    for stat, rating in stat_dict.items():
                        assert(stat in self.stats), f"{stat} is not a stat in the dataset"
                        self.rating_matrix[self.protag2index[protag_id], self.stat2index[stat]] = rating

            else:
                print(f"Warning: new ids found: {new_ids}, but no priors provided. Using default priors.")

            print("New teams/players have been added.")
        assert(not np.isnan(np.sum(self.rating_matrix)))
        return
    
    
    def info(self):
        """
        returns meta info, usually used right after initialization
        """
        
        print(f"There are {self.num_stats} stats: {self.stats}")
        print(f"There are {self.num_protags:,} unique players/teams.")
        if 'date' in self.data.columns:
            print(f"There are {self.num_games:,} games from {self.data.date.min()} to {self.data.date.max()}.")
        else:
            print(f"There are {self.num_games:,} games over {self.data.rating_period.max()-self.data.rating_period.min()} rating periods.")
        
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
        ratings[self.protag_id] = ratings['protag_idx'].map(self.idx2protag)
        ratings['stat'] = ratings['stat_idx'].map(self.idx2stat)
        
        return ratings.drop(columns=['protag_idx','stat_idx'])
    
    def run_history(self, data=None, k=None, hfa=None, metric='brier', is_update=False):
        """
        calculates entire pre-game (non-leaky) ratings using current parameters
        
        returns: those ratings
        """
        
        assert(metric in ['brier','log_loss']), 'please use an implemented metric (brier, log_loss)' 

        if data is None:
            data = self.data.copy()

        if hfa is None:
            hfa = self.hfa
        elif type(hfa)==dict:
            data['hfa'] = data['stat'].map(hfa).copy()
        elif type(hfa)==int:
            data['hfa'] = hfa
        else:
            raise ValueError("hfa must be a dictionary or an integer")

        if k is None:
            k = self.k
        elif type(k)==dict:
            data['k'] = data['stat'].map(k).copy()
        elif type(k)==int:
            data['k'] = k
        else:
            raise ValueError("k must be a dictionary or an integer")
        
        if is_update:
            if data is None:
                raise ValueError("If is_update=True, data must be provided.")
        else:
            self.reset_ratings_mat()

        history = []
        quick_iterator = data.groupby(['rating_period'])
        for rp_index, rating_period in tqdm(quick_iterator, total=len(quick_iterator)):
            
            # if rp_index == 1:
            #     print(rating_period)

            ## append pregame ratings to history
            pregame_protag_ratings = self.rating_matrix[rating_period.protag_idx.values, rating_period.stat_idx.values]
            pregame_antag_ratings = self.rating_matrix[rating_period.antag_idx.values, rating_period.stat_idx.values]
            if self.has_date:
                to_append = rating_period[['date','rating_period',self.protag_id,self.antag_id,'is_home','hfa','stat',self.result_col]].copy()
            else:
                to_append = rating_period[['rating_period',self.protag_id,self.antag_id,'is_home','hfa','stat',self.result_col]].copy()
            to_append['pregame_rating'] = pregame_protag_ratings
            to_append['pregame_opp_rating'] = pregame_antag_ratings
            
            ## account for hfa
            pregame_protag_ratings = pregame_protag_ratings+(rating_period.hfa.values/2)
            pregame_antag_ratings = pregame_antag_ratings-(rating_period.hfa.values/2)

            rating_adjustments = self._rating_period_update(pregame_protag_ratings, pregame_antag_ratings, rating_period.k.values, rating_period[self.result_col].values)
            
            ## reset ratings
            pregame_protag_ratings = pregame_protag_ratings-(rating_period.hfa.values/2)
            pregame_antag_ratings = pregame_antag_ratings+(rating_period.hfa.values/2)

            ## apply update
            new_ratings = pregame_protag_ratings+rating_adjustments
            to_append['rating_adjustment'] = rating_adjustments
            
            history.append(to_append)
            
            self.rating_matrix[rating_period.protag_idx.values, rating_period.stat_idx.values] = new_ratings

        if is_update:
            self.history = pd.concat([self.history, pd.concat(history, axis=0).reset_index(drop=True)], axis=0).reset_index(drop=True)
        else:
            self.history = pd.concat(history, axis=0).reset_index(drop=True)
        
        self.history['rtg_diff'] = self.history['pregame_opp_rating'].copy()-(self.history['pregame_rating'].copy()+(self.history['is_home'].copy()*self.history.hfa.values))
        self.history['probability'] = 1/(1+10**((self.history['rtg_diff'])/400))
        if metric == 'brier':
            ## allow some time for ratings to stabilize
            initial = int(0.15*len(self.history))

            score = ((self.history[initial:][self.result_col].copy()-self.history[initial:]['probability'].copy())**2).mean()
        self.history = self.history.drop(columns=['rtg_diff'])

        return self.history, score
    
    
    def optimize(self, 
                 init_points=10, 
                 n_iter=50, 
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
            if self.has_hfa:
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
            if self.has_hfa:
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
    
    def update(self, new_data, affirm_update=False, priors=None):

        if type(self.history) == str:
            print("Please use run_history() function before updating with new data. Consider combining new and old data and running all at once.")
            return
        
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
            assert('date' in list(self.data)), "Old data does not contain date column, new data does. Confused whether to use rating period or date."
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

        ### convert ids to indices
        new_data['protag_idx'] = new_data[self.protag_id].apply(lambda x: self.protag2index[x]).astype(int)
        new_data['antag_idx'] = new_data[self.antag_id].apply(lambda x: self.protag2index[x]).astype(int)
        new_data['stat_idx'] = new_data['stat'].apply(lambda x: self.stat2index[x]).astype(int)
        self._add_k(data=new_data)
        self._add_hfa(data=new_data)

        assert(new_data[['protag_idx', 'antag_idx', 'stat_idx']].isnull().sum().sum()==0), "Error converting ids to indices"
        ### reset ratings matrix to prior to new data
        self._reset_ratings(old_data, priors)
        ### update ratings
        self.run_history(data=new_data, is_update=True)

        return
    
    def predict(self, upcoming_matches, priors=None):

        ### bunch of validation checks ###
        assert(type(self.history) != str), "Please run run_history() function before predicting"
        assert(type(upcoming_matches) == pd.DataFrame), "upcoming_matches must be a pandas dataframe"
        assert(self.protag_id in list(upcoming_matches)), f"upcoming_matches must contain a {self.protag_id} column, like in the original data"
        assert(self.antag_id in list(upcoming_matches)), f"upcoming_matches must contain a {self.antag_id} column, like in the original data"
        assert('stat' in list(upcoming_matches)), f"upcoming_matches must contain a stat column"
        col_names = list(upcoming_matches)
        if 'date' in col_names:
            ### use date
            date_dtype = upcoming_matches.date.dtype
            assert(is_datetime(date_dtype)), "Date column must be of type datetime"
            oldest_date = upcoming_matches.date.min()
            assert('date' in list(self.data)), "Old data does not contain date column, new data does. Confused whether to use rating period or date."
            if oldest_date < self.data.date.max():
                print(f"Warning: oldest date in new data is {oldest_date}, which is before the most recent date in the old data. This will result in a prediction using the old data. Consider chronological ordering of data.")
        else:
            ### use rating period instead
            assert('rating_period' in list(upcoming_matches)), "New data does not contain rating period or date column. Need one or the other to update."
            oldest_rp = upcoming_matches.rating_period.min()
            if oldest_rp < self.data.rating_period.max():
                print(f"Warning: oldest rating period in new data is {oldest_rp}, which is before the most recent rating period in the old data. This will result in a prediction using the old data. Consider chronological ordering of data.")

        ### check that no new stats have been added
        new_stats = set(upcoming_matches.stat.unique())
        old_stats = set(self.stats)
        new_stats = new_stats.difference(old_stats)
        assert(len(new_stats)==0), f"New stats found: {new_stats}. This class is not currently able to handle new stats added via update. Please re-initialize starting from beginning. Feel free to submit request to add this functionality"

        ### check that no new ids have been added
        new_ids = set(upcoming_matches[self.protag_id].unique())
        new_ids = new_ids.union(set(upcoming_matches[self.antag_id].unique()))
        old_ids = set(self.protag_ids.union(self.antag_ids))
        new_ids = new_ids.difference(old_ids)
        print(f"{len(new_ids)} new ids found.")

        if len(new_ids) > 0:
            self._update_rating_matrix(new_ids, priors)
        
        ### run prediction
        curr_ratings = self.current_ratings().rename(columns={'rating':'protag_rating'})
        antag_curr_ratings = curr_ratings.copy().rename(columns={self.protag_id:self.antag_id, 'rating':'antag_rating'})

        upcoming_matches = pd.merge(upcoming_matches, curr_ratings, on=self.protag_id, how='left')
        upcoming_matches = pd.merge(upcoming_matches, antag_curr_ratings, on=self.antag_id, how='left')
        upcoming_matches['rating_diff'] = upcoming_matches['protag_rating'] - upcoming_matches['antag_rating']
        upcoming_matches['prob'] = upcoming_matches['rating_diff'].apply(lambda x: 1/(1+10**(-x/400)))
        upcoming_matches['pred'] = upcoming_matches['prob'].apply(lambda x: 1 if x > 0.5 else 0)

        return upcoming_matches
    
    
