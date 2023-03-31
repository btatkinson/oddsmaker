
import sys
import unittest
import numpy as np  
import pandas as pd

from faker import Faker
from itertools import combinations
# setting path
sys.path.append('../oddsmaker/')
from state_space import Elo


def create_fake_data(num_games=10000, date=True):

    print(f"Creating fake data for {num_games} games...")

    fake = Faker()
    fake_ids = [fake.name() for i in range(10)]

    ## guard against duplicates
    fake_names = []
    [fake_names.append(x) for x in fake_ids if x not in fake_names]  
    fake_ratings = [np.random.normal(1500, 100) for i in range(len(fake_names))]

    matchup_combinations = list(combinations(fake_names, 2))
    ### create 10000 matchups
    matchups = [matchup_combinations[i] for i in np.random.randint(0, len(matchup_combinations), num_games)]
    matchups_df = pd.DataFrame(matchups, columns=['team1', 'team2'])
    matchups_df['date'] = pd.date_range(start='1/1/2018', end='1/1/2022', periods=num_games)
    if date == False:
        matchups_df['rating_period'] = matchups_df.groupby(['date']).ngroup()
        matchups_df = matchups_df.drop(columns=['date'])
    ## make copy of matchups_df with teams reversed
    matchups_df2 = matchups_df.copy()
    matchups_df2['team1'] = matchups_df['team2']
    matchups_df2['team2'] = matchups_df['team1']

    matchups_df['stat'] = 'chess'
    ## add fake ratings
    matchups_df['rating1'] = matchups_df['team1'].map(dict(zip(fake_names, fake_ratings)))
    matchups_df['rating2'] = matchups_df['team2'].map(dict(zip(fake_names, fake_ratings)))
    ## get probability of team1 winning
    matchups_df['prob1'] = 1 / (1 + 10 ** ((matchups_df['rating2'] - matchups_df['rating1']) / 400))
    ## randomly create outcomes
    matchups_df['outcome'] = np.random.binomial(1, matchups_df['prob1'])

    matchups_df2['stat'] = 'chess'
    ## add fake ratings
    matchups_df2['rating1'] = matchups_df2['team1'].map(dict(zip(fake_names, fake_ratings)))
    matchups_df2['rating2'] = matchups_df2['team2'].map(dict(zip(fake_names, fake_ratings)))
    ## get probability of team1 winning
    matchups_df2['prob1'] = 1 / (1 + 10 ** ((matchups_df2['rating2'] - matchups_df2['rating1']) / 400))
    ## outcome is reversed
    matchups_df2['outcome'] = 1 - matchups_df['outcome']
    ## combine matchups_df and matchups_df2
    if date:
        matchups_df = pd.concat([matchups_df, matchups_df2], axis=0).sort_values(by='date').reset_index(drop=True)
    else:
        matchups_df = pd.concat([matchups_df, matchups_df2], axis=0).sort_values(by='rating_period').reset_index(drop=True)

    return matchups_df, fake_names, fake_ratings



class TestEloClass(unittest.TestCase):

    def test_elo_update_with_dates(self):

        fake_data, fake_names, fake_ratings = create_fake_data()

        ## this one runs all at once
        elo1 = Elo(fake_data, protag_id='team1', antag_id='team2', k=3, result_col='outcome')
        history1, error1 = elo1.run_history()
        final_ratings1 = elo1.current_ratings().copy()
        final_ratings1['actual_rating'] = final_ratings1['team1'].map(dict(zip(fake_names, fake_ratings)))
        assert(final_ratings1[['rating','actual_rating']].corr().iloc[0][1] > 0.93)
        assert(np.isclose(final_ratings1['rating'].mean(), 1500))

        fake_data_subset_1 = fake_data.copy()[fake_data['date'] < pd.to_datetime('2021-01-01')]
        fake_data_subset_2 = fake_data.copy()[fake_data['date'] >= pd.to_datetime('2021-01-01')]
        ## this one runs in two parts
        elo2 = Elo(fake_data_subset_1, protag_id='team1', antag_id='team2', k=3, result_col='outcome')
        history2, error2 = elo2.run_history()
        elo2.update(fake_data_subset_2)
        final_ratings2 = elo2.current_ratings().copy()
        final_ratings = pd.merge(final_ratings1, final_ratings2, on='team1', suffixes=('_1', '_2'))

        fake_data_subset_3 = fake_data.sample(frac=0.2)
        fake_data_subset_4 = fake_data[~fake_data.index.isin(fake_data_subset_3.index)]

        ## this one runs in two parts (random split)
        elo3 = Elo(fake_data_subset_4, protag_id='team1', antag_id='team2', k=3, result_col='outcome')
        history3, error3 = elo3.run_history()
        elo3.update(fake_data_subset_3)
        final_ratings3 = elo3.current_ratings().copy()
        final_ratings = pd.merge(final_ratings, final_ratings3, on='team1', suffixes=('_1', '_3'))
        final_ratings = final_ratings.rename(columns={'rating':'rating_3'})
        final_ratings['1_2_diff'] = final_ratings['rating_1'] - final_ratings['rating_2']
        final_ratings['1_3_diff'] = final_ratings['rating_1'] - final_ratings['rating_3']

        assert(np.abs(final_ratings['1_2_diff'].mean()) < 1e-3)
        assert(np.abs(final_ratings['1_3_diff'].mean()) < 1e-3)

    def test_elo_update_with_rps(self):

        fake_data, fake_names, fake_ratings = create_fake_data(date=False)
        ## this one runs all at once
        elo1 = Elo(fake_data, protag_id='team1', antag_id='team2', k=3, result_col='outcome')
        history1, error1 = elo1.run_history()
        final_ratings1 = elo1.current_ratings().copy()
        final_ratings1['actual_rating'] = final_ratings1['team1'].map(dict(zip(fake_names, fake_ratings)))
        assert(final_ratings1[['rating','actual_rating']].corr().iloc[0][1] > 0.93)
        assert(np.isclose(final_ratings1['rating'].mean(), 1500))

        fake_data_subset_1 = fake_data.copy()[fake_data['rating_period'] < 7000]
        fake_data_subset_2 = fake_data.copy()[fake_data['rating_period'] >= 7000]
        ## this one runs in two parts
        elo2 = Elo(fake_data_subset_1, protag_id='team1', antag_id='team2', k=3, result_col='outcome')
        history2, error2 = elo2.run_history()
        elo2.update(fake_data_subset_2)
        final_ratings2 = elo2.current_ratings().copy()
        final_ratings = pd.merge(final_ratings1, final_ratings2, on='team1', suffixes=('_1', '_2'))

        fake_data_subset_3 = fake_data.sample(frac=0.2)
        fake_data_subset_4 = fake_data[~fake_data.index.isin(fake_data_subset_3.index)]

        ## this one runs in two parts (random split)
        elo3 = Elo(fake_data_subset_4, protag_id='team1', antag_id='team2', k=3, result_col='outcome')
        history3, error3 = elo3.run_history()
        elo3.update(fake_data_subset_3)
        final_ratings3 = elo3.current_ratings().copy()
        final_ratings = pd.merge(final_ratings, final_ratings3, on='team1', suffixes=('_1', '_3'))
        final_ratings = final_ratings.rename(columns={'rating':'rating_3'})
        final_ratings['1_2_diff'] = final_ratings['rating_1'] - final_ratings['rating_2']
        final_ratings['1_3_diff'] = final_ratings['rating_1'] - final_ratings['rating_3']
        
        assert(np.abs(final_ratings['1_2_diff'].mean()) < 1e-3)
        assert(np.abs(final_ratings['1_3_diff'].mean()) < 1e-3)



if __name__ == '__main__':
    unittest.main()
