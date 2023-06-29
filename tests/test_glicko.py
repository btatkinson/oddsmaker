import sys
import unittest
import numpy as np  
import pandas as pd

from faker import Faker
from itertools import combinations
# setting path
sys.path.append('../oddsmaker/')
from utils import make_symmetrical
from state_space import Glicko




class TestGlickoClass(unittest.TestCase):

    def test_paper_example(self):

        """Test the Glicko class on the example from the paper."""

        data = pd.DataFrame({
            'rating_period':[1,1,1],
            'protag_id':[1,1,1],
            'antag_id':[2,3,4],
            'stat':np.repeat('strokes_gained',3),
            'result':[1,0,0]
        })

        data = make_symmetrical(data)

        priors = {
            1:{'strokes_gained':{'rating':1500, 'RD':200}},
            2:{'strokes_gained':{'rating':1400, 'RD':30}},
            3:{'strokes_gained':{'rating':1550, 'RD':100}},
            4:{'strokes_gained':{'rating':1700, 'RD':300}}
        }

        glicko = Glicko(data, priors=priors)
        pregame_ratings = glicko.create_pregame_ratings()
        assert(int(np.round(pregame_ratings.iloc[0]['rating'])) == 1464)
        assert(int(np.round(pregame_ratings.iloc[0]['RD'])) == 151)


        return
    
    def test_multiple_stats(self):

        data = pd.DataFrame({
            'rating_period':[1,1,1,1,1,1],
            'protag_id':[1,1,1,1,1,1],
            'antag_id':[2,3,4,2,3,4],
            'stat':np.repeat(['strokes_gained_OTT','strokes_gained_APP'],3),
            'result':[1,0,0,1,0,0]
        })

        data = make_symmetrical(data)

        priors = {
            1:{'strokes_gained_OTT':{'rating':1500, 'RD':200},
               'strokes_gained_APP':{'rating':1500, 'RD':200}},
            2:{'strokes_gained_OTT':{'rating':1400, 'RD':30},
               'strokes_gained_APP':{'rating':1400, 'RD':30}},
            3:{'strokes_gained_OTT':{'rating':1550, 'RD':100},
               'strokes_gained_APP':{'rating':1550, 'RD':100}},
            4:{'strokes_gained_OTT':{'rating':1750, 'RD':150},
               'strokes_gained_APP':{'rating':1700, 'RD':300}}
        }

        glicko = Glicko(data, priors=priors)
        pregame_ratings = glicko.create_pregame_ratings()
        print(pregame_ratings)
        assert(int(np.round(pregame_ratings.iloc[0]['rating'])) == 1464)
        assert(int(np.round(pregame_ratings.iloc[0]['RD'])) == 151)

        assert(int(np.round(pregame_ratings.iloc[4]['rating'])) == 1468)
        assert(int(np.round(pregame_ratings.iloc[4]['RD'])) == 150)


        return
    

if __name__ == '__main__':
    unittest.main()

