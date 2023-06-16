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

        # Create the Glicko object
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
    

if __name__ == '__main__':
    unittest.main()

