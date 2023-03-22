import unittest
from faker import Faker
from oddsmaker.state_space import Elo


class TestEloClass(unittest.TestCase):

    def test_elo_dates(self):
        
        ### create fake data first set of dates 

        ### create fake data second set of dates (does not overlap)

        ### create fake data third set of dates (overlaps)
        elo = Elo()

        self.assertEqual(1, 1)

    def test_elo_rating_periods(self):
        
        ### create fake data rating period 1 

        ### create fake data rating period 2 (does not overlap)

        ### create fake data rating period 3 (overlaps)
        elo = Elo()

        self.assertEqual(1, 1)


if __name__ == '__main__':
    unittest.main()
