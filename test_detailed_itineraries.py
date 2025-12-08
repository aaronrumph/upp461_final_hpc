import unittest
from tools_for_places import *
import logging
from io import StringIO

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')

# using explicit main function to allow easier importing without running tests
def main():
    # using oop style test case
    class Tester(unittest.TestCase):


        logging.info("Testing whether running detailed itineraries works")
        # want to check that detailed itineraries work
        def test_whether_detailed_itineraries_work(self):


            self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    unittest.main()
