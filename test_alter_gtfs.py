""" This is a unit test for the altering_gtfs module. """

import unittest
import altering_gtfs
from altering_gtfs import *
import logging

# set logger cuz don't need to see all that debugging stuff, that's what tests are for!
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger("altering_gtfs").setLevel(logging.INFO)

# test parameters
test_station = "Monroe-Blue"
test_station_ids = ["30154", "30153"]
test_input_gtfs = "temp_gtfs/no_monroe_cta.zip"

class TestAlteringGTFS(unittest.TestCase):
    logging.info("Starting unit tests for altering_gtfs module")

    
    
    def test_alter_gtfs_for_simple_errors(self):
        logging.info("Testing to see if any exceptions are raised for simple errors")
        test_dfs = read_in_gtfs(test_input_gtfs)
        removing_monroe_test = remove_stop_from_gtfs(test_dfs, test_station)
        print(f"Resulting dict: removing_monroe_test")
        logging.info("No exceptions raised for simple errors")
        
    def test_if_stop_removed(self):
        logging.info("Testing to see if the stop is actually removed from the GTFS data")
        test_dfs = read_in_gtfs(test_input_gtfs)
        modified_gtfs = remove_stop_from_gtfs(test_dfs, test_station)
        self.assertNotIn(test_station, modified_gtfs['stops']['stop_name'].values)
        logging.debug("Not in stops dataframe after removal")
        for stop_id in test_station_ids:
            self.assertNotIn(stop_id, modified_gtfs['stop_times']['stop_id'].values)
            logging.debug(f"Stop ID {stop_id} not in stop_times dataframe after removal")
        
        logging.info("Stop successfully removed from GTFS data")

if __name__ == '__main__':
    unittest.main()
        
        
