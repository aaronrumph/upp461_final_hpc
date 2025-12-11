import pandas as pd
from pandas import DataFrame
import os
import numpy as np
import logging
import geopandas as gpd # for taz geojsons

# housekepeing
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
pd.set_option("display.max_columns", None)


def main():
    logging.info("Processing CMAP trip roster data")

    # relevant paths
    path_to_trip_roster = os.path.join("input_data", "cmap_trips_data", "trip_roster.csv")
    # eventually going to output to trips.csv in trips folder in output_data
    output_path = os.path.join("output_data", "trips", "trips.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # read in the trip roster
    trip_roster:DataFrame = pd.read_csv(path_to_trip_roster)

    # debug logging
    logging.debug(f"Columns in trip roster: {trip_roster.columns.tolist()}")
    logging.debug(f"Trip roster data frame head: \n {trip_roster.head()}")

    # simply save the trip roster to output path for now
    trip_roster.to_csv(output_path, index=True)
    logging.info(f"Saved trip roster to {output_path}")

if __name__ == "__main__":
    main()