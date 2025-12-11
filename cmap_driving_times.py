""" This script takes the driving travel time CSV files and converts them to travel time matrices """

import os
import pandas as pd
from pandas import DataFrame
import logging
from pathlib import Path

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    logging.info("Converting CMAP driving travel time csvs to matrices")
    # relevant paths
    this_folder = os.path.dirname(__file__)
    cmap_skims_folder = os.path.join(this_folder, "input_data", "cmap_skims")

    # the mapping dict which says which road network times to use
    road_network_time_mapping = {"0030": "mf46.csv",
                                 "0600": "mf44.csv",
                                 "0700": "mf44.csv",
                                 "0900": "mf44.csv",
                                 "1000": "mf46.csv",
                                 "1430": "mf46.csv",
                                 "1600": "mf44.csv",
                                 "1800": "mf44.csv"}

    # the directory to save the output matrices ALREADY INCLUDE DRIVING IN PATH
    base_output_dir = os.path.join(this_folder, "output_data", "matrices", "driving")

    # go through each time of day and convert the corresponding csv to matrix
    for time_of_day, filename in road_network_time_mapping.items():
        logging.info(f"time of day: {time_of_day}, input filename: {filename}")

        # the path to the input csv
        input_csv_path = os.path.join(cmap_skims_folder, filename)
        logging.info(f"Converting {input_csv_path} to matrix format")

        # read the travel times csv into df
        travel_times:DataFrame = pd.read_csv(input_csv_path)
        logging.debug("Converted travel times csv to DataFrame")

        # need to rename weird p/q/[val] column to "roigin"
        travel_times.rename(columns={"p/q/[val]": "origin"}, inplace=True)

        # now set index to origin column
        travel_times.set_index("origin", inplace=True)

        # log first 100 for debugging
        logging.debug(f" \n {travel_times.iloc[:10, :10]} \n")

        # then number of unique origins and destination pairs should match the shape of the matrix

        num_od_pairs_matrix = travel_times.shape[0] * travel_times.shape[1]

        logging.info(f"Shape of travel time matrix: {travel_times.shape}")

        # save the matrix to csv
        day_constant = "20251209" # using this to be considstent with other file names

        # the name of the output file
        output_filename = f"driving_{day_constant}_{time_of_day}.csv"
        # full path
        output_path = os.path.join(base_output_dir, output_filename)

        # ensure base output directory exists
        os.makedirs(base_output_dir, exist_ok=True)
        logging.info(f"Output path for time {time_of_day}: {output_path}")

        # making sure I don't forget index like an idiot after setting it
        travel_times.to_csv(output_path, index=True)
        logging.info(f"Finished converting travel times to matrix format and saved to CSV at {output_path}")

    logging.info(f"successfully converted {len(os.listdir(base_output_dir))} travel times files")

# safe running environment
if __name__ == "__main__":
    main()