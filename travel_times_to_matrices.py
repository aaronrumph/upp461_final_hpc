import logging
import os
import pandas as pd
from pandas import DataFrame
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


def convert_travel_times_to_matrices(input_travel_time_csv_path: str, desired_output_dir: str) -> DataFrame:
    """Convert travel times from a CSV file to matrix format (idx = origin and col = dest) and save as CSV files."""
    logging.info(f"Starting conversion of travel times from {input_travel_time_csv_path} to matrices,"
                 f"writing to {desired_output_dir}")

    # read the travel times csv
    travel_times:DataFrame = pd.read_csv(input_travel_time_csv_path)
    logging.debug("Converted travel times CSV to DataFrame")

    # files follow format "scneario"_"day [constant]"_time_of_day.csv
    base_filename = os.path.basename(input_travel_time_csv_path)
    filename_parts = base_filename.replace(".csv", "").split("_")

    # scenario name is either no_monroe, driving, or current_network
    scenario_name = filename_parts[:-2]
    scenario_name = "_".join(scenario_name)
    day_constant = filename_parts[-2]
    time_of_day = filename_parts[-1]

    # pivot the dataframe to create the matrix (origins is "from_id", destinations is "to_id")
    travel_time_matrix:DataFrame = travel_times.copy()
    travel_time_matrix = travel_time_matrix.pivot(columns="to_id", index="from_id", values="travel_time")


    # log the two matrices for comparison
    logging.debug(f"{travel_time_matrix.iloc[:10, :10]}")
    logging.debug(f"{travel_times.iloc[:10, :10]}")

    # then number of unique origins and destination pairs should match the shape of the matrix
    num_od_pairs_original = len(travel_times)
    num_od_pairs_matrix = travel_time_matrix.shape[0] * travel_time_matrix.shape[1]
    logging.info(f"Number of unique od pairs in original data vs number of od pairs in matrix format: "
                 f"{num_od_pairs_original} vs {num_od_pairs_matrix}")
    logging.info(f"Shape of travel time matrix: {travel_time_matrix.shape}")
    if num_od_pairs_original < num_od_pairs_matrix:
        logging.warning("The number of OD pairs in the matrix format exceeds the original data")

    # save the matrix to csv
    output_filename = f"{scenario_name}_{day_constant}_{time_of_day}_matrix.csv"
    output_path = os.path.join(desired_output_dir, output_filename)
    travel_time_matrix.to_csv(output_path)

    logging.info("Finished converting travel times to matrix format and saved to CSV")
    return travel_time_matrix


# now the actual script to go through all travel time csvs and convert them to matrices
if __name__ == "__main__":
    r""" 
    now need to go through all travel time csvs in the output_data\travel_times\current_transit and 
    ...\no_monroe directories and convert them to matrix format"""

    # define the directories
    output_data_directory_subfolders = ["current_transit", "no_monroe"]
    base_travel_times_directory = os.path.join(os.path.dirname(__file__), "output_data", "travel_times")

    # base output directory is output_data/matrices
    base_output_data_directory = os.path.join(os.path.dirname(__file__), "output_data", "matrices")

    # if any of the output subdirectories don't exist, create them
    os.makedirs(base_output_data_directory, exist_ok=True)

    # the output directory will be \output_data\matrices\{subfolder}
    for subfolder in output_data_directory_subfolders:
        logging.info(f"Processing subfolder: {subfolder}")

        # define the input and output directories for this specific scenario (driving, current_transit, no_monroe)
        travel_times_directory = os.path.join(base_travel_times_directory, subfolder)
        desired_output_dir = os.path.join(base_output_data_directory, subfolder)

        # make the output directory if it doesn't exist!
        os.makedirs(desired_output_dir, exist_ok=True)

        # go through all csv files in the travel_times_directory
        for filename in os.listdir(travel_times_directory):
            if filename.endswith(".csv"):
                logging.info(f"Processing file: {filename}")

                # define the full input path
                input_travel_time_csv_path = os.path.join(travel_times_directory, filename)
                convert_travel_times_to_matrices(input_travel_time_csv_path, desired_output_dir)

                # log factory!
                logging.info(f"Converted {filename} to matrix format and saved to {desired_output_dir}")
        logging.info(f"Finished processing all files in {subfolder}")
    logging.info("Completed conversion of all travel time csvs to matrix format")







