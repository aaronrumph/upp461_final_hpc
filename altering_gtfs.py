""" This module goes through the CTA GTFS data and alters it to create a modified version of the GTFS data"""
import datetime
import os
import zipfile
from pathlib import Path
import pandas as pd
from pandas import DataFrame
import geopandas as gpd
from zipfile import ZipFile, ZIP_DEFLATED
import logging
from geopy import distance

# basic set up
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
pd.set_option("display.max_columns", None)

def read_in_gtfs(gtfs_zip_path) -> dict[str, DataFrame]:
    """Reads GTFS data from a zip file into a dictionary of DataFrames"""
    logging.info(f"Reading GTFS data from zip file: {gtfs_zip_path}")

    # the dict to hold all the gtfs dataframes
    gtfs_data = {}
    
    # creating a temporary folder to extract the gtfs data into
    _temp_agency_name = str(os.path.basename(gtfs_zip_path).replace(".zip", ""))
    _temp_directory = os.path.join(Path(__file__).parent, "_temp_gtfs_extraction", _temp_agency_name)
    
    # make the temp directory if it doesn't already exist
    os.makedirs(_temp_directory, exist_ok=True)
    
    # check if any previous files exist in the temp directory and remove them
    for existing_file in os.listdir(_temp_directory):
        file_path = os.path.join(_temp_directory, existing_file)
        if os.path.isfile(file_path):
            os.remove(file_path)
    
    # extracting the gtfs zip file into the temp directory
    with zipfile.ZipFile(gtfs_zip_path, 'r') as zipped_gtfs:
        zipped_gtfs.extractall(_temp_directory)
        logging.debug(f"Extracted GTFS data to temporary directory: {_temp_directory}")
    
    # reading in each gtfs txt file as a dataframe
    for gtfs_file in os.listdir(_temp_directory):

        gtfs_file_path = os.path.join(_temp_directory, gtfs_file)
        # we only want the txt files not any html or other files
        if gtfs_file.endswith(".txt"):

            # the name of the dataframe will be the gtfs file name without the .txt extension
            df_name = gtfs_file.replace(".txt", "")
            gtfs_data[df_name] = pd.read_csv(gtfs_file_path)
            logging.debug(f"Read GTFS file: {gtfs_file} into DataFrame")

        # then can remove the file after reading it in
        os.remove(gtfs_file_path)
        # debugging info
        logging.debug(f"Removed temporary GTFS file: {gtfs_file_path}")
    # debugging info
    logging.debug(f"Temporary directory {_temp_directory} cleaned up. {len(os.listdir(_temp_directory))} files left")

    logging.info(f"Completed reading GTFS data from zip file: {gtfs_zip_path}; total files read: {len(gtfs_data)}")
    return gtfs_data

def calculate_distance_between_stops(stop_1: tuple[float, float], stop_2: tuple[float, float]) -> float:
    """
    Calculates the distance between two stops given their latitude and longitude using geopy.
    :param stop_1: (tuple[float]) Latitude and Longitude of stop 1 (lat, lon).
    :param stop_2: (tuple[float]) Latitude and Longitude of stop 2 (lat, lon).
    :return distance_ft: (float) Distance between the two stops in feet.
    """
    logging.debug(f"Calculating distance between {stop_1}, {stop_2}")

    # using geopy to calculate distance
    point_1 = distance.Point(stop_1[0], stop_1[1])
    point_2 = distance.Point(stop_2[0], stop_2[1])

    # geodesic distance in feet
    dist_in_feet = distance.geodesic(point_1, point_2).feet

    return dist_in_feet

def calculate_time_from_distance(distance_ft: float, avg_acceleration: float, avg_deceleration: float,
                                 top_speed_mph: float) -> datetime.timedelta:
    logging.debug(f"Calculating time to cover distance {distance_ft} ft with avg acceleration "
                 f"{avg_acceleration} ft/(s^2), and avg deceleration {avg_deceleration} ft/(s^2) ")

    # first convert top speed to ft/s
    top_speed_ft_per_s = (top_speed_mph * 5280) / 3600

    # calculate time to reach top speed
    time_to_top_speed_seconds = top_speed_ft_per_s / avg_acceleration
    # then distance covered during acceleration to top speed (integral)
    distance_to_top_speed_ft = 0.5 * avg_acceleration * (time_to_top_speed_seconds ** 2)

    # same for deceleration
    time_to_stop_seconds = top_speed_ft_per_s / avg_deceleration
    distance_to_stop_ft = 0.5 * avg_deceleration * (time_to_stop_seconds ** 2)

    # check if distance is enough to reach top speed
    if distance_ft >= (distance_to_top_speed_ft + distance_to_stop_ft):
        logging.debug(f"Distance {distance_ft} is sufficient to reach top speed")

        # distance at top speed
        distance_at_top_speed_ft = distance_ft - (distance_to_top_speed_ft + distance_to_stop_ft)
        # time spent at top speed
        time_at_top_speed_seconds = distance_at_top_speed_ft / top_speed_ft_per_s

        # then the total time is the sum of how long to accelerate, time at top speed, and time to decelerate
        total_time_seconds = time_to_top_speed_seconds + time_at_top_speed_seconds + time_to_stop_seconds

    # in case distance is too short to reach top speed
    else:
        """
        Need to calculate the achievable speed within the given distance. Using the equations of motion:
        distance = (0.5) * acceleration * (time_a ** 2) + (0.5) * deceleration * (time_b ** 2) where time_a is time 
        to accelerate to achievable speed (max_velocity_reachable/acceleration) and time_b is
        time to decelerate from achievable speed (max_velocity_reachable/deceleration.
        """
        logging.debug(f"Distance {distance_ft} is insufficient to reach top speed")

        # max velocity reachable within the given distance (weird equation ik)
        max_velocity_reachable = (2 * distance_ft / (1/avg_acceleration + 1/avg_deceleration)) ** 0.5
        # time to accelerate to that speed
        time_to_max_velocity_seconds = max_velocity_reachable / avg_acceleration
        # time to decelerate from that speed
        time_to_stop_from_max_velocity_seconds = max_velocity_reachable / avg_deceleration
        # total time is sum of both times
        total_time_seconds = time_to_max_velocity_seconds + time_to_stop_from_max_velocity_seconds

    logging.debug(f"Total time to cover distance {distance_ft} ft is {total_time_seconds} seconds")
    logging.debug("Do not forget, returned time is a timedelta object")

    return datetime.timedelta(seconds=total_time_seconds)


def calculate_time_gained_by_skipping_stop(gtfs_data: dict[str, DataFrame], stop_name_to_remove: str,
                                           avg_acceleration: float, avg_deceleration: float,
                                           top_speed_mph: float, dwell_time_seconds:float=30) -> float:
    """
    Calculates the time gained by skipping the specified stop based on average acceleration/deceleration and top speed.

    :param gtfs_data: (dict[str, DataFrame]) Dictionary of GTFS DataFrames.
    :param stop_name_to_remove: (str) The name of the stop to be removed.
    :param avg_acceleration: (float) Average acceleration in ft/s² for service at stop for calculating time adjustments.
    :param avg_deceleration: (float) Average deceleration in ft/s² for service at stop for calculating time adjustments.
    :param top_speed_mph: (float) Top speed in miles per hour for calculating time adjustments.
    :param dwell_time_seconds: (float) Average dwell time at the stop in seconds.

    :return time_gained: (float) Estimated time gained in seconds by skipping the stop.
    """
    logging.debug(f"Calculating time gained by skipping stop {stop_name_to_remove}")

    # to return
    time_gained_seconds: float = 0
    time_gained_for_each_trip: list[float] = []

    # look up the stop in the stop_times dataframe to find all trips that include this stop
    _stop_times_df = gtfs_data["stop_times"]
    _stops_df = gtfs_data["stops"]

    # Get all stop_ids for this stop name (usually 2, one for each direction)
    _this_stop_ids = _stops_df[_stops_df["stop_name"] == stop_name_to_remove]["stop_id"].values

    if len(_this_stop_ids) == 0:
        logging.error(f"Stop '{stop_name_to_remove}' not found in stops data")
        return 0.0

    # get all the trip ids that include this stop
    _trips_with_this_stop = _stop_times_df[_stop_times_df["stop_id"].isin(_this_stop_ids)]["trip_id"].unique()

    # next go through each trip and find neighboring stops
    for trip_id in _trips_with_this_stop:
        # get the stop times for this trip
        _stop_times_for_this_trip = _stop_times_df[_stop_times_df["trip_id"] == trip_id].set_index("stop_sequence")

        # find the stop time entry for this stop
        _this_stop_stop_time = _stop_times_for_this_trip[_stop_times_for_this_trip["stop_id"].isin(_this_stop_ids)]

        if len(_this_stop_stop_time) == 0:
            continue

        # get the stop sequence of this stop
        _this_stop_sequence = _this_stop_stop_time.index.values[0]

        # Get the actual stop_id used in THIS trip
        _this_stop_id_for_trip = _this_stop_stop_time.iloc[0]["stop_id"]

        # Get lat/lon for THIS specific stop_id
        _this_stop_lat = _stops_df[_stops_df["stop_id"] == _this_stop_id_for_trip]["stop_lat"].values[0]
        _this_stop_lon = _stops_df[_stops_df["stop_id"] == _this_stop_id_for_trip]["stop_lon"].values[0]

        logging.debug(f"Processing trip_id '{trip_id}' with stop_sequence '{_this_stop_sequence}' "
                      f"for stop '{stop_name_to_remove}'")

        _has_prev_stop = False
        _has_next_stop = False

        # get the previous stop (if exists)
        if (_this_stop_sequence - 1) in _stop_times_for_this_trip.index:
            _prev_stop_id = _stop_times_for_this_trip.loc[_this_stop_sequence - 1, "stop_id"]
            _prev_stop_lat = _stops_df[_stops_df["stop_id"] == _prev_stop_id]["stop_lat"].values[0]
            _prev_stop_lon = _stops_df[_stops_df["stop_id"] == _prev_stop_id]["stop_lon"].values[0]

            # get the name of the previous stop for logging
            _prev_stop_name = _stops_df[_stops_df["stop_id"] == _prev_stop_id]["stop_name"].values[0]

            logging.debug(f"Previous stop for stop '{stop_name_to_remove}' for trip id {trip_id} is {_prev_stop_name}")
            _has_prev_stop = True

        # in case no previous stop
        else:
            logging.debug(f"No previous stop for stop '{stop_name_to_remove}'. "
                          f"Skipping previous stop distance calculation")
            _has_prev_stop = False

        # get the next stop (if exists)
        if (_this_stop_sequence + 1) in _stop_times_for_this_trip.index:
            _next_stop_id = _stop_times_for_this_trip.loc[_this_stop_sequence + 1, "stop_id"]
            _next_stop_lat = _stops_df[_stops_df["stop_id"] == _next_stop_id]["stop_lat"].values[0]
            _next_stop_lon = _stops_df[_stops_df["stop_id"] == _next_stop_id]["stop_lon"].values[0]

            # get the name of the next stop for logging
            _next_stop_name = _stops_df[_stops_df["stop_id"] == _next_stop_id]["stop_name"].values[0]
            logging.debug(f"Next stop for stop '{stop_name_to_remove}' for trip id {trip_id} is {_next_stop_name}")
            _has_next_stop = True

        # calculate distance from previous stop to this stop
        else:
            logging.debug(f"No next stop for stop '{stop_name_to_remove}'. "
                          f"Skipping next stop distance calculation")
            _has_next_stop = False

        # raise error if neither previous nor next stop exists (should not happen in valid data)
        if not _has_prev_stop or not _has_next_stop:
            logging.debug(
                f"Stop '{stop_name_to_remove}' in trip id {trip_id} has neither previous nor next stop. Skipping")
            continue  # Changed from raise to continue

        # using try-except block to catch any unexpected errors during distance/time calculations
        try:

            # now if both previous and next stops exist, can calculate distances
            if _has_prev_stop and _has_next_stop:
                # calculate distances for existing network
                distance_prev_to_this_ft = calculate_distance_between_stops(stop_1=(_prev_stop_lat, _prev_stop_lon),
                                                                            stop_2=(_this_stop_lat, _this_stop_lon))
                distance_this_to_next_ft = calculate_distance_between_stops(stop_1=(_this_stop_lat, _this_stop_lon),
                                                                            stop_2=(_next_stop_lat, _next_stop_lon))

                # calculate distance between previous and next stop (skipping this stop)
                distance_prev_to_next_ft = calculate_distance_between_stops(stop_1=(_prev_stop_lat, _prev_stop_lon),
                                                                            stop_2=(_next_stop_lat, _next_stop_lon))

                # calculate time for existing distances
                seconds_prev_to_this = calculate_time_from_distance(distance_ft=distance_prev_to_this_ft,
                                                                    avg_acceleration=avg_acceleration,
                                                                    avg_deceleration=avg_deceleration,
                                                                    top_speed_mph=top_speed_mph)
                seconds_this_to_next = calculate_time_from_distance(distance_ft=distance_this_to_next_ft,
                                                                    avg_acceleration=avg_acceleration,
                                                                    avg_deceleration=avg_deceleration,
                                                                    top_speed_mph=top_speed_mph)

                # total time for existing stop arrangement (calculated)
                total_time_existing_seconds = (seconds_prev_to_this.total_seconds() + dwell_time_seconds +
                                                  seconds_this_to_next.total_seconds())

                # calculate time for skipping this stop
                time_prev_to_next = calculate_time_from_distance(distance_ft=distance_prev_to_next_ft,
                                                                 avg_acceleration=avg_acceleration,
                                                                 avg_deceleration=avg_deceleration,
                                                                 top_speed_mph=top_speed_mph)

                # checking the actual departure and arrival times for debugging and error correction
                actual_departure_time_prev = _stop_times_for_this_trip[_stop_times_for_this_trip["stop_id"] ==
                                                                       _prev_stop_id]["departure_time"].values[0]
                actual_arrival_time_next = _stop_times_for_this_trip[_stop_times_for_this_trip["stop_id"] ==
                                                                     _next_stop_id]["arrival_time"].values[0]
                # based on the GTFS data, what is the actual time difference
                actual_seconds_prev_to_next = datetime.datetime.strptime(actual_arrival_time_next, "%H:%M:%S") - \
                                              datetime.datetime.strptime(actual_departure_time_prev, "%H:%M:%S")

                # need a scaling factor to adjust for real-world conditions (adjust based on actual GTFS times)
                scaling_factor = total_time_existing_seconds / actual_seconds_prev_to_next.total_seconds()

                # time gained by skipping stop is existing time minus new time, times the scaling factor
                time_gained_for_this_trip = (total_time_existing_seconds -
                                             time_prev_to_next.total_seconds()) * scaling_factor

                time_gained_for_each_trip.append(time_gained_for_this_trip)
                logging.debug(f"Time gained for trip id {trip_id} by skipping stop '{stop_name_to_remove}' is "
                              f"{time_gained_for_this_trip} seconds")

                # debugging info  (IMPORTANT INFO)
                logging.debug("")
                logging.debug(f"Actual time between stops '{_prev_stop_name}' and '{_next_stop_name}' for trip id "
                              f"{trip_id} is {actual_seconds_prev_to_next.total_seconds()} seconds")
                logging.debug(f"Calculated time between stops'{_prev_stop_name}' and '{stop_name_to_remove}' in"
                              f"existing network is {total_time_existing_seconds} seconds")
                logging.debug(f"Calculated time between stops '{_prev_stop_name}' and '{_next_stop_name}' by skipping "
                              f"'{stop_name_to_remove}' is {time_prev_to_next.total_seconds()} seconds")
                logging.debug("")

        # error handling
        except Exception as e:
            logging.error(f"Error calculating time gained for trip id {trip_id} by skipping stop "
                          f"'{stop_name_to_remove}': {e}")
            continue

    # average time gained across all trips
    if len(time_gained_for_each_trip) > 0:
        time_gained_seconds = sum(time_gained_for_each_trip) / len(time_gained_for_each_trip)
    else:
        time_gained_seconds = 0
        logging.warning(f"No valid trips found for stop '{stop_name_to_remove}' to calculate time gained")

    logging.info(f"Average time gained by skipping stop '{stop_name_to_remove}' is {time_gained_seconds} seconds")
    return time_gained_seconds

# the actual function to remove a stop from the gtfs data
def remove_stop_from_gtfs(gtfs_data: dict[str, DataFrame], stop_name_to_remove: str, avg_acceleration: float=2.79,
                          avg_deceleration: float=2.79, top_speed_mph: float=55,
                          dwell_time_seconds:float=30) -> dict[str, DataFrame]:
    """
    Removes a stop from the GTFS data, and then recalculates stop sequences and stop times and updates related files.
    :param gtfs_data: (dict[str, DataFrame]) Dictionary of GTFS DataFrames.
    :param stop_name_to_remove: (str) The name (must match GTFS data) of the stop to be removed.
    :param avg_acceleration: (float) Average acceleration in ft/s² for service at stop for calculating time adjustments.
    :param avg_deceleration: (float) Average deceleration in ft/s² for service at stop for calculating time adjustments.
    :param top_speed_mph: (float) Top speed in miles per hour for calculating time adjustments.
    :param dwell_time_seconds: (float) Average dwell time at the stop in seconds.

    :return modified_gtfs_data: (dict[str, DataFrame]) Dictionary of modified GTFS DataFrames.
    """
    logging.info(f"Removing stop '{stop_name_to_remove}' from GTFS data")

    # can just pass args to time saved calculation function to get time adjustment
    secs_saved_skipping_stop = calculate_time_gained_by_skipping_stop(gtfs_data=gtfs_data,
                                                                      stop_name_to_remove=stop_name_to_remove,
                                                                      avg_acceleration=avg_acceleration,
                                                                      avg_deceleration=avg_deceleration,
                                                                      top_speed_mph=top_speed_mph,
                                                                      dwell_time_seconds=dwell_time_seconds)
    
    # make a copy of the gtfs data to modify
    _copy_current_gtfs_data = gtfs_data.copy()

    # get the stops dataframe
    _stops_df = _copy_current_gtfs_data["stops"]

    # make copy with names as index for easier searching
    _stops_df_indexed_by_name = _stops_df.set_index("stop_name")

    # check if the stop exists
    if stop_name_to_remove not in _stops_df_indexed_by_name.index:
        logging.error(f"Stop '{stop_name_to_remove}' not found in GTFS data")
        raise ValueError(f"Stop '{stop_name_to_remove}' not found in GTFS data")

    # get the stop_id of the stop to remove (default is two, one for eac direction)
    _this_stop_id = _stops_df_indexed_by_name.loc[stop_name_to_remove, "stop_id"]

    # get lat and lon to calculate stop distances later
    _this_stop_lat = _stops_df_indexed_by_name.loc[stop_name_to_remove, "stop_lat"]
    _this_stop_lon = _stops_df_indexed_by_name.loc[stop_name_to_remove, "stop_lon"]

    # now need to remove the stop from the actual (not temp indexed) stops dataframe
    _copy_current_gtfs_data["stops"] = _stops_df[~_stops_df["stop_id"].isin(_this_stop_id)].reset_index(drop=True)
    logging.info(f"Removed stop '{stop_name_to_remove}' with stop_id '{_this_stop_id}' from stops.txt")

    # now need to remove the stop from stop_times.txt and recalculate stop sequences and times
    _stop_times_df = _copy_current_gtfs_data["stop_times"]

    # get all stop times entries for the stop to be removed
    _stop_times_with_this_stop = _stop_times_df[_stop_times_df["stop_id"].isin(_this_stop_id)]
    # now get all stop times entries that do NOT have this stop (to keep)
    _stop_times_to_keep = _stop_times_df[~_stop_times_df["stop_id"].isin(_this_stop_id)]

    """
    Now the important part is recalculating the stop sequences and stop times for the trips that had this stop removed.
    this requires knowing the distances between stops to estimate time adjustments based on average
    acceleration/deceleration rates.
    """
    logging.info("Recalculating stop sequences and stop times for affected trips")
    num_of_trips = len(_stop_times_with_this_stop["trip_id"].unique())

    # going through each trip that had the specified stop
    for _trip_idx, trip_id in enumerate(_stop_times_with_this_stop["trip_id"].unique()):
        # get the stop sequence of the removed stop
        _removed_stop_sequence = _stop_times_with_this_stop[
            _stop_times_with_this_stop["trip_id"] == trip_id
            ]["stop_sequence"].values[0]

        # Find all stops in this trip that come AFTER the removed stop
        mask = (_stop_times_to_keep["trip_id"] == trip_id) & \
               (_stop_times_to_keep["stop_sequence"] > _removed_stop_sequence)

        # Vectorized update: decrement stop_sequence for all matching rows
        _stop_times_to_keep.loc[mask, "stop_sequence"] -= 1

        # Vectorized update: adjust times for all matching rows
        for idx in _stop_times_to_keep[mask].index:
            row = _stop_times_to_keep.loc[idx]

            # Parse GTFS times
            def parse_gtfs_time(time_str: str) -> tuple[int, datetime.datetime]:
                parts = time_str.split(":")
                hours = int(parts[0])
                minutes = int(parts[1])
                seconds = int(parts[2])
                days_offset = hours // 24
                hours_normalized = hours % 24
                dt = datetime.datetime.strptime(f"{hours_normalized:02d}:{minutes:02d}:{seconds:02d}", "%H:%M:%S")
                return days_offset, dt

            def format_gtfs_time(days_offset: int, dt: datetime.datetime) -> str:
                hours = dt.hour + (days_offset * 24)
                return f"{hours:02d}:{dt.minute:02d}:{dt.second:02d}"

            # Parse and adjust arrival time
            arrival_days, arrival_dt = parse_gtfs_time(row["arrival_time"])
            new_arrival = arrival_dt - datetime.timedelta(seconds=secs_saved_skipping_stop)
            if new_arrival.day < arrival_dt.day:
                arrival_days -= 1
            _stop_times_to_keep.at[idx, "arrival_time"] = format_gtfs_time(arrival_days, new_arrival)

            # Parse and adjust departure time
            departure_days, departure_dt = parse_gtfs_time(row["departure_time"])
            new_departure = departure_dt - datetime.timedelta(seconds=secs_saved_skipping_stop)
            if new_departure.day < departure_dt.day:
                departure_days -= 1
            _stop_times_to_keep.at[idx, "departure_time"] = format_gtfs_time(departure_days, new_departure)

        percent_processed = _trip_idx / num_of_trips * 100
        print(f"Processed trip {_trip_idx + 1} of {num_of_trips} ({percent_processed:.2f}%)", end="\r", flush=True)


    # after going through all trips and stop times, update the stop_times dataframe in the gtfs data
    _stop_times_df = _stop_times_to_keep.reset_index(drop=True)
    logging.info(f"Updated stop_times.txt after removing stop '{stop_name_to_remove}'")

    # stop removed from stops and stop_times dataframes
    logging.info(f"Successfully removed stop '{stop_name_to_remove}' from GTFS data")

    # add the modified files back to a gtfs data dict to return
    modified_gtfs_data = {"stops": _copy_current_gtfs_data["stops"], "stop_times": _stop_times_df}

    # then add all the other unmodified dataframes
    for df_name, df in _copy_current_gtfs_data.items():
        # check that not already added
        if df_name not in modified_gtfs_data.keys():
            modified_gtfs_data[df_name] = df

    return modified_gtfs_data

# function to write modified gtfs data back to zip file
def write_gtfs_to_zip(gtfs_data: dict[str, DataFrame], output_zip_path: str, ask_me_to_overwrite: bool = True):
    """Writes GTFS data from a dictionary of DataFrames back to a zip file"""
    logging.info(f"Writing GTFS data to zip file: {output_zip_path}")

    # creating a temporary folder to hold the gtfs txt files
    _temp_agency_name = str(os.path.basename(output_zip_path).replace(".zip", ""))
    _temp_directory = os.path.join(Path(__file__).parent, "_temp_gtfs_writing", _temp_agency_name)

    # make the temp directory if it doesn't already exist
    os.makedirs(_temp_directory, exist_ok=True)

    # check if any previous files exist in the temp directory and remove them
    for existing_file in os.listdir(_temp_directory):
        logging.debug("Checking for existing files in temporary directory")
        file_path = os.path.join(_temp_directory, existing_file)
        if os.path.isfile(file_path):
            os.remove(file_path)
            logging.debug(f"Removed existing GTFS file {file_path} to make room for new files")

    # check that the temp directory is empty
    if len(os.listdir(_temp_directory)) > 0:
        logging.error(f"Temporary directory {_temp_directory} is not empty after cleanup")
        raise Exception(f"Temporary directory {_temp_directory} is not empty after cleanup")

    # writing each dataframe to a txt file in the temp directory
    for df_name, df in gtfs_data.items():
        # making sure write to a txt file NOT a csv file
        gtfs_file_path = os.path.join(_temp_directory, f"{df_name}.txt")
        df.to_csv(gtfs_file_path, index=False)
        logging.debug(f"Wrote DataFrame to GTFS file: {gtfs_file_path}")

    # check if the output zip file already exists and remove it
    if os.path.exists(output_zip_path):
        # adding this so that if not sure about overwriting, will double check before getting rid of original
        if ask_me_to_overwrite:
            user_input = input(f"The file {output_zip_path} already exists. Do you want to overwrite it? (y/n): ")

            if user_input.lower() != "y" or user_input.lower() != "yes":
                logging.info("User chose not to overwrite the existing GTFS zip file. Exiting write operation")
                raise FileExistsError(f"The file {output_zip_path} already exists and was not overwritten")

        os.remove(output_zip_path)
        logging.debug(f"Removed existing GTFS zip file: {output_zip_path}")

    # creating the zip file from the txt files
    with ZipFile(output_zip_path, "w", ZIP_DEFLATED) as zipped_gtfs:
        # use each file in the temp directory
        for gtfs_file in os.listdir(_temp_directory):
            # the path ends up looking like temp_directory/gtfs_file.txt
            gtfs_file_path = os.path.join(_temp_directory, gtfs_file)
            # write it to the zip file
            zipped_gtfs.write(gtfs_file_path, arcname=gtfs_file)
            logging.debug(f"Added GTFS file to zip: {gtfs_file_path}")
            # then can remove the file in the temp folder after adding it to the zip
            os.remove(gtfs_file_path)
            logging.debug(f"Removed temporary GTFS file: {gtfs_file_path}")

    # removing the temporary directory
    os.rmdir(_temp_directory)
    logging.info(f"Completed writing GTFS data to zip file: {output_zip_path}")




if __name__ == "__main__":
    gtfs_zip_path_cta_test = "temp_gtfs/no_monroe_cta.zip"
    gtfs_data_cta_test = read_in_gtfs(gtfs_zip_path_cta_test)
    remove_monroe_test = remove_stop_from_gtfs(gtfs_data_cta_test, "Monroe-Blue")
    output_path = "input_data/gtfs_data/no_monroe_gtfs/modified_no_monroe_cta.zip"
    write_gtfs_to_zip(remove_monroe_test, output_path, ask_me_to_overwrite=False)


""" 
::::::::::::DEBUGGING NOTES:::::::::::
- read_in_gtfs function works as intended, reads in GTFS data from zip file into dictionary of DataFrames.
- added function to write modified GTFS data back to zip file.

Now need to add the functions that actually modify the GTFS data as needed:
1. remove stop from stops.txt and all references to that stop in other files
2. recalculate stop_times.txt based on new stop sequence
3. check if other files need to be updated based on stop removal (e.g., trips.txt, routes.txt)

"""