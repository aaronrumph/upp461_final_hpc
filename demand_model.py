"""
Now that have all the necessary inputs in their right places, can run the demand model.
This is quite a hefty module, so going to try to write good, readable code
"""

# essential modules of course
import os
import pandas as pd
from pandas import DataFrame
import logging
from pathlib import Path
import numpy as np
import geopandas as gpd
from functools import wraps

# housekeeping
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
pd.set_option("display.max_columns", None)

# timer function
def time_this_function(func):
    """Decorator to time functions"""
    import time

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        logging.info(f"Function {func.__name__} took {end_time - start_time:.2f} seconds")
        return result
    return wrapper

""" This section contains global variables and reference dictionaries (rosetta means translation dict)"""

# important paths
this_dir_path = os.path.dirname(__file__)
input_data_path = os.path.join(this_dir_path, "input_data")
output_data_path = os.path.join(this_dir_path, "output_data")

# dictionary for modes because impossible to remember all mode codes
trip_mode_to_code_rosetta = {1: "SOV",
                             2: "HOV2",
                             3: "HOV3+",
                             4: "Taxi",
                             5: "TNC Regular",
                             6: "TNC Shared",
                             7: "Transit",
                             8: "Walk",
                             9: "Bicycle"}

# same for time periods
time_period_to_time_rosetta = {"EA": "0030",
                               "AM1": "0600",
                               "AM2": "0700",
                               "AM3": "0900",
                               "MD": "1000",
                               "PM1": "1430",
                               "PM2": "1600",
                               "PM3": "1800"}


# and reverse mapping dict for time periods as well
time_to_time_period_rosetta = {analyzed_time: time_period_name for time_period_name, analyzed_time in
                               time_period_to_time_rosetta.items()}

# finally, just list of the time periods to analyze
all_times_to_analyze = tuple(time_period_to_time_rosetta.values())

# next the mode choice coefficients based on the CMAP 2040 documentation (Table 5.1; CMAP, 2010)
mode_choice_coefficients = {"HBWH_CBD": {
                                             "description": "home-based work high-income to cbd",
                                             "beta_time": 0.0159,
                                             "mode_constant": -1.0000},
                            "HBWH_NON_CBD": {
                                            "description": "home-based work high-income to non-cbd",
                                            "beta_time": 0.0186,
                                            "mode_constant": -2.0000},
                            "HBWL_CBD": {
                                            "description": "home-based work low-income to cbd",
                                            "beta_time": 0.0159,
                                            "mode_constant": -1.0000},
                            "HBWL_NON_CBD": {
                                            "description": "home-based work low-income to non-cbd",
                                            "beta_time": 0.0186,
                                            "mode_constant": -2.0000},
                            "HBO": {
                                            "description": "home-based other",
                                            "beta_time": 0.0114,
                                            "mode_constant": -1.9000},
                            "HBS": {
                                            "description": "home-based shopping",
                                            "beta_time": 0.0114,
                                            "mode_constant": -1.9000},
                            "NHB": {
                                            "description": "non-home-based",
                                            "beta_time": 0.0114,
                                            "mode_constant": -1.9000},
                            "VISIT": {
                                            "description": "visitor trips",
                                            "beta_time": 0.0114,
                                            "mode_constant": -1.9000},
                            "DEFAULT": {
                                            "description": "default fallback coefficients",
                                            "beta_time": 0.0114,
                                            "mode_constant": -1.9000}}


# taz path
taz_geojson_path = os.path.join(input_data_path, "taz_data", "tazs_centroids.geojson")
# now the taz gdf (useful as global)
taz_geodataframe: gpd.GeoDataFrame = gpd.read_file(taz_geojson_path)
logging.info(f"Read in taz geodataframe with {len(taz_geodataframe)} tazs")

# want to make the tazs that are in the cbd have their own gdf for easy access
cbd_tazs = taz_geodataframe[taz_geodataframe["cbd"] == 1]
# get the zone ids for the cbd tazs (NUMPY ARRAY NOT LIST)
cbd_taz_ids = cbd_tazs["zone17"].values

logging.debug(f"There are {len(cbd_taz_ids)} tazs in the cbd area")

# final log message for set up
logging.info(f"Successfully set up important prerequisites for demand model")

""" Functions to load necessary input data """


@time_this_function
def load_trip_roster(filter_transit_only: bool = True,
                     filter_valid_timeperiod: bool = True,
                     custom_time_periods: tuple[str] | None = None,
                     assign_missing_timeperiods: bool = True) -> DataFrame:
    """
    Load and filter the trip roster from input data

    :param filter_transit_only: only include transit trips (mode 7)
    :param filter_valid_timeperiod: only include trips with valid time periods
    :param custom_time_periods: custom time period NAMES to filter to
    :param assign_missing_timeperiods: assign time periods to trips with NaN timeperiod

    :return: filtered trip roster DataFrame
    """

    # load the trip roster
    trip_roster_path = os.path.join(output_data_path, "trips", "trips.csv")
    trip_roster_df = pd.read_csv(trip_roster_path)
    logging.info(f"Loaded {len(trip_roster_df):,} total trip records")

    # check for NaN time periods
    nan_timeperiod_count = trip_roster_df["timeperiod"].isna().sum()
    logging.info(f"Trips with NaN timeperiod: {nan_timeperiod_count:,} ({nan_timeperiod_count / len(trip_roster_df) * 100:.1f}%)")

    # filter to only transit trips if requested
    if filter_transit_only:
        original_count = len(trip_roster_df)
        trip_roster_df = trip_roster_df[trip_roster_df["mode"] == 7].copy()
        logging.info(f"Filtered to {len(trip_roster_df):,} transit trips (from {original_count:,})")

        # check NaN timeperiods in transit trips specifically
        transit_nan_count = trip_roster_df["timeperiod"].isna().sum()
        logging.info(f"Transit trips with NaN timeperiod: {transit_nan_count:,} ({transit_nan_count / len(trip_roster_df) * 100:.1f}%)")

    # assign time periods to trips with NaN if requested
    if assign_missing_timeperiods:
        nan_mask = trip_roster_df["timeperiod"].isna()
        num_nan = nan_mask.sum()

        if num_nan > 0:
            # get distribution from trips that have time periods
            has_timeperiod = trip_roster_df["timeperiod"].notna()

            if has_timeperiod.sum() > 0:
                # use distribution from non-NaN trips
                timeperiod_dist = trip_roster_df[has_timeperiod]["timeperiod"].value_counts(normalize=True)
                logging.info(f"Assigning {num_nan:,} NaN timeperiods based on distribution:")
                for period, pct in timeperiod_dist.items():
                    logging.info(f"  {period}: {pct * 100:.1f}%")

                assigned_periods = np.random.choice(
                    timeperiod_dist.index,
                    size=num_nan,
                    p=timeperiod_dist.values
                )
            else:
                # all trips have NaN so distribute evenly
                logging.warning("All trips have NaN timeperiod - distributing evenly across all periods")
                valid_periods = list(time_period_to_time_rosetta.keys())
                assigned_periods = np.random.choice(valid_periods, size=num_nan)

            trip_roster_df.loc[nan_mask, "timeperiod"] = assigned_periods
            logging.info(f"Assigned time periods to {num_nan:,} trips")

    # filter to valid time periods if requested
    if filter_valid_timeperiod:
        # determine which period names to keep
        if custom_time_periods is not None:
            valid_period_names = list(custom_time_periods)
            logging.info(f"Using custom time period names: {valid_period_names}")
        else:
            valid_period_names = list(time_period_to_time_rosetta.keys())
            logging.info(f"Using all {len(valid_period_names)} defined time periods: {valid_period_names}")

        # filter the dataframe
        original_count = len(trip_roster_df)
        trip_roster_df = trip_roster_df[trip_roster_df["timeperiod"].isin(valid_period_names)].copy()
        filtered_count = len(trip_roster_df)
        logging.info(f"Kept {filtered_count:,} trips (removed {original_count - filtered_count:,})")

    # log final statistics
    total_person_trips = trip_roster_df["trips"].sum()
    logging.info(f"Final trip roster: {len(trip_roster_df):,} records, {total_person_trips:,.0f} person trips")

    return trip_roster_df


# function to load travel time matrices for a given scenario
@time_this_function
def load_travel_time_matrices(scenario_name: str) -> dict[str, DataFrame]:
    """
    Loads travel time matrices for a given scenario

    :param scenario_name: scenario name matching directory structure

    :return: dictionary mapping time codes to matrix DataFrames
    """

    # the directory for the scenario REMEMBER IS IN "output_data' PATH
    scenario_dir = os.path.join(output_data_path, "matrices", scenario_name)

    # check whether actually exists
    if not os.path.exists(scenario_dir):
        logging.error(f"Scenario directory not found: {scenario_dir}")
        return {}

    # empty dict to hold matrices
    matrices = {}

    # go through each time code and load the corresponding matrix
    for time_code in all_times_to_analyze:
        # list all files in the scenario directory matching the time code
        matrix_files = [f for f in os.listdir(scenario_dir) if time_code in f and f.endswith(".csv")]

        # if there is a matching file, load it
        if matrix_files:
            matrix_path = os.path.join(scenario_dir, matrix_files[0])
            matrix = pd.read_csv(matrix_path, index_col=0)

            # convert index and columns to int so comparing like with like
            matrix.index = matrix.index.astype(int)
            matrix.columns = matrix.columns.astype(int)

            # add to dict
            matrices[time_code] = matrix
            logging.debug(f"Loaded {scenario_name} matrix for {time_code}")

    logging.info(f"Loaded {len(matrices)} matrices for scenario: {scenario_name}")
    return matrices


""" Now we can move on to the actual demand model"""


def get_mode_choice_coefficients(trip_purpose: str,
                                 destination_zone_id: int) -> tuple[float, float]:
    """
    Get the mode choice coefficients based on trip purpose and destination zone
    Based on CMAP 2040 documentation

    :param trip_purpose: purpose of the trip (as in trip roster)
    :param destination_zone_id: destination zone id of the trip

    :return: tuple of (beta_time_coefficient, transit_mode_constant)
    """
    logging.debug(f"Getting mode choice coefficients for trip purpose: {trip_purpose}, destination zone: {destination_zone_id}")

    # capitalize everything for consistency
    _trip_purpose_caps = trip_purpose.upper()

    # flag whether destination is in cbd
    _is_cbd_flag = destination_zone_id in cbd_taz_ids

    # first, check whether bare purpose is in the dict
    if _trip_purpose_caps in mode_choice_coefficients.keys():  # in case not HBWH or HBWL
        coeffs = mode_choice_coefficients[_trip_purpose_caps]
        logging.debug(f"Found coefficients for purpose {_trip_purpose_caps}: {coeffs}")

        # get values out
        beta_time = coeffs["beta_time"]
        mode_constant = coeffs["mode_constant"]

        # return as tuple
        return beta_time, mode_constant

    # if made it past that, then in case of HBWH or HBWL, need to check cbd vs non-cbd
    if _is_cbd_flag:
        _flagged_purpose = f"{_trip_purpose_caps}_CBD"
    else:
        _flagged_purpose = f"{_trip_purpose_caps}_NON_CBD"

    if _flagged_purpose in mode_choice_coefficients.keys():
        coeffs = mode_choice_coefficients[_flagged_purpose]
        logging.debug(f"Found coefficients for purpose {_flagged_purpose}: {coeffs}")

        # get values out
        beta_time = coeffs["beta_time"]
        mode_constant = coeffs["mode_constant"]

        # return as tuple
        return beta_time, mode_constant

    # if still not found, return default
    logging.warning(f"Trip purpose {trip_purpose} with cbd flag {_is_cbd_flag} not found, using DEFAULT coefficients")
    default_coeffs = mode_choice_coefficients["DEFAULT"]

    # get default values
    beta_time = default_coeffs["beta_time"]
    mode_constant = default_coeffs["mode_constant"]

    return beta_time, mode_constant


# based on the travel times and value of time, calculate the probability of choosing transit
def calculate_mode_choice_probability(transit_travel_time: float, driving_travel_time: float,
                                      beta_time: float, mode_constant: float) -> float:
    """
    Calculates the probability of choosing transit using logit mode choice model

    Formula used (simple logit model based on Ortuzar and Willumsen, 2011):

    utility_transit = beta_time * transit_travel_time plus mode_constant
    utility_driving = beta_time * driving_travel_time
    probability_of_taking_transit = (e ** utility_transit) / (exp(utility_transit) + e ** utility_driving))

    :param transit_travel_time: transit travel time (minutes)
    :param driving_travel_time: auto travel time (minutes)
    :param beta_time: time coefficient (negative utility)
    :param mode_constant: transit mode constant (bias term)
    :return: probability of choosing transit (0 to 1), or nan if inputs invalid
    """
    # handling missing values
    if pd.isna(transit_travel_time) or pd.isna(driving_travel_time):
        # use numpy nan for missing values rather than None
        logging.debug(f"missing travel time(s): transit travel time: {transit_travel_time} driving travel time: {driving_travel_time}")
        return np.nan

    # calculate utilities
    utility_transit = beta_time * transit_travel_time + mode_constant
    utility_driving = beta_time * driving_travel_time

    # calculating probability using logit equation
    try:
        # need e to the utility values
        exp_transit = np.exp(utility_transit)
        exp_auto = np.exp(utility_driving)

        # then the probability of taking transit is
        probability_of_taking_transit = exp_transit / (exp_transit + exp_auto)

        # snoring logs
        logging.debug(f"calculated mode choice probability for transit : transit travel time: {transit_travel_time}, driving travel time :{driving_travel_time}")
        logging.debug(f"probability of choosing transit : {probability_of_taking_transit}")
        return probability_of_taking_transit

    except (OverflowError, FloatingPointError):
        # handle numerical overflow
        logging.warning(f"numerical overflow: transit travel time: {transit_travel_time}, driving travel time :{driving_travel_time}")
        return np.nan


# main ridership impact calculation function
def calculate_ridership_impact(trip_roster_dataframe: DataFrame, alternate_scenario_name: str,
                               current_transit_matrices_dir_path: str, alternate_gtfs_matrices_dir_path: str,
                               driving_matrices_dir_path: str, date_of_analysis: str = "20251209",
                               times_of_day_to_analyze: tuple[str] = all_times_to_analyze) -> DataFrame:
    """
    Calculate ridership impact using vectorized operations for massive speedup

    :param trip_roster_dataframe: dataframe of all trips
    :param alternate_scenario_name: name of alternate scenario for file naming consistency
    :param current_transit_matrices_dir_path: path of current transit network travel time matrices directory
    :param alternate_gtfs_matrices_dir_path: path of alternate transit network travel time matrices directory
    :param driving_matrices_dir_path: path of driving network travel time matrices directory
    :param date_of_analysis: date of analysis for file naming consistency (str format YYYYMMDD)
    :param times_of_day_to_analyze: list of times to analyze

    :return: dataframe with results for each trip
    """
    logging.info("Starting vectorized ridership impact calculation")
    logging.info(f"Processing {len(trip_roster_dataframe)} trips")

    # load all matrices for each time period
    current_transit_matrices_dict = {}
    alternate_transit_matrices_dict = {}
    driving_matrices_dict = {}

    for time_code in times_of_day_to_analyze:
        logging.debug(f"Loading matrices for {time_code}")

        # construct filenames
        current_transit_filename = f"current_network_{date_of_analysis}_{time_code}_matrix.csv"
        alternate_transit_filename = f"{alternate_scenario_name}_{date_of_analysis}_{time_code}_matrix.csv"
        driving_filename = f"driving_{date_of_analysis}_{time_code}_matrix.csv"

        # full paths
        current_transit_filepath = os.path.join(current_transit_matrices_dir_path, current_transit_filename)
        alternate_transit_filepath = os.path.join(alternate_gtfs_matrices_dir_path, alternate_transit_filename)
        driving_filepath = os.path.join(driving_matrices_dir_path, driving_filename)

        # error checking
        if not os.path.exists(current_transit_filepath):
            logging.error(f"Current transit matrix not found: {current_transit_filepath}")
            continue
        if not os.path.exists(alternate_transit_filepath):
            logging.error(f"Alternate transit matrix not found: {alternate_transit_filepath}")
            continue
        if not os.path.exists(driving_filepath):
            logging.error(f"Driving matrix not found: {driving_filepath}")
            continue

        # read matrices and fill NaN values with 240 minutes (4 hours) since r5py stopped at 180 minutes
        # this means OD pairs with NaN are just very long trips rather than no connection
        current_transit_matrices_dict[time_code] = pd.read_csv(current_transit_filepath, index_col=0)
        alternate_transit_matrices_dict[time_code] = pd.read_csv(alternate_transit_filepath, index_col=0)
        driving_matrices_dict[time_code] = pd.read_csv(driving_filepath, index_col=0)

        # Convert index and columns to int for proper lookup
        for matrices_dict in [current_transit_matrices_dict, alternate_transit_matrices_dict, driving_matrices_dict]:
            matrices_dict[time_code].index = matrices_dict[time_code].index.astype(int)
            matrices_dict[time_code].columns = matrices_dict[time_code].columns.astype(int)

        logging.debug(f"Loaded matrices for time code: {time_code}")

    logging.info(f"Loaded {len(current_transit_matrices_dict)} time period matrices")

    # create a working copy
    results_df = trip_roster_dataframe.copy()

    # map time period names to time codes
    results_df["time_code"] = results_df["timeperiod"].map(time_period_to_time_rosetta)

    # filter out trips with no valid time code
    valid_trips_mask = results_df["time_code"].notna()
    results_df = results_df[valid_trips_mask].copy()
    logging.info(f"After time code filtering: {len(results_df):,} trips")

    # determine if destination is in CBD (vectorized)
    results_df["is_cbd"] = results_df["d_zone"].isin(cbd_taz_ids)

    # create purpose key for coefficient lookup (vectorized)
    def get_purpose_key(row):
        purpose_upper = row["purpose"].upper()
        # check if it needs CBD suffix
        if purpose_upper in ["HBWH", "HBWL"]:
            if row["is_cbd"]:
                return f"{purpose_upper}_CBD"
            else:
                return f"{purpose_upper}_NON_CBD"
        elif purpose_upper in mode_choice_coefficients:
            return purpose_upper
        else:
            return "DEFAULT"

    results_df["purpose_key"] = results_df.apply(get_purpose_key, axis=1)

    # map coefficients (vectorized)
    results_df["beta_time"] = results_df["purpose_key"].map(
        lambda x: mode_choice_coefficients[x]["beta_time"]
    )
    results_df["mode_constant"] = results_df["purpose_key"].map(
        lambda x: mode_choice_coefficients[x]["mode_constant"]
    )

    logging.info("Mapped mode choice coefficients")

    # now lookup travel times for each time period
    results_list = []

    for time_code in times_of_day_to_analyze:
        logging.info(f"Processing time period {time_code}")

        # filter to trips in this time period
        period_trips = results_df[results_df["time_code"] == time_code].copy()

        if len(period_trips) == 0:
            continue

        # get matrices for this time period
        current_transit_matrix = current_transit_matrices_dict[time_code]
        alternate_transit_matrix = alternate_transit_matrices_dict[time_code]
        driving_matrix = driving_matrices_dict[time_code]

        # vectorized lookup using list comprehension
        period_trips["transit_time_current"] = [
            current_transit_matrix.loc[o, d] if (o in current_transit_matrix.index and d in current_transit_matrix.columns) else np.nan
            for o, d in zip(period_trips["o_zone"], period_trips["d_zone"])
        ]

        period_trips["transit_time_alternate"] = [
            alternate_transit_matrix.loc[o, d] if (o in alternate_transit_matrix.index and d in alternate_transit_matrix.columns) else np.nan
            for o, d in zip(period_trips["o_zone"], period_trips["d_zone"])
        ]

        period_trips["driving_time"] = [
            driving_matrix.loc[o, d] if (o in driving_matrix.index and d in driving_matrix.columns) else np.nan
            for o, d in zip(period_trips["o_zone"], period_trips["d_zone"])
        ]

        results_list.append(period_trips)
        logging.info(f"Completed {time_code}: {len(period_trips):,} trips")

    # concatenate all time periods back together
    results_df = pd.concat(results_list, ignore_index=True)
    logging.info(f"Combined all time periods: {len(results_df):,} trips")

    # CRITICAL: filter to only trips that had valid transit connections in BOTH scenarios
    # this excludes trips where removing monroe created new NaN values (no connection)
    # and trips that were already NaN in baseline (not viable transit trips)
    valid_in_both_scenarios = (
            (results_df["transit_time_current"] < 900) &  # was valid in current (not NaN)
            (results_df["transit_time_alternate"] < 900) &  # is valid in alternate (not NaN)
            results_df["transit_time_current"].notna() &
            results_df["transit_time_alternate"].notna() &
            results_df["driving_time"].notna()
    )

    excluded_trip_count = len(results_df) - valid_in_both_scenarios.sum()
    excluded_person_trips = results_df.loc[~valid_in_both_scenarios, "trips"].sum()

    results_df = results_df[valid_in_both_scenarios].copy()

    logging.info(f"Filtered to trips with valid transit in BOTH scenarios: {len(results_df):,} trips")
    logging.info(f"Excluded {excluded_trip_count:,} trip records ({excluded_person_trips:,.0f} person trips)")

    # check if we have valid travel times
    has_valid_times = (
            results_df["transit_time_current"].notna() &
            results_df["transit_time_alternate"].notna() &
            results_df["driving_time"].notna()
    )
    logging.info(f"Trips with valid travel times: {has_valid_times.sum():,}")

    # vectorized mode choice calculations
    # calculate utilities (vectorized operations on entire columns)
    results_df["utility_transit_current"] = (
            results_df["beta_time"] * results_df["transit_time_current"] +
            results_df["mode_constant"]
    )
    results_df["utility_transit_alternate"] = (
            results_df["beta_time"] * results_df["transit_time_alternate"] +
            results_df["mode_constant"]
    )
    results_df["utility_driving"] = results_df["beta_time"] * results_df["driving_time"]

    # calculate probabilities (vectorized) with overflow protection
    # clip utilities to prevent overflow
    results_df["utility_transit_current"] = results_df["utility_transit_current"].clip(-500, 500)
    results_df["utility_transit_alternate"] = results_df["utility_transit_alternate"].clip(-500, 500)
    results_df["utility_driving"] = results_df["utility_driving"].clip(-500, 500)

    results_df["exp_transit_current"] = np.exp(results_df["utility_transit_current"])
    results_df["exp_transit_alternate"] = np.exp(results_df["utility_transit_alternate"])
    results_df["exp_driving"] = np.exp(results_df["utility_driving"])

    results_df["P_transit_baseline"] = (
            results_df["exp_transit_current"] /
            (results_df["exp_transit_current"] + results_df["exp_driving"])
    )
    results_df["P_transit_alternate"] = (
            results_df["exp_transit_alternate"] /
            (results_df["exp_transit_alternate"] + results_df["exp_driving"])
    )

    # log probability distribution
    logging.info(f"Transit probability stats - baseline mean: {results_df['P_transit_baseline'].mean():.4f}, median: {results_df['P_transit_baseline'].median():.4f}")
    logging.info(f"Trips with P_transit_baseline > 0.01: {(results_df['P_transit_baseline'] > 0.01).sum():,}")
    logging.info(f"Trips with P_transit_baseline > 0.50: {(results_df['P_transit_baseline'] > 0.50).sum():,}")

    # calculate probability ratio with better handling
    # initialize with NaN
    results_df["probability_ratio"] = np.nan
    results_df["new_trips"] = np.nan
    results_df["trip_change"] = np.nan
    results_df["pct_change"] = np.nan

    # only calculate for trips where we have valid travel times and baseline probability is positive (cant divide by zero)
    valid_calc_mask = (
            has_valid_times &
            (results_df["P_transit_baseline"] > 0) &
            (results_df["P_transit_baseline"].notna()) &
            (results_df["P_transit_alternate"].notna())
    )

    logging.info(f"Trips valid for calculation: {valid_calc_mask.sum():,}")

    # calculate for valid trips
    results_df.loc[valid_calc_mask, "probability_ratio"] = (
            results_df.loc[valid_calc_mask, "P_transit_alternate"] /
            results_df.loc[valid_calc_mask, "P_transit_baseline"]
    )

    results_df.loc[valid_calc_mask, "new_trips"] = (
            results_df.loc[valid_calc_mask, "trips"] *
            results_df.loc[valid_calc_mask, "probability_ratio"]
    )

    results_df.loc[valid_calc_mask, "trip_change"] = (
            results_df.loc[valid_calc_mask, "new_trips"] -
            results_df.loc[valid_calc_mask, "trips"]
    )

    # calculate percent change (avoiding division by zero)
    nonzero_trips = results_df.loc[valid_calc_mask, "trips"] > 0
    results_df.loc[valid_calc_mask & nonzero_trips, "pct_change"] = (
            (results_df.loc[valid_calc_mask & nonzero_trips, "trip_change"] /
             results_df.loc[valid_calc_mask & nonzero_trips, "trips"]) * 100
    )

    # rename columns to match original output
    results_df = results_df.rename(columns={
        "trips": "baseline_trips",
        "transit_time_current": "transit_travel_time_current",
        "transit_time_alternate": "transit_travel_time_alternate",
        "driving_time": "travel_time_driving",
        "timeperiod": "time_period"
    })

    # select and order columns to match original output
    output_columns = [
        "o_zone", "d_zone", "purpose", "time_period", "time_code",
        "baseline_trips", "transit_travel_time_current", "transit_travel_time_alternate",
        "travel_time_driving", "P_transit_baseline", "P_transit_alternate",
        "probability_ratio", "new_trips", "trip_change", "pct_change"
    ]

    results_df = results_df[output_columns].copy()

    # filter to only trips where we successfully calculated results
    valid_results = results_df["probability_ratio"].notna()
    processed_count = valid_results.sum()
    skipped_count = len(results_df) - processed_count

    results_df = results_df[valid_results].copy()

    logging.info(f"Finished processing: {processed_count:,} trips processed, {skipped_count:,} skipped")
    logging.info(f"Successfully calculated ridership impact for {len(results_df):,} trips")

    return results_df


def get_valid_zones_from_matrices(matrices_dict: dict[str, DataFrame]) -> set[int]:
    """
    Get the set of all valid zone ids from travel time matrices (i.e., those that exist in the matrices)

    :param matrices_dict: dictionary of travel time matrices

    :return: set of valid zone ids that exist in the matrices
    """
    logging.info("Getting valid zones from travel time matrices")

    # get zones from first matrix since all matrices should have same zones and structure
    first_matrix = list(matrices_dict.values())[0]
    # using union of index and columns to get all zones with at least one entry
    valid_zone_ids = set(first_matrix.index) | set(first_matrix.columns)

    # they call me mr logger
    logging.info(f"valid zones in matrices: {len(valid_zone_ids)}")
    logging.debug(f"zone range: {min(valid_zone_ids)} to {max(valid_zone_ids)}")

    # for the sake of debugging, check whether zones are actually equal, but not necessary
    for time_code, matrix in matrices_dict.items():
        # get zones in this matrix
        matrix_zones = set(matrix.index) | set(matrix.columns)
        # if any of them dont match, log warning
        if matrix_zones != valid_zone_ids:
            logging.warning(f"Zone mismatch in matrix {time_code}")


    # return the set of valid zone ids
    return valid_zone_ids


def filter_trip_roster_by_valid_zones(trip_roster_dataframe: DataFrame,
                                      valid_zone_ids: set) -> DataFrame:
    """
    Filter trip roster to only include trips where both origin and destination are in the valid zone set

    :param trip_roster_dataframe: trip roster dataframe with all trips
    :param valid_zone_ids: set of valid zone ids from travel time matrices
    :return: filtered trip roster dataframe
    """
    original_trip_record_count = len(trip_roster_dataframe)
    original_person_trips_total = trip_roster_dataframe["trips"].sum()

    # filter to only trips with valid origin and destination zones
    trip_roster_filtered = trip_roster_dataframe[
        (trip_roster_dataframe["o_zone"].isin(valid_zone_ids)) &
        (trip_roster_dataframe["d_zone"].isin(valid_zone_ids))
        ].copy()

    excluded_trip_records = original_trip_record_count - len(trip_roster_filtered)
    excluded_person_trips = original_person_trips_total - trip_roster_filtered["trips"].sum()
    excluded_person_trips_percentage = (excluded_person_trips / original_person_trips_total) * 100

    logging.warning(f"excluded {excluded_trip_records:,} trip records with zones not in matrices")
    logging.warning(f"excluded {excluded_person_trips:,.0f} person trips ({excluded_person_trips_percentage:.1f}%)")
    logging.info(f"valid trips remaining: {len(trip_roster_filtered):,}")
    logging.info(f"valid person trips: {trip_roster_filtered['trips'].sum():,.0f}")

    return trip_roster_filtered


""" Summary Statistics """


def summarize_results(results_dataframe: DataFrame) -> None:
    """
    Gets summary statistics of ridership impact analysis

    :param results_dataframe: dataframe with trip-level results
    """
    # overall impact
    total_baseline_trips = results_dataframe["baseline_trips"].sum()
    total_project_trips = results_dataframe["new_trips"].sum()
    total_trip_change = results_dataframe["trip_change"].sum()
    percent_change = (total_trip_change / total_baseline_trips) * 100

    print(f"\n{'=' * 70}")
    print(f"monroe station removal ridership impact summary")
    print(f"{'=' * 70}")
    print(f"baseline daily transit trips: {total_baseline_trips:,.0f}")
    print(f"project daily transit trips:  {total_project_trips:,.0f}")
    print(f"net change:                   {total_trip_change:,.0f} ({percent_change:.2f}%)")
    print(f"{'=' * 70}")

    # by time period
    period_summary = results_dataframe.groupby("time_period").agg({
        "baseline_trips": "sum",
        "new_trips": "sum",
        "trip_change": "sum"
    })
    period_summary["pct_change"] = (period_summary["trip_change"] / period_summary["baseline_trips"]) * 100
    print(period_summary.to_string())

    # by purpose
    print(f"\nimpact by trip purpose:")
    print(f"{'-' * 70}")
    purpose_summary = results_dataframe.groupby("purpose").agg({
        "baseline_trips": "sum",
        "new_trips": "sum",
        "trip_change": "sum"
    })
    purpose_summary["pct_change"] = (purpose_summary["trip_change"] / purpose_summary["baseline_trips"]) * 100
    print(purpose_summary.to_string())

    # travel time changes
    print(f"\ntravel time statistics:")
    print(f"{'-' * 70}")
    mean_time_change = (results_dataframe["transit_travel_time_alternate"] - results_dataframe["transit_travel_time_current"]).mean()
    max_time_increase = (results_dataframe["transit_travel_time_alternate"] - results_dataframe["transit_travel_time_current"]).max()
    print(f"mean transit time change: {mean_time_change:.2f} minutes")
    print(f"max transit time increase: {max_time_increase:.2f} minutes")

    # trips with significant impact
    significant_impact_trips = results_dataframe[results_dataframe["trip_change"] < -1]
    print(f"\nod pairs with more than 1 trip per day loss: {len(significant_impact_trips):,}")
    print(f"total trips lost from these pairs: {significant_impact_trips['trip_change'].sum():,.0f}")


# this is where the main function runs SO exciting
if __name__ == "__main__":
    logging.info("Running full demand model analysis for monroe station removal scenario")

    # load trip roster
    cmap_trip_roster = load_trip_roster(
        filter_transit_only=True,
        filter_valid_timeperiod=True,
        assign_missing_timeperiods=True
    )

    # load current transit matrices to get valid zones
    current_transit_matrices = load_travel_time_matrices("current_transit")
    valid_zones = get_valid_zones_from_matrices(current_transit_matrices)

    # filter trip roster to only valid zones
    cmap_trip_roster = filter_trip_roster_by_valid_zones(cmap_trip_roster, valid_zones)

    # base path for matrices
    matrices_base_path = os.path.join(output_data_path, "matrices")
    # then for each scenario
    current_transit_matrices_dir_path = os.path.join(matrices_base_path, "current_transit")
    no_monroe_matrices_dir_path = os.path.join(matrices_base_path, "no_monroe")
    driving_matrices_dir_path = os.path.join(matrices_base_path, "driving")

    # calculate ridership impact (pass directory paths, not loaded matrices)
    results_df = calculate_ridership_impact(trip_roster_dataframe=cmap_trip_roster,
                                            alternate_scenario_name="no_monroe",
                                            current_transit_matrices_dir_path=current_transit_matrices_dir_path,
                                            alternate_gtfs_matrices_dir_path=no_monroe_matrices_dir_path,
                                            driving_matrices_dir_path=driving_matrices_dir_path)

    # save detailed results
    output_file = os.path.join(output_data_path, "demand_model", "monroe_ridership_secondary_detailed.csv")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    results_df.to_csv(output_file, index=False)
    logging.info(f"saved detailed results to: {output_file}")

    # print summary
    summarize_results(results_df)

    logging.info(f"Finished demand model analysis for no monroe scenario")