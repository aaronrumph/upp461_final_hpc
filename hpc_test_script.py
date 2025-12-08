""" This script is meant to test the performance of the HPC to see whether feasible or not to use detailed
# itinerary calculations there. It uses a small subset of the data for testing purposes."""

import logging
import os
from functools import wraps

import r5py
import zipfile
from zipfile import ZipFile, ZIP_DEFLATED
from pathlib import Path
from datetime import datetime as dt_dt
import shutil
import pandas as pd
import geopandas as gpd
import multiprocessing as mp
import math
import time
import datetime
from concurrent.futures import ThreadPoolExecutor
import itertools

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# timer function to use as decorator
def timer(func):
    """ Decorator function to time other functions"""

    @wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        logging.info(f"Finished {func.__name__!r} in {run_time:.2f} secs")
        return value

    return wrapper_timer


# function to clean the r5py cache
def clean_r5py_cache():
    """Completely cleans r5py cache to avoid corruption issues."""
    r5py_cache = Path.home() / "AppData" / "Local" / "r5py"

    if r5py_cache.exists():
        try:
            logging.info("Completely cleaning r5py cache directory")
            shutil.rmtree(r5py_cache, ignore_errors=True)
            r5py_cache.mkdir(parents=True, exist_ok=True)
            logging.info("Successfully cleaned and recreated r5py cache")
        except Exception as e:
            logging.warning(f"Could not clean r5py cache: {e}")


def clean_gtfs_zip(zip_path):
    """
    Removes empty optional GTFS files from a zip that cause R5 validation errors.
    Creates a new cleaned version of the zip file.

    :param zip_path: path to zip file
    :return: path to cleaned zip file (or original if no cleaning needed)
    """

    # required GTFS files that must be present (even if small)
    required_files = {
        "agency.txt", "stops.txt", "routes.txt",
        "trips.txt", "stop_times.txt"
    }

    # optional files that R5 complains about when empty
    optional_files_to_check = {
        "areas.txt", "farezone_attributes.txt", "frequencies.txt",
        "transfers.txt", "fare_rules.txt", "fare_attributes.txt",
        "calendar_dates.txt", "attributions.txt", "feed_info.txt",
        "translations.txt", "levels.txt", "pathways.txt"
    }

    # check if cleaning is needed
    needs_cleaning = False
    files_to_remove = set()

    with ZipFile(zip_path, "r") as original_zip:
        for file_info in original_zip.filelist:
            if file_info.filename.endswith(".txt"):
                base_filename = os.path.basename(file_info.filename)

                # remove if it's an optional file AND either:
                # 1. file size is < 100 bytes (empty or header-only)
                # 2. file has only a header line (check content)
                if base_filename in optional_files_to_check:
                    file_content = original_zip.read(file_info.filename).decode('utf-8', errors='ignore')
                    lines = file_content.strip().split('\n')

                    # remove if: empty, only whitespace, or only has header (1 line)
                    if len(lines) <= 1 or file_info.file_size < 50:
                        needs_cleaning = True
                        files_to_remove.add(file_info.filename)

    if not needs_cleaning:
        logging.debug(f"{os.path.basename(zip_path)}: No empty optional files found")
        return zip_path

    # create cleaned version
    logging.info(f"Cleaning {os.path.basename(zip_path)}")
    cleaned_zip_path = zip_path.replace(".zip", "_cleaned.zip")

    # remove old cleaned version if it exists
    if os.path.exists(cleaned_zip_path):
        os.remove(cleaned_zip_path)

    with ZipFile(zip_path, "r") as original_zip:
        with ZipFile(cleaned_zip_path, "w", ZIP_DEFLATED) as cleaned_zip:
            for file_info in original_zip.filelist:
                # skip directories and files marked for removal
                if file_info.is_dir():
                    continue

                if file_info.filename not in files_to_remove:
                    # Copy file content (not file_info to avoid metadata issues)
                    file_content = original_zip.read(file_info.filename)
                    cleaned_zip.writestr(file_info.filename, file_content, compress_type=ZIP_DEFLATED)
                else:
                    logging.info(f"Removed: {file_info.filename}")

    logging.info(f"Created cleaned version: {os.path.basename(cleaned_zip_path)}")
    return cleaned_zip_path


# just cleans all gtfs zips in a directory
@timer
def clean_gtfs_zips_in_directory(gtfs_dir):
    """
    Cleans all GTFS zip files in a directory by removing empty optional files.
    :param gtfs_dir: directory containing GTFS zip files
    :return: list of paths to cleaned GTFS zip files
    """
    logging.info(f"Cleaning gtfs zips in directory: {gtfs_dir}")

    cleaned_zip_paths = []
    # go through each and use the clean function
    for filename in os.listdir(gtfs_dir):
        # Skip already-cleaned files and non-zip files
        if filename.endswith(".zip") and "_cleaned.zip" not in filename:
            zip_path = os.path.join(gtfs_dir, filename)
            logging.info(f"Processing: {filename}")  # Add this line
            try:
                cleaned_zip_path = clean_gtfs_zip(zip_path)
                if cleaned_zip_path is not None:
                    cleaned_zip_paths.append(cleaned_zip_path)
            except Exception as e:
                logging.error(f"Failed to process {filename}: {e}")
                continue

    logging.info(f"Cleaned zips in directory: {gtfs_dir}")
    return cleaned_zip_paths

# function to setup r5 network
@timer
def setup_r5_network(osm_pbf_path, gtfs_zip_paths):
    """
    Sets up R5 network with given OSM PBF and GTFS zip files.
    :param osm_pbf_path:
    :param gtfs_zip_paths:
    :return: r5py.R5TransportNetwork object
    """
    logging.info("Setting up R5 network")
    # just take the input data and build the network
    r5_network = r5py.TransportNetwork(osm_pbf=osm_pbf_path, gtfs=gtfs_zip_paths)
    logging.info("R5 network setup complete")

    return r5_network


# function to calculate ideal chunk size for multiprocessing
def calculate_ideal_chunk_size(origins_count, destinations_count):
    """
    Calculates ideal chunk size for multiprocessing based on number of origins/destinations and available CPUs.
    :param origins_count: Number of origin points
    :param destinations_count: Number of destination points
    :return: Tuple of (origin_or_destination, number_of_chunks, ideal_chunk_size)
    """
    available_cpus = mp.cpu_count()
    logging.info(f"Available CPU cores: {available_cpus}")

    total_od_pairs = origins_count * destinations_count
    logging.info(f"Total OD pairs: {total_od_pairs:,}")

    # determine whether to chunk by origins or destinations

    # in case where more origins than destinations, chunk by origins
    if origins_count > destinations_count:
        logging.info("Chunking by origins (origins > destinations)")
        origin_or_destination = "origin"

        # set number of chunks to min of available cpus or origins count so not trying to make more chunks than data
        number_of_chunks = min(available_cpus, origins_count)
        # ideal chunk size is total origins divided by number of chunks
        ideal_chunk_size = math.ceil(origins_count / number_of_chunks)

        logging.info(f"Will split {origins_count} origins into {number_of_chunks} chunks of ~{ideal_chunk_size} each")
        return origin_or_destination, number_of_chunks, ideal_chunk_size

    # otherwise chunk by destinations
    else:
        logging.info("Chunking by destinations (destinations >= origins)")
        origin_or_destination = "destination"
        # set number of chunks to min of available cpus or destinations count
        number_of_chunks = min(available_cpus, destinations_count)
        # ideal chunk size is total destinations divided by number of chunks
        ideal_chunk_size = math.ceil(destinations_count / number_of_chunks)

        logging.info(
            f"Will split {destinations_count} destinations into {number_of_chunks} chunks of ~{ideal_chunk_size} each")
        return origin_or_destination, number_of_chunks, ideal_chunk_size


# function to split gdf into chunks based on ideal chunk size
def split_gdf_into_chunks(gdf, number_of_chunks, ideal_chunk_size):
    """
    Splits a GeoDataFrame into chunks for multiprocessing.

    :param gdf: GeoDataFrame to split
    :param number_of_chunks: Number of chunks to create
    :param ideal_chunk_size: Target size for each chunk

    :return list: List of GeoDataFrame chunks
    """
    chunks = []
    for chunk_idx in range(number_of_chunks):
        start_idx = chunk_idx * ideal_chunk_size
        end_idx = min((chunk_idx + 1) * ideal_chunk_size, len(gdf))
        chunk = gdf.iloc[start_idx:end_idx].copy()
        chunks.append(chunk)
        logging.debug(f"Chunk {chunk_idx}: rows {start_idx} to {end_idx - 1} ({len(chunk)} items)")

    return chunks


@timer
def calculate_itineraries_chunk(transport_network, transport_modes, origins_gdf, destinations_gdf,
                                departure_datetime, chunk_id):
    """
    Worker function to calculate itineraries for a chunk of origins/destinations.
    NO TIME LIMIT - calculates all possible routes.

    :param transport_network: the r5py transportation network | r5py.R5TransportNetwork
    :param transport_modes: list of transport modes to consider | r5py.TransportMode
    :param origins_gdf: GeoDataFrame of origin points for this chunk | geopandas.GeoDataFrame
    :param destinations_gdf: GeoDataFrame of destination points for this chunk | geopandas.GeoDataFrame
    :param departure_datetime: datetime for departure time | datetime.datetime
    :param chunk_id: identifier for this chunk | int
    """
    chunk_start_time = time.perf_counter()

    logging.info(
        f"[Chunk {chunk_id}] Processing {len(origins_gdf)} origins x {len(destinations_gdf)} destinations = {len(origins_gdf) * len(destinations_gdf)} OD pairs")

    try:
        # uses the r5py detailed itineraries function to get all possible routes for the chunk
        detailed_itineraries = r5py.DetailedItineraries(
            transport_network=transport_network,
            origins=origins_gdf,
            destinations=destinations_gdf,
            departure=departure_datetime,
            transport_modes=transport_modes,
            snap_to_network=True,
            max_public_transport_rides=4,
            max_time_walking=datetime.timedelta(minutes=30))

        chunk_elapsed = time.perf_counter() - chunk_start_time
        logging.info(
            f"[Chunk {chunk_id}] completed: {len(detailed_itineraries)} itineraries calculated in {chunk_elapsed:.1f} seconds")
        return detailed_itineraries, chunk_elapsed

    except Exception as e:
        chunk_elapsed = time.perf_counter() - chunk_start_time
        logging.error(f"[Chunk {chunk_id}] Error after {chunk_elapsed:.1f} seconds: {str(e)}")
        return None, chunk_elapsed


# calculate simple travel times for chunk
@timer
def calculate_travel_times_chunk(transport_network, transport_modes, origins_gdf, destinations_gdf,
                                 departure_datetime, chunk_id):
    """
    Worker function to calculate itineraries for a chunk of origins/destinations.
    NO TIME LIMIT - calculates all possible routes.

    :param transport_network: the r5py transportation network | r5py.R5TransportNetwork
    :param transport_modes: list of transport modes to consider | r5py.TransportMode
    :param origins_gdf: GeoDataFrame of origin points for this chunk | geopandas.GeoDataFrame
    :param destinations_gdf: GeoDataFrame of destination points for this chunk | geopandas.GeoDataFrame
    :param departure_datetime: datetime for departure time | datetime.datetime
    :param chunk_id: identifier for this chunk | int
    """
    chunk_start_time = time.perf_counter()

    logging.info(
        f"[Chunk {chunk_id}] Processing {len(origins_gdf)} origins x {len(destinations_gdf)} destinations = {len(origins_gdf) * len(destinations_gdf)} OD pairs")

    try:
        # uses the r5py detailed itineraries function to get all possible routes for the chunk
        travel_times = r5py.TravelTimeMatrix(
            transport_network=transport_network,
            origins=origins_gdf,
            destinations=destinations_gdf,
            departure=departure_datetime,
            transport_modes=transport_modes,
            snap_to_network=True
        )

        chunk_elapsed = time.perf_counter() - chunk_start_time
        logging.info(
            f"[Chunk {chunk_id}] completed: {len(travel_times)} itineraries calculated in {chunk_elapsed:.1f} seconds")
        return travel_times, chunk_elapsed

    except Exception as e:
        chunk_elapsed = time.perf_counter() - chunk_start_time
        logging.error(f"[Chunk {chunk_id}] Error after {chunk_elapsed:.1f} seconds: {str(e)}")
        return None, chunk_elapsed


def run_analysis(testing_func, r5py_network, origins_gdf, destinations_gdf, departure_datetime, transport_modes):
    """
    Runs the full analysis by splitting the origins/destinations into chunks and processing them in parallel.

    :param r5py_network: r5py transportation network | r5py.R5TransportNetwork
    :param origins_gdf: GeoDataFrame of origin points | geopandas.GeoDataFrame
    :param destinations_gdf: GeoDataFrame of destination points | geopandas.GeoDataFrame
    :param departure_datetime: datetime for departure time | datetime.datetime
    :param transport_modes: list of transport modes to consider | r5py.TransportMode
    """

    # calculate ideal chunking strategy
    origin_or_destination, number_of_chunks, ideal_chunk_size = calculate_ideal_chunk_size(
        len(origins_gdf), len(destinations_gdf)
    )

    # split gdfs into chunks
    if origin_or_destination == "origin":
        origin_chunks = split_gdf_into_chunks(origins_gdf, number_of_chunks, ideal_chunk_size)
        destination_chunks = [destinations_gdf] * number_of_chunks  # same destinations for all origin chunks
    else:
        destination_chunks = split_gdf_into_chunks(destinations_gdf, number_of_chunks, ideal_chunk_size)
        origin_chunks = [origins_gdf] * number_of_chunks  # same origins for all destination chunks

    # args for multiprocessing
    process_args = []

    # go through each chunk and prepare args
    for chunk_id in range(number_of_chunks):
        chunk_args = (r5py_network, transport_modes, origin_chunks[chunk_id],
                      destination_chunks[chunk_id], departure_datetime, chunk_id)
        process_args.append(chunk_args)

    # number of processes to use
    num_processes = min(mp.cpu_count(), number_of_chunks)

    try:
        # create pool and map the function to the args (using threadpool)
        with mp.pool.ThreadPool(num_processes) as pool:
            combined_results = pool.starmap(testing_func, process_args)

    except Exception as e:
        logging.error(f"Error during multiprocessing: {str(e)}")
        raise e

    # filter out failed chunks and convert to dataframes
    valid_results = []
    for chunk_id, (result, chunk_time) in enumerate(combined_results):
        if result is not None and len(result) > 0:
            valid_results.append(result)

    # combine results from all chunks
    combined_travel_times = pd.concat(valid_results, ignore_index=True)
    logging.info(f"Combined travel times: {len(combined_travel_times)} total itineraries")

    return combined_travel_times


def geojson_to_gdf(geojson_path):
    """
    Loads a GeoJSON file into a GeoDataFrame with specified CRS.

    :param geojson_path: path to GeoJSON file
    :return: GeoDataFrame
    """
    gdf = gpd.read_file(geojson_path)
    if "id" not in gdf.columns:
        gdf["id"] = range(len(gdf))
    return gdf


if __name__ == "__main__":
    # setup logging
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting current network analysis")

    # reset r5py cache
    logging.info("Resetting r5py cache")
    clean_r5py_cache()

    # input transportation network data (pbf for streets and gtfs)
    current_gtfs_data_dir = os.path.join(Path(__file__).parent, "input_data", "gtfs_data", "current")
    current_osm_pbf_path = os.path.join(Path(__file__).parent, "input_data", "pbf_data", "cmap_road_network.pbf")

    # first clean the gtfs zips
    current_gtfs_zip_files = clean_gtfs_zips_in_directory(gtfs_dir=current_gtfs_data_dir)

    # remove megabus feed due to known issues
    current_gtfs_zip_files = [f for f in current_gtfs_zip_files if 'megabus' not in f.lower()]
    logging.info(f"Using {len(current_gtfs_zip_files)} GTFS feeds (filtered out problematic feeds)")

    # create the r5py transportation network
    current_transportation_network = setup_r5_network(osm_pbf_path=current_osm_pbf_path,
                                                      gtfs_zip_paths=current_gtfs_zip_files)

    # dates to use (based on CMAP model documentation)
    departure_times = [dt_dt(2025, 12, 9, 0, 30, 0)]

    # load origins and destinations
    path_to_taz_geojson = os.path.join(Path(__file__).parent, "input_data", "taz_data", "tazs_centroids.geojson")
    taz_gdf = geojson_to_gdf(geojson_path=path_to_taz_geojson)
    logging.info(f"Loaded {len(taz_gdf)} TAZ centroids")

    # the travel modes that are need for the transit analysis
    transit_transport_modes = [r5py.TransportMode.WALK,
                               r5py.TransportMode.BUS,
                               r5py.TransportMode.RAIL,
                               r5py.TransportMode.SUBWAY,
                               r5py.TransportMode.TRAM,
                               r5py.TransportMode.GONDOLA,
                               r5py.TransportMode.CABLE_CAR,
                               r5py.TransportMode.FERRY]

    # run analysis for each departure time
    for departure_time in departure_times:
        logging.info(f"Running analysis for departure time: {departure_time}")

        # analysis name to keep track of each run
        analysis_name = f"current_network_{departure_time.strftime('%Y%m%d_%H%M')}"
        # run the analysis

        logging.info("Starting analysis")
        travel_times_df = run_analysis(testing_func=calculate_travel_times_chunk,
            r5py_network=current_transportation_network,
            origins_gdf=taz_gdf,
            destinations_gdf=taz_gdf,
            departure_datetime=departure_time,
            transport_modes=transit_transport_modes
        )

        # save results to csv
        output_data_dir_path = os.path.join(Path(__file__).parent, "output_data")
        travel_times_path = os.path.join(output_data_dir_path, "travel_times")

        # make sure output directory exists
        os.makedirs(travel_times_path, exist_ok=True)

        output_csv_path = os.path.join(travel_times_path, f"{analysis_name}.csv")
        travel_times_df.to_csv(output_csv_path, index=False)
        logging.info(f"Saved travel times for {analysis_name} to {output_csv_path}")

        detailed_itineraries_df = run_analysis(testing_func=calculate_itineraries_chunk,
                                       r5py_network=current_transportation_network,
                                       origins_gdf=taz_gdf,
                                       destinations_gdf=taz_gdf,
                                       departure_datetime=departure_time,
                                       transport_modes=transit_transport_modes
                                       )

        # save results to csv
        output_data_dir_path = os.path.join(Path(__file__).parent, "output_data")
        detailed_itineraries_path = os.path.join(output_data_dir_path, "detailed_itineraries")

        # make sure output directory exists
        os.makedirs(detailed_itineraries_path, exist_ok=True)

        output_csv_path = os.path.join(detailed_itineraries_path, f"{analysis_name}.csv")
        detailed_itineraries_df.to_csv(output_csv_path, index=False)
        logging.info(f"Saved travel times for {analysis_name} to {output_csv_path}")

