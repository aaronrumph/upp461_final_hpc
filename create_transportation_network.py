import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pickle
import logging
import osmnx as ox  # used for getting streets data
import geopandas as gpd
import pandas as pd
import networkx as nx
import time  # used for checking runtimes of functions/methods
import requests  # used for USGS API querying
import multiprocessing as mp  # used for querying in bulk
import asyncio  # used for querying USGS api asynchronously
import aiohttp  # same as requests (basically) but for asyncio
from itertools import repeat
import re
import platform
import isodate
from datetime import datetime
from zipfile import ZipFile, ZIP_DEFLATED
import random
from shapely.geometry import Point
import numpy as np
import r5py  # used for R5 routing
import math


import general_tools
# local module(s)
import network_types
from general_tools import *
from gtfs_tools import *
from gtfs_tools import transit_land_api_key


# logging setup
logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# osmnx setup
ox.settings.timeout = 3000
ox.settings.max_query_area_size = 5000000000000
ox.settings.use_cache = True



class CacheFolder:
    """ Class for cache folder, takes param
        network_snake_name: str (MUST be in snake case) which will be the name of cache folder

        Attributes:
            network_snake_name | str: snake name of the network (passing snake name rather than StreetNetwork object
                because CacheFolder class must preceed StreetNetwork class in the code)
            env_dir_path | Path: path to the environment directory
            path | str: path to the cache folder

        Methods:
            check_if_cache_folder_exists(): Returns True if cache folder already exists for the city.
            set_up_cache_folder(): Return True if there is already a cache folder for city. If not, creates one.
            reset_cache_folder(): Completely reset the cache folder for the city (highly unadvisable because deletes
                osm data and elevation data
    """

    def __init__(self, snake_name_with_scope):
        self.snake_name_with_scope = snake_name_with_scope
        self.env_dir_path = Path(__file__).parent
        self.path = os.path.join(self.env_dir_path, "place_caches", f"{self.snake_name_with_scope}_cache")

    def check_if_cache_folder_exists(self):
        """ Returns True if cache folder already exists for the city."""
        if os.path.exists(self.path):
            return True
        else:
            return False

    def set_up_cache_folder(self):
        """Return True if there is already a cache folder for city. If not, creates one."""
        if os.path.exists(self.path):
            raise Exception(f"There is already a cache folder for {self.snake_name_with_scope}")
        else:
            os.makedirs(self.path)

    def reset_cache_folder(self):
        # completely reset the cache folder for the city
        if not os.path.exists(self.path):
            raise Exception(f"Cannot reset the cache folder for {self.snake_name_with_scope} "
                            f"because no such folder exists"
                            )
        else:
            os.makedirs(self.path, exist_ok=True)


# simple Cache class with obvious methods (read, write, check if exists)
class Cache:
    """
    Class for cache for use in saving street network data.

    Attributes:
        cache_folder | CacheFolder obj: cache folder for the street network
        cache_name | str: name of the cache
        cache_path | str: path to the cache

    Methods:
        check_if_cache_already_exists(): checks if cache already exists
        read_cache_data(): reads cached data from cache file
        write_cache_data(): writes desired data to cache file
    """

    def __init__(self, cache_folder: CacheFolder, cache_name):
        self.cache_folder = cache_folder
        self.cache_name = cache_name
        self.cache_path = os.path.join(self.cache_folder.path, self.cache_name)

    def check_if_cache_already_exists(self):
        if os.path.exists(self.cache_path):
            return True
        else:
            return False

    def read_cache_data(self):
        if self.check_if_cache_already_exists():
            with open(self.cache_path, "rb") as cache_file:
                cache_data = pickle.load(cache_file)
            return cache_data
        else:
            raise Exception("Cannot get cache data because there is no cache")

    def write_cache_data(self, data_to_cache):
        with open(self.cache_path, "wb") as cache_file:
            pickle.dump(data_to_cache, cache_file)


# StreetNetwork class very bare bones, just gets street network graph and makes associated GeoDataframse
class StreetNetwork:
    """
    Class representing street network for a location

    Attributes:
        geographic_scope (str): Geographic scope of the street network {"place_only", "msa", "csa"}
        reference_place_list: list of reference places to get network for (len will be one if using city limits, other-
        wise, will contain all the places in the MSA if MSA desired and in CSA if CSA desired)
        network_type (str): type of network being created {"walk_no_z", "walk_z", "bike_no_z", "bike_z",

        Local attributes:
        "transit_no_z", "transit_z", "drive", "transit_plus_biking_no_z", "transit_plus_biking_z"}
        bound_boxes (list): Bounding boxes passed (if bounding box used, len of list will be one)
        snake_name (str): Snake name of the city
        cache_folder (CacheFolder): Cache folder for the street network
        graph_cache (Cache): Cache for the street network graph
        nodes_cache (Cache): Cache for the street network nodes
        edges_cache (Cache): Cache for the street network edges
        network_graph (networkx.Graph): Street network graph
        network_nodes (geopandas.GeoDataFrame): Street network nodes
        network_edges (geopandas.GeoDataFrame): Street network edges
        elevation_enabled (bool): Whether elevation is enabled
        osmnx_type (str): Type of street network for osmnx ("walk", "bike", "drive", "drive_service", "all", "all_public")

        Methods:
            get_street_network_from_osm(timer_on=True, reset=False): Gets street network from OpenStreetMaps or
            from cache
    """

    def __init__(self, geographic_scope: str, reference_place_list: list[ReferencePlace],  states_to_include: set[str],
                 network_type="walk_no_z"):
        #### need to make __init__ method cleaner ####
        self.geographic_scope = geographic_scope
        self.reference_place_list = reference_place_list
        self.states_to_include = states_to_include
        self.network_type = network_type

        self.merged_pbf_filepath = None

        # get place name and bbox out of reference place
        self.place_names = [reference_place.place_name for reference_place in reference_place_list]
        self.bound_boxes = [reference_place.bound_box for reference_place in reference_place_list]
        self.main_reference_place = reference_place_list[0]

        if self.main_reference_place.bound_box:
            self.geographic_scope = "bbox"

        # create snake name for StreetNetwork (the first item in the list is always the main place)
        self.snake_name = create_snake_name(self.main_reference_place)
        # create snake name with geographic scope encoded
        self.snake_name_with_scope = f"{self.snake_name}_{self.geographic_scope}"
        # link cache folder
        self.cache_folder = CacheFolder(self.snake_name_with_scope)
        # decode the network type into proper network type designation for osmnx query
        self.osmnx_type = network_types.network_types_attributes[self.network_type]["osmnx_network_type"]

        # check if there is a cache folder for desired street network
        if not self.cache_folder.check_if_cache_folder_exists():
            self.cache_folder.set_up_cache_folder()

        # setting up caches for this street network
        self.graph_cache = Cache(self.cache_folder, "graph_cache")
        self.nodes_cache = Cache(self.cache_folder, "nodes_cache")
        self.edges_cache = Cache(self.cache_folder, "edges_cache")
        self.edges_cache.cache_folder.check_if_cache_folder_exists()

        # placeholder for now, but will update in get_street_network_from_osm method
        # so can access gdfs and graph when passing instance as argument
        self.network_graph = None
        self.network_nodes = None
        self.network_edges = None
        self.elevation_enabled = False

    def get_street_network_graph_from_osmnx(self, timer_on=True, reset=False):
        """
        Main function for class that gets street network edges and nodes, either from cache, if cache exists, or from
        OpenStreetMaps (through osmnx)
        :param timer_on: logs run time for method if, on by default
        :param reset: if True, will reset cache and get street network from OSM
        :return: street network graph, and the nodes and egdes that make up the graph in geodataframes
        """
        process_start_time = time.perf_counter()
        logging.info("Getting street network")

        # first if not using cache or no cache data
        if reset or (not self.graph_cache.check_if_cache_already_exists()):
            logging.info("Getting street network from OSM")

            # if using city, getting OSM data if not using cache or if no cache exists
            if self.main_reference_place.bound_box is None:

                # if there is more than one reference place (not using city limits) need to combine street networks
                if len(self.reference_place_list) > 1:
                    network_graphs = []  # list of the networkx graph objects representing city street networks
                    for reference_place in self.reference_place_list:
                        logging.info(f"Getting street network for {reference_place.pretty_name}")

                        # getting each individual graph one at a time first
                        single_network_graph = ox.graph_from_place(reference_place.place_name,
                                                                   network_type=self.osmnx_type, retain_all=True,
                                                                   truncate_by_edge=True)
                        # need to use truncate_by_edge when using
                        # multiple reference places to avoid gaps at borders

                        # add each individual graph to the list so can combine
                        network_graphs.append(single_network_graph)

                    logging.info("Combining street networks")
                    # using compose all to combine the various street grids
                    network_graph = nx.compose_all(network_graphs)

                # when using city limits, only need street grid for main place
                elif len(self.reference_place_list) == 1:
                    logging.info(f"Getting street network for {self.main_reference_place.pretty_name} city proper")
                    network_graph = ox.graph_from_place(self.main_reference_place.place_name,
                                                        network_type=self.osmnx_type, retain_all=True)

                else:
                    raise Exception("Cannot get street network because no place was specified")

                # turn graph into gdfs
                network_nodes, network_edges = ox.graph_to_gdfs(network_graph, nodes=True, edges=True)


            # if using bound box, getting OSM Data
            else:
                logging.info(f"Getting street network for {self.main_reference_place.pretty_name}")
                network_graph = ox.graph_from_bbox(bbox=self.main_reference_place.bound_box,
                                                   network_type=self.osmnx_type, retain_all=True)
                # again graph to gdfs
                network_nodes, network_edges = ox.graph_to_gdfs(network_graph, nodes=True, edges=True)

        # otherwise using cached data
        else:
            logging.info("Using cached street network")
            network_graph = self.graph_cache.read_cache_data()
            network_nodes, network_edges = self.nodes_cache.read_cache_data(), self.edges_cache.read_cache_data()

        # just because I want to keep track of everything
        if timer_on:
            logging.info(f"Got street network from OSM in {time.perf_counter() - process_start_time} seconds")
            logging.info(f"Street network for {self.main_reference_place.place_name} {self.geographic_scope} had "
                         f"{len(network_nodes)} nodes and "
                         f"{len(network_edges)} edges")

        self.graph_cache.write_cache_data(network_graph)
        self.nodes_cache.write_cache_data(network_nodes)
        self.edges_cache.write_cache_data(network_edges)
        self.network_graph, self.network_nodes, self.network_edges = network_graph, network_nodes, network_edges
        return network_graph, network_nodes, network_edges

    # methods to count nodes and edges in street network. No real reason to exist other than interest
    def count_nodes(self):
        if self.network_nodes is None:
            raise Exception("Cannot count nodes because there are no nodes")
        else:
            return len(self.network_nodes)

    def count_edges(self):
        if self.network_edges is None:
            raise Exception("Cannot count edges because there are no edges")
        else:
            return len(self.network_edges)

    # adding a method that gets the PBF osm data because need for R5
    @timer
    def get_osm_pbf_data(self, reset=False):
        """
        Downloads the necessary osm data in pbf form for use in R5 routing based on geographic scope
        :param reset:
        :return:
        """
        logging.info("Getting OSM PBF data for place")

        # set up pbf folder in cache folder
        pbf_folder_path = os.path.join(self.cache_folder.path, "osm_pbf_data")
        os.makedirs(pbf_folder_path, exist_ok=True)\

        # the base url for geofabrik
        geofabrik_base_url = "https://download.geofabrik.de/north-america/us/"

        # dictionary mapping state names to geofabrik pbf file names'
        state_to_geofabrik_pbf_filename = {}

        # remove state abbreviations from set
        set_can_remove_from = self.states_to_include.copy()
        for state in self.states_to_include:
            # in case where state is abbreviation
            if state in general_tools.state_fips_and_abbreviations.keys():
                state_name = general_tools.state_fips_and_abbreviations[state]["name"]
                # then state is an abbreviation so need to get full state name and see if still need or not
                if state_name in self.states_to_include:
                    set_can_remove_from.remove(state)
                # in case where abbreviation is only thing provided, need to get full state name for geofabrik url
                else:
                    set_can_remove_from.add(state_name)
                    set_can_remove_from.remove(state)
        self.states_to_include = set_can_remove_from

        for state in self.states_to_include:
            # geofabrik uses dashes instead of spaces
            state_dash_name = state.lower().replace(" ", "-")

            # the url for the pbf file for the state
            state_pbf_url = f"{geofabrik_base_url}{state_dash_name}-latest.osm.pbf"

            # the output file url
            output_pbf_filepath = os.path.join(pbf_folder_path, f"{state_dash_name}.osm.pbf")
            # add paths to dictionary
            state_to_geofabrik_pbf_filename[state] = output_pbf_filepath

            # if the pbf already exists, need to delete it and then redownload it to ensure good data
            if os.path.exists(output_pbf_filepath):
                logging.info("PBF file already exists, deleting and redownloading to ensure good data")
                os.remove(output_pbf_filepath)
                logging.info(f"Downloading OSM PBF data for {state}")
                response = requests.get(state_pbf_url, stream=True)
                response.raise_for_status()

                # writing the content that was returned from geofabrik to a pbf file
                with open(output_pbf_filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

            # in case where no data 3exists yet
            else:
                logging.info(f"Downloading OSM PBF data for {state}")
                response = requests.get(state_pbf_url, stream=True)
                response.raise_for_status()

                # writing the content that was returned from geofabrik to a pbf file
                with open(output_pbf_filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

        # now if multiple states, need to merge them into one pbf file
        merged_pbf_filepath = os.path.join(pbf_folder_path, f"{self.snake_name_with_scope}_merged.osm.pbf")
        if len(self.states_to_include) > 1:
            logging.info("Merging state PBF files into one")
            # using osmosis to merge the pbf files
            state_pbf_filepaths = [state_to_geofabrik_pbf_filename[state] for state in self.states_to_include]

            import osmium as osm
            import sys


            # need to delete merged pbf file if it already exists
            if os.path.exists(merged_pbf_filepath):
                logging.info("Merged PBF file already exists, deleting and recreating to ensure good data")
                os.remove(merged_pbf_filepath)

            class MergeHandler(osm.SimpleHandler):
                def __init__(self, writer):
                    super().__init__()
                    self.writer = writer

                def node(self, n):
                    self.writer.add_node(n)

                def way(self, w):
                    self.writer.add_way(w)

                def relation(self, r):
                    self.writer.add_relation(r)

            def merge_pbf_files(input_paths, output_path):
                """
                Merge input PBFs into output_path using pyosmium.SimpleWriter.
                Ensure temporary files are created on the same drive by temporarily
                setting TMP/TEMP to the output folder.
                """
                import os as _os
                import tempfile

                # backup existing TMP/TEMP/TMPDIR
                _tmp_keys = ("TMP", "TEMP", "TMPDIR")
                _backup = {k: _os.environ.get(k) for k in _tmp_keys}

                # get the absolute path and ensure the directory exists
                output_path = _os.path.abspath(output_path)
                output_dir = _os.path.dirname(output_path)
                _os.makedirs(output_dir, exist_ok=True)

                # create a temp directory on the same drive as output
                temp_dir = _os.path.join(output_dir, '.tmp_osmium')
                _os.makedirs(temp_dir, exist_ok=True)

                try:
                    # set all temp environment variables to our temp directory
                    for k in _tmp_keys:
                        _os.environ[k] = temp_dir

                    # also set tempfile's temp directory
                    tempfile.tempdir = temp_dir

                    writer = osm.SimpleWriter(output_path)
                    try:
                        handler = MergeHandler(writer)
                        for p in input_paths:
                            print(f"Processing {p} ...")
                            handler.apply_file(p, locations=True)
                    finally:
                        # always close the writer so tmp file is finalized/removed
                        try:
                            writer.close()
                        except Exception as e:
                            print(f"Warning: Error closing writer: {e}")
                finally:
                    # restore TMP/TEMP/TMPDIR
                    for k, v in _backup.items():
                        if v is None:
                            _os.environ.pop(k, None)
                        else:
                            _os.environ[k] = v

                    # restore tempfile's temp directory
                    tempfile.tempdir = None

                    # clean up our temp directory
                    try:
                        if _os.path.exists(temp_dir):
                            import shutil
                            shutil.rmtree(temp_dir, ignore_errors=True)
                    except Exception as e:
                        print(f"Warning: Could not clean up temp directory: {e}")

            merge_pbf_files(state_pbf_filepaths, merged_pbf_filepath)

        elif len(self.states_to_include) == 1:
            logging.info("Only one state, copying PBF file to merged location")
            single_state_filepath = list(state_to_geofabrik_pbf_filename.values())[0]

            # Use copy instead of symlink (Windows symlinks require admin privileges)
            if not os.path.exists(merged_pbf_filepath) or reset:
                shutil.copyfile(single_state_filepath, merged_pbf_filepath)
                logging.info(f"Copied {single_state_filepath} to {merged_pbf_filepath}")
        else:
            raise ValueError("No states to include - cannot create merged PBF file")

        # now set the merged pbf filepath as an attribute
        self.merged_pbf_filepath = merged_pbf_filepath
        return merged_pbf_filepath
        # go through each state to include and download the pbf data for it

class TransitNetwork:
    def __init__(self, geographic_scope, reference_place_list: list[ReferencePlace],
                 modes: list = None, agencies_to_include: list[TransitAgency] = None,
                 own_gtfs_data = dict[TransitAgency, str] | None):
        """
        Transit network class for place

        Attributes:
            geographic_scope: GeographicScope | geographic scope of the transit network {"place_only", "msa", "csa"}
            reference_place_list: list[ReferencePlace] | list of reference places for the transit network
            modes: list | modes to be included in the transit network {"all", "bus", "heavy_rail", "light_rail",
            "regional_rail", "ferry", "gondola", "funicular", "trolleybus", "monorail"}
            (for more, see gtfs_tools.route_types documentation)

        **Methods:**
            get_transit_agencies_for_place: creates a list of transit agencies that serve place (see method doc)


        """
        self.geographic_scope = geographic_scope
        self.reference_place_list = reference_place_list

        self.place_names = [reference_place.place_name for reference_place in reference_place_list]
        self.bound_box = [reference_place.bound_box for reference_place in reference_place_list]
        self.main_reference_place = reference_place_list[0]

        if self.main_reference_place.bound_box:
            self.geographic_scope = "bbox"

        self.snake_name = create_snake_name(self.main_reference_place)
        self.snake_name_with_scope = f"{self.snake_name}_{self.geographic_scope}"
        self.modes = modes

        # eventually this is what will be used to pass gtfs data to create network
        self.gtfs_folders = None
        self.agency_feed_valid_dates = {}

        # link to cache folder
        self.cache_folder = CacheFolder(self.snake_name_with_scope)

        # if no agencies are specified, will default to all agencies that serve the place
        self.agencies_to_include = agencies_to_include
        if self.agencies_to_include is None:
            logging.info("a list of transit agencies to include was not specified so all agencies that serve the place"
                         " will be used")
            self.agencies_to_include = self.get_agencies_for_place()

        # adding the ability to bring your own gtfs data
        self.own_gtfs_data = own_gtfs_data

        # check whether own gtfs provided, if not, fetch it
        if self.own_gtfs_data is None:
            logging.info(f"Own GTFS data not provided, will query TransitLand API to get transit agencies that serve"
                         f"{self.main_reference_place.pretty_name}")
            self.get_agencies_for_place()
            self.gtfs_zip_files = self.get_gtfs_for_transit_agencies()
            self.gtfs_zip_files = self.repackage_gtfs_zips()
            self.gtfs_zip_files = self.clean_gtfs_zips()
            self.check_whether_data_valid()

        else:
            logging.info("Using own GTFS data provided by user")
            self.gtfs_zip_files = self.own_gtfs_data
            self.check_whether_data_valid()
            self.gtfs_zip_files = self.repackage_gtfs_zips()
            self.gtfs_zip_files = self.clean_gtfs_zips()

    def clean_r5py_cache(self):
        """
        Cleans up r5py cache to avoid permission and stale file issues.
        Completely removes and recreates the cache directory.
        """
        import shutil
        r5py_cache = Path.home() / 'AppData' / 'Local' / 'r5py'

        if r5py_cache.exists():
            try:
                logging.info("Completely cleaning r5py cache directory")
                # Remove entire cache directory to clear corrupted files
                shutil.rmtree(r5py_cache, ignore_errors=True)
                # Recreate it fresh
                r5py_cache.mkdir(parents=True, exist_ok=True)
                logging.info("Successfully cleaned and recreated r5py cache")
            except Exception as e:
                logging.warning(f"Could not clean r5py cache: {e}")


    # using requests instead of aihttp for this because only single request
    def get_agencies_for_place(self):
        """
        Takes a place name of format 'city, state, country'
        and returns a list of transit agencies that serve the place

        :return: list[TransitAgency] | list of transit agencies (TransitAgencyObjects) that serve the place
        """
        logging.info(f"Getting agencies that serve {self.main_reference_place.pretty_name}")
        # list of TransitAgency objects to be returned
        agencies_for_place = []

        # in case where using place (with geographic scope) rather than bounding box
        if self.main_reference_place.bound_box is None:

            # iterate through reference places to get the agencies that serve them
            for reference_place in self.reference_place_list:
                # transit land's API only requires the city name (although this seems stupid)
                place_short_name = reference_place.place_name.split(",")[0]

                transit_land_response = requests.get(f"https://transit.land/api/v2/rest/agencies?api_key="
                                                     f"{transit_land_api_key}"
                                                     f"&city_name={place_short_name}")
                transit_land_response.raise_for_status()

                # json containing the agencies
                transit_land_data = transit_land_response.json()

                # going through the agency dicts provided by the api and using them as kwargs for TransitAgency object
                for agency_data in transit_land_data["agencies"]:
                    # fill out TransitAgency objects using the data from the API
                    temp_agency = TransitAgency(**agency_data)
                    agencies_for_place.append(temp_agency)
                logging.info(f"Found {len(agencies_for_place)} agencies that serve {reference_place.pretty_name}"
                             f" {self.geographic_scope}")

        # in case where using bounding box
        else:
            # the bbox is concatenated into a string to be used in the query to transitland's API
            bbox_query_string = ",".join(self.main_reference_place.bound_box)
            transit_land_response = requests.get(f"https://transit.land/api/v2/rest/agencies?api_key="
                                                 f"{transit_land_api_key}"
                                                 f"&bbox={bbox_query_string}")
            transit_land_response.raise_for_status()
            transit_land_data = transit_land_response.json()

            for agency_data in transit_land_data["agencies"]:
                # fill out TransitAgency objects using the data from the API
                temp_agency = TransitAgency(**agency_data)
                agencies_for_place.append(temp_agency)

            logging.info(f"Found {len(agencies_for_place)} agencies that serve {self.main_reference_place.pretty_name}")

        # now can set self.agencies_that_serve_place
        return agencies_for_place

    def get_gtfs_for_transit_agencies(self):
        """
            Gets the latest static GTFS data for agencies desired.

            :param agencies_to_include: list[TransitAgency] | list of transit agencies to get GTFS data for
                (by default will be all agencies that serve the reference place)
            :return: gtfs_zip_folders | dict with TransitAgencies as keys and
             the (zipped) file names where the gtfs feeds are written as values.
            :return: agency_feed_valid_dates: dict{TransitAgency: {"last_updated": MMDDYYYY, "valid_until": MMDDYYYY}}
                | dictionary of the valid dates for each agency's feed and when it was last updated (because will have
                to deal with feeds that are not current or currently valid.
        """
        # if a list of agencies to include was not provided then by default will use every transit agency serving place
        logging.info(f"Getting GTFS data for {len(self.agencies_to_include)} transit agencies")
        # outputs for the method
        gtfs_zip_folders = {}

        # next, iterating through the list of desired agencies and getting their data
        for agency in self.agencies_to_include:
            # onestop_id (used to query for feed) is in feed_version["feed"]["onestop_id"]
            feed_version = agency.feed_version
            onestop_id = feed_version["feed"]["onestop_id"]
            transit_land_api_url = (f"https://transit.land/api/v2/rest/feeds/{onestop_id}/download_latest_feed_version"
                                    f"?api_key={transit_land_api_key}")

            # standard API query
            response = requests.get(transit_land_api_url)
            response.raise_for_status()

            # the file path where the zipped gtfs data will be saved to (yes complicated, but best for organization
            file_path = os.path.join(self.cache_folder.path, "gtfs_caches", f"{onestop_id}", "zipped_gtfs",
                                     f"{onestop_id}.zip")

            # set up folders in case they don't already exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            # writing the content that was returned from transit land to a zip file
            with open(file_path, "wb") as gtfs_zipped_file:
                gtfs_zipped_file.write(response.content)

            # now need to see 1. when the feed was fetched (and that it's not too out of date) and 2. whether it's valid
            feed_version_query_response = requests.get(f"https://transit.land/api/v2/rest/feeds/{onestop_id}"
                                                       f"?api_key={transit_land_api_key}")
            feed_version_query_response.raise_for_status()
            feed_version_query_response_json = feed_version_query_response.json()

            # when the latest static feed was fetched by transit land
            latest_feed_fetch_date = (feed_version_query_response_json["feeds"][0]["feed_versions"]
            [0]["fetched_at"])
            # the latest date in the calendar that the data is valid for (can either extend or not use)
            latest_feed_valid_until = (feed_version_query_response_json["feeds"][0]["feed_versions"]
            [0]["latest_calendar_date"])

            # adding the zipped gtfs folder to the path dict
            gtfs_zip_folders[agency] = file_path
            # creating a dictionary with these dates needed for each agency
            self.agency_feed_valid_dates[agency] = {"last_updated": latest_feed_fetch_date,
                                                    "valid_until": latest_feed_valid_until}

        logging.info("Successfully downloaded GTFS data for desired agencies")
        return gtfs_zip_folders

    def repackage_gtfs_zips(self):
        """
        Repackages GTFS zip files to ensure all .txt files are at the root level,
        not nested in subdirectories, and removes empty optional files that cause R5 errors.
        This is required by r5py/R5.

        :return: dict{TransitAgency: path to repackaged zip file}
        """
        import tempfile
        from zipfile import ZipFile, ZIP_DEFLATED

        logging.info("Repackaging GTFS zip files to ensure proper structure for r5py")
        repackaged_zip_files = {}

        # Required GTFS files that must not be removed even if empty
        required_files = {
            'agency.txt', 'stops.txt', 'routes.txt',
            'trips.txt', 'stop_times.txt'
        }

        for agency, zip_path in self.gtfs_zip_files.items():
            onestop_id = agency.feed_version["feed"]["onestop_id"]

            # Check if repackaging is needed by inspecting the zip
            needs_repackaging = False
            has_empty_files = False

            with ZipFile(zip_path, 'r') as original_zip:
                # Check if any GTFS files are in subdirectories or empty
                for filename in original_zip.namelist():
                    if filename.endswith('.txt'):
                        if '/' in filename:
                            needs_repackaging = True
                            logging.debug(f"  {onestop_id}: Found nested file {filename}, needs repackaging")

                        # Check if file is empty (optional files only)
                        file_size = original_zip.getinfo(filename).file_size
                        base_filename = os.path.basename(filename)
                        if file_size == 0 or file_size < 10:  # Less than 10 bytes is effectively empty
                            if base_filename not in required_files:
                                has_empty_files = True
                                logging.debug(f"  {onestop_id}: Found empty optional file {filename}")

            if not needs_repackaging and not has_empty_files:
                logging.debug(f"  {onestop_id}: Already has correct structure")
                repackaged_zip_files[agency] = zip_path
                continue

            # Create new zip with files at root level and without empty optional files
            logging.info(f"  Repackaging {onestop_id}")
            new_zip_path = zip_path.replace('.zip', '_repackaged.zip')

            with ZipFile(zip_path, 'r') as original_zip:
                with ZipFile(new_zip_path, 'w', ZIP_DEFLATED) as new_zip:
                    for file_info in original_zip.filelist:
                        # Only copy .txt files (GTFS data files)
                        if file_info.filename.endswith('.txt'):
                            # Extract just the filename without any directory path
                            base_filename = os.path.basename(file_info.filename)

                            # Read the file content
                            file_content = original_zip.read(file_info.filename)

                            # Skip empty optional files
                            if len(file_content) < 10 and base_filename not in required_files:
                                logging.debug(f"    Skipping empty optional file: {file_info.filename}")
                                continue

                            # Write to new zip at root level
                            new_zip.writestr(base_filename, file_content)
                            if file_info.filename != base_filename:
                                logging.debug(f"    Moved {file_info.filename} -> {base_filename}")
                            else:
                                logging.debug(f"    Kept {base_filename}")

            repackaged_zip_files[agency] = new_zip_path
            self.gtfs_zip_files = repackaged_zip_files

        logging.info("Successfully repackaged GTFS zip files")
        return repackaged_zip_files

    def clean_gtfs_zips(self):
        """
        Cleans up the gtfs zip files so that they do not contain any files which are there in name only, but
        do not have any data, as r5py is very sensitive to this and will throw errors. 
        :return: cleaned_gtfs_zip_filepaths: dict{TransitAgency: path of cleaned gtfs zip file}
        """
        logging.info("Cleaning up the gtfs zip files")

        # to avoid modifying dict while iterating
        prepared_paths = []
        # to keep track of which files were excluded from the cleaned zip files
        excluded_files = {agency: [] for agency in self.gtfs_zip_files.keys()}

        # IMPORTANT: where the updated gtfs_paths will go
        cleaned_gtfs_zip_filepaths = {}

        
        # the set of files that must not be removed no matter what
        required_files = {"agency.txt", "stops.txt", "routes.txt", "trips.txt", "stop_times.txt"}
        
        # go through each ageancy to determine whether to include at all
        for agency, agency_zip_file in self.gtfs_zip_files.items():
            # needs to be cleaned flag
            agency_needs_cleaning = False
            # the files which need to be removed for this agency
            files_to_remove = []


            # safe open the existing zip file
            with ZipFile(agency_zip_file, "r") as original_zip_file:
                # go through each file in the zip file to see if any are empty
                for gtfs_file in original_zip_file.filelist:

                    # check whether file is a txt file (gtfs data files)
                    if gtfs_file.filename.endswith(".txt"):
                        # get the base name of the file out
                        base_filename = os.path.basename(gtfs_file.filename)

                    # check whether file is empty (or effectively empty (less than 50 bytes))
                    if gtfs_file.file_size < 50 and base_filename not in required_files:
                        # if so, need to exclude it from the cleaned zip file
                        agency_needs_cleaning = True
                        # add it to the list of files to remove
                        files_to_remove.append(base_filename)

            # if agency doesn't need cleaning, just add the original path
            if not agency_needs_cleaning:
                logging.debug(f"Agency {agency.agency_name} GTFS zip file does not need cleaning")
                cleaned_gtfs_zip_filepaths[agency] = agency_zip_file

            # otherwise, need to clean the gtfs zip file
            else:
                logging.info(f"Cleaning GTFS zip file for agency {agency.agency_name}")

                # the path for the cleaned zip file
                cleaned_zip_file_path = agency_zip_file.replace(".zip", "_cleaned.zip")

                # if already exists, delete it
                if os.path.exists(cleaned_zip_file_path):
                    os.remove(cleaned_zip_file_path)

                # create a new zip file without the empty files
                with ZipFile(agency_zip_file, "r") as original_zip_file:
                    # creating the new cleaned zip file
                    with ZipFile(cleaned_zip_file_path, 'w', ZIP_DEFLATED) as cleaned_zip_file:
                        # go through each file in the original zip file
                        for gtfs_file in original_zip_file.filelist:
                            base_filename = os.path.basename(gtfs_file.filename)

                            # if the file is not in the files to remove list, add it to the cleaned zip file
                            if base_filename not in files_to_remove:
                                # read the file data
                                file_data = original_zip_file.read(gtfs_file.filename)
                                # write it to the cleaned zip file
                                cleaned_zip_file.writestr(base_filename, file_data)

                            else:
                                # log that the file was excluded
                                excluded_files[agency].append(base_filename)
                                logging.debug(f"  Excluded empty file {base_filename} from agency "
                                              f"{agency.agency_name} GTFS zip")

                # add the cleaned zip file path to the output dict
                cleaned_gtfs_zip_filepaths[agency] = cleaned_zip_file_path
                logging.info(f"Cleaned GTFS zip file {cleaned_zip_file_path} for agency"
                             f" {agency.agency_name}, excluded files: {excluded_files[agency]}")


        logging.debug(f"Cleaned zipfiles details: {excluded_files}")
        logging.info(f"Successfully cleaned {len(cleaned_gtfs_zip_filepaths)}GTFS zip files")
        # set the gtfs_zip_files attribute and return the cleaned zip file paths
        self.gtfs_zip_files = cleaned_gtfs_zip_filepaths
        return cleaned_gtfs_zip_filepaths




            
            

    # this method is vestigial from arcpy version, but keeping it for now
    def unzip_gtfs_data(self):
        """
        Unzips the GTFS data that was downloaded from Transit Land.
        :return: unzipped_gtfs_filepaths: dict{TransitAgency: path of unzipped gtfs folder}
        """
        agency_zip_folders = self.gtfs_zip_files
        logging.info("Unzipping the downloaded GTFS data")
        unzipped_gtfs_filepaths = {}

        # go through each agency in the provided agency_zip_folders
        for agency in agency_zip_folders:
            # Get the onestop_id for unique naming
            onestop_id = agency.feed_version["feed"]["onestop_id"]

            # the path where each unzipped gtfs folder will be saved to
            onestop_id_directory = os.path.dirname(os.path.dirname(agency_zip_folders[agency]))

            # Use a unique name for each agency instead of generic "unzipped_gtfs"
            unzipped_gtfs_filepath = os.path.join(onestop_id_directory,
                                                  f"unzipped_gtfs")  # Changed this line

            # set up folders in case they don't already exist
            os.makedirs(unzipped_gtfs_filepath, exist_ok=True)

            # using zipfile module to extract all .txt files provided
            with ZipFile(agency_zip_folders[agency], "r") as zipped_file:
                zipped_file.extractall(path=unzipped_gtfs_filepath)

            # adding the unzipped gtfs folder to the path dict
            unzipped_gtfs_filepaths[agency] = unzipped_gtfs_filepath

        # set the gtfs_folders attribute and return the unzipped folder paths
        self.gtfs_folders = unzipped_gtfs_filepaths
        logging.info("Successfully unzipped the downloaded GTFS data")
        return unzipped_gtfs_filepaths

    def check_whether_data_valid(self):
        """
        Checks whether the data is valid for the desired agencies.
        :return:
        """
        logging.info("Checking whether the downloaded data is still valid (and can therefore be used to create a "
                     "network dataset in ArcGIS)")

        # a dictionary that says for each agency if the downloaded
        still_valid_gtfs_data = {}
        for agency in self.agency_feed_valid_dates:
            # check if current date is past the "valid_until" date
            valid_until_date_iso = isodate.parse_date(self.agency_feed_valid_dates[agency]["valid_until"])
            current_date_iso = isodate.parse_date(datetime.now().strftime("%Y-%m-%d"))

            if valid_until_date_iso <= current_date_iso:
                logging.warning(
                    f"The data for {agency.agency_name} is no longer valid (valid until {valid_until_date_iso})")
            else:

                still_valid_gtfs_data[agency] = self.gtfs_folders[agency]

        # don't want to use invalid data so discarding invalid data
        self.gtfs_folders = still_valid_gtfs_data

        logging.info(f"Data was valid for {len(still_valid_gtfs_data)}/{len(self.agency_feed_valid_dates)} agencies")
        return still_valid_gtfs_data


# because switching to using r5py need to compose everything into one graph/network]
class R5Network:
    def __init__(self, street_network: StreetNetwork, transit_network: TransitNetwork|None=None,
                 network_type:str="transit"):
        """
        R5Network class for creating R5 instance using street network and transit network GTFS data.
        """
        self.street_network = street_network
        self.transit_network = transit_network
        self.network_type = network_type

        # clean r5py cache before setting up network to avoid permission and stale file issues
        if self.transit_network is not None:
            self.transit_network.clean_r5py_cache()

        # where the r5 network will go
        self.transportation_network = self.setup_r5_network()

        # setting up transport modes
        self.transport_modes = []

        # look up network type in network_types module and get r5py mode type
        # using getattr so can dynamically get the transport mode from r5py.TransportMode
        r5py_mode_types = [getattr(r5py.TransportMode, mode) for mode in network_types.r5modes[self.network_type]]

        self.transport_modes = r5py_mode_types

    @timer
    def setup_r5_network(self):
        """
        Sets up R5 network using street network and transit network GTFS data.
        Automatically handles and skips GTFS files with data integrity errors.
        :return: r5py.R5Network instance
        """
        logging.info("Setting up R5 network")

        # parameters for R5 network
        input_osm_pbf = self.street_network.merged_pbf_filepath
        if self.transit_network is not None:
            input_gtfs_paths = list(self.transit_network.gtfs_zip_files.values())
            original_count = len(input_gtfs_paths)

            logging.info(f"Using {len(input_gtfs_paths)} GTFS zip files for R5 network")
            for path in input_gtfs_paths:
                logging.debug(f"  GTFS zip: {path}")
        else:
            input_gtfs_paths = None
            original_count = 0

        # r5 transportation network
        if self.transit_network is None:
            r5_transportation_network = r5py.TransportNetwork(osm_pbf=input_osm_pbf)
        else:
            max_retries = len(input_gtfs_paths)  # At most, we'll remove all files
            retry_count = 0

            while retry_count < max_retries:
                try:
                    r5_transportation_network = r5py.TransportNetwork(osm_pbf=input_osm_pbf, gtfs=input_gtfs_paths)

                    # Success!
                    if retry_count > 0:
                        skipped = original_count - len(input_gtfs_paths)
                        logging.info(
                            f"Successfully created network with {len(input_gtfs_paths)} GTFS files ({skipped} skipped due to data errors)")
                    break

                except r5py.util.exceptions.GtfsFileError as e:
                    retry_count += 1
                    error_msg = str(e)
                    logging.error(f"GTFS validation error (attempt {retry_count}): {error_msg[:200]}")

                    # Try to identify which file is problematic
                    if "Could not load GTFS file" in error_msg:
                        # Extract filename from error
                        import re
                        match = re.search(r"Could not load GTFS file ([^\s\.]+)", error_msg)
                        if match:
                            problematic_file = match.group(1)
                            logging.warning(f"Identified problematic GTFS file: {problematic_file}")

                            # Remove the problematic file
                            before_count = len(input_gtfs_paths)
                            input_gtfs_paths = [p for p in input_gtfs_paths if problematic_file not in str(p)]
                            after_count = len(input_gtfs_paths)

                            if after_count < before_count:
                                logging.warning(
                                    f"Removed problematic GTFS file. Retrying with {after_count}/{original_count} GTFS files")
                            else:
                                logging.error("Could not identify which GTFS file to remove from the list")
                                raise

                            # Check if we still have GTFS files left
                            if len(input_gtfs_paths) == 0:
                                logging.error("All GTFS files have been removed due to errors. Cannot create network.")
                                raise ValueError("No valid GTFS files available")
                        else:
                            logging.error("Could not parse error message to identify problematic file")
                            raise
                    else:
                        raise

        logging.info("Successfully set up R5 network")
        return r5_transportation_network

    
# because using multiprocessing, these functions must be in top scope
def calculate_itineraries_with_chunks(r5_transport_network, r5_transport_modes, origins_gdf, destinations_gdf,
                                      itinerary_departure_time, snap_to_street=True):
    """ Base function to calculate an OD matrix with routes using R5Py for a chunk of origins/destinations"""
    logging.info("Calculating OD matrix with routes")

    # need id column in origins gdf
    if "id" not in origins_gdf.columns:
        origins_gdf = origins_gdf.copy()
        origins_gdf["id"] = range(len(origins_gdf))

    # need id column in destinations gdf as well
    if "id" not in destinations_gdf.columns:
        destinations_gdf = destinations_gdf.copy()
        destinations_gdf["id"] = range(len(destinations_gdf))

    # detailed itineraries so can keep track of which route it uses
    detailed_itineraries = r5py.DetailedItineraries(transport_network=r5_transport_network,
                                                    origins=origins_gdf,
                                                    destinations=destinations_gdf,
                                                    departure=itinerary_departure_time,
                                                    transport_modes=r5_transport_modes,
                                                    force_all_to_all=True)

    return detailed_itineraries


def calculate_travel_times_with_chunks(r5_transport_network, r5_transport_modes, origins_gdf, destinations_gdf,
                                      itinerary_departure_time, snap_to_street=True):
    """ Base function to calculate an OD matrix with routes using R5Py for a chunk of origins/destinations"""
    logging.info("Calculating OD matrix with routes")

    # need id column in origins gdf
    if "id" not in origins_gdf.columns:
        origins_gdf = origins_gdf.copy()
        origins_gdf["id"] = range(len(origins_gdf))

    # need id column in destinations gdf as well
    if "id" not in destinations_gdf.columns:
        destinations_gdf = destinations_gdf.copy()
        destinations_gdf["id"] = range(len(destinations_gdf))

    # travel time matrix just gives shortest path
    travel_time_matrix = r5py.TravelTimeMatrix(transport_network=r5_transport_network,
                                                    origins=origins_gdf,
                                                    destinations=destinations_gdf,
                                                    departure=itinerary_departure_time,
                                                    transport_modes=r5_transport_modes,
                                                    snap_to_network=snap_to_street,
                                                    force_all_to_all=True)
    return travel_time_matrix


def calculate_ideal_od_chunk_size(origins_count:int, destinations_count:int) -> tuple[str, int, int]:
    """
    Calculates ideal chunk size for multiprocessing based on number of origins and destinations and threads
    available
    :param origins_count: int | number of origins
    :param destinations_count: int | number of destinations
    :return: tuple[str, int, int] | tuple containing origin_or_destination (str), number_of_chunks (int),
     ideal_chunk_size (int)
    """
    logging.info("Calculating ideal OD chunk size for multiprocessing")
    # getting number of available threads
    available_threads = mp.cpu_count()

    # calculating total number of origin-destination pairs
    total_od_pairs = origins_count * destinations_count

    # want to break up whichever is larger, origins or destinations
    if origins_count > destinations_count:
        logging.info("Origins count is greater than destinations count")
        origin_or_destination = "origin"

        # ideal chunk size is total od pairs divided by number of threads
        number_of_origin_chunks = math.ceil(origins_count / available_threads)

        # the number of origins to include in each chunk
        ideal_origin_chunk_size = math.ceil(origins_count / number_of_origin_chunks)

        logging.info(f"Ideal number of origin chunks: {number_of_origin_chunks}, "
                     f"ideal origin chunk size: {ideal_origin_chunk_size}")
        return origin_or_destination, number_of_origin_chunks, ideal_origin_chunk_size

    elif origins_count <= destinations_count:
        logging.info("Destinations count is greater than or equal to origins count")
        origin_or_destination = "destination"
        # ideal chunk size is total od pairs divided by number of threads
        number_of_destination_chunks = math.ceil(destinations_count / available_threads)

        # the number of destinations to include in each chunk
        ideal_destination_chunk_size = math.ceil(destinations_count / number_of_destination_chunks)

        logging.info(f"Ideal number of destination chunks: {number_of_destination_chunks}, "
                     f"ideal destination chunk size: {ideal_destination_chunk_size}")
        return origin_or_destination, number_of_destination_chunks, ideal_destination_chunk_size



def split_gdfs_into_chunks(origins_gdf:gpd.GeoDataFrame, destinations_gdf:gpd.GeoDataFrame,
                           ideal_od_chunks:tuple[str, int, int]):
    """
    Splits origins or destinations geodataframe into chunks for multiprocessing based on ideal chunk size
    :param origins_gdf: gpd.GeoDataFrame | geodataframe of origins
    :param destinations_gdf: gpd.GeoDataFrame | geodataframe of destinations
    :param ideal_od_chunks: tuple[str, int, int] | tuple containing origin_or_destination (str), number_of_chunks (int),
     ideal_chunk_size (int)
    :return: dict{"origins": [origins gdf(s)], "destinations": [destination gdf(s)] | dict of geodataframe chunks
    """
    logging.info("Splitting geodataframe into chunks for multiprocessing")
    origin_or_destination, number_of_chunks, ideal_chunk_size = ideal_od_chunks

    # lists of gdfs to be returned
    output_origin_gdfs = []
    output_destinations_gdfs = []

    # if origins was bigger, split origins gdfs
    if origin_or_destination == "origin":
        logging.info("Splitting origins geodataframe into chunks")

        # go through each chunk and create a gdf for it
        for chunk_idx in range(number_of_chunks):
            # start aned end indices for chunk
            start_index = chunk_idx * ideal_chunk_size
            end_index = min((chunk_idx + 1) * ideal_chunk_size, len(origins_gdf))

            # the chunk gdf
            origin_chunk_gdf = origins_gdf.iloc[start_index:end_index]
            # add to list of output gdfs
            output_origin_gdfs.append(origin_chunk_gdf)

        # set the output destinations gdfs to be the same since chunking origins
        output_destinations_gdfs.append(destinations_gdf)
        return {"origins": output_origin_gdfs, "destinations": output_destinations_gdfs}

    # if destinations was bigger, split destinations gdfs
    elif origin_or_destination == "destination":
        logging.info("Splitting destinations geodataframe into chunks")

        # go through each chunk and create a gdf for it
        for chunk_idx in range(number_of_chunks):
            # start and end indices for chunk
            start_index = chunk_idx * ideal_chunk_size
            end_index = min((chunk_idx + 1) * ideal_chunk_size, len(origins_gdf))

            # the chunk gdf
            destination_chunk_gdf = origins_gdf.iloc[start_index:end_index]
            # add to list of output gdfs
            output_destinations_gdfs.append(destination_chunk_gdf)

        # set the output destinations gdfs to be the same since chunking origins
        output_origin_gdfs.append(origins_gdf)
        return {"origins": output_origin_gdfs, "destinations": output_destinations_gdfs}
    else:
        raise ValueError("origin_or_destination must be either 'origin' or 'destination'")

# process the itinerary departure time into proper format
class DetailedItineraries:
    def __init__(self, analysis_name:str, r5_network:R5Network, origins_geojson_path:str, destinations_geojson_path:str,
                 date:str, departure_time:str):
        self.analysis_name = analysis_name
        self.r5_network = r5_network
        self.origins_geojson_path = origins_geojson_path
        self.destinations_geojson_path = destinations_geojson_path
        self.date = date
        self.departure_time = departure_time

        # cache_folder
        self.cache_folder = os.path.join(r5_network.street_network.cache_folder.path, "detailed_itineraries",
                                         analysis_name)
        # make cache folder if doesn't exist
        os.makedirs(self.cache_folder, exist_ok=True)

        # format the departure time correctly
        self.itinerary_departure_time = datetime.strptime(f"{date} {departure_time}", "%m/%d/%Y %H:%M")

        # read in origins and destinations gdfs
        self.origins_gdf = gpd.read_file(self.origins_geojson_path)
        self.destinations_gdf = gpd.read_file(self.destinations_geojson_path)

        # number of processes to use
        self.num_processes = None


    @timer
    def calculate_detailed_itineraries(self):
        """
        Calculates detailed itineraries using R5py for all origin-destination pairs
        :return:
        """
        logging.info("Calculating detailed itineraries")

        # count of origins and destinations
        origins_count = len(self.origins_gdf)
        destinations_count = len(self.destinations_gdf)
        od_pairs_count = origins_count * destinations_count
        logging.info(f"Number of origins: {origins_count}, number of destinations: {destinations_count}"
                     f"; total OD pairs: {od_pairs_count}")

        # calculating ideal chunk size
        ideal_od_chunks = calculate_ideal_od_chunk_size(origins_count, destinations_count)
        origin_or_destination, number_of_chunks, ideal_chunk_size = ideal_od_chunks

        # splitting gdfs into chunks
        gdf_chunks = split_gdfs_into_chunks(self.origins_gdf, self.destinations_gdf, ideal_od_chunks)

        # origin and destination chunked gdfs
        origin_chunked_gdfs = gdf_chunks["origins"]
        destination_chunked_gdfs = gdf_chunks["destinations"]

        # setting up multiprocessing pool
        process_args = []

        # if chunking origins
        if origin_or_destination == "origin":
            for chunk_idx, origin_chunk in enumerate(origin_chunked_gdfs):

                # the arguments for the caclulate_od_with_routes_chunk function
                process_args.append((self.r5_network.transportation_network,
                                     self.r5_network.transport_modes,
                                     origin_chunk,
                                     destination_chunked_gdfs[0],
                                     self.itinerary_departure_time,
                                     True))


        # if chunking destinations insterd
        if origin_or_destination == "destination":
            for chunk_idx, destination_chunk in enumerate(destination_chunked_gdfs):

                # the arguments for the caclulate_od_with_routes_chunk function
                process_args.append((self.r5_network.transportation_network,
                                     self.r5_network.transport_modes,
                                     origin_chunked_gdfs[0],
                                     destination_chunk,
                                     self.itinerary_departure_time,
                                     True))

        # the number of processes to use (len(process_args) should always be <= number of cpus)
        num_processes = min(mp.cpu_count(), len(process_args))
        self.num_processes = num_processes
        logging.info(f"Using {num_processes} processes for multiprocessing")

        try:
            with mp.pool.ThreadPool(processes=num_processes) as pool:
                # using starmap to pass multiple arguments to the function
                chunk_results = pool.starmap(calculate_itineraries_with_chunks, process_args)
                # manually closing pool
                pool.close()
                pool.join()

            # combine the results from all chunks
            logging.info("Combining results from all chunks")
            all_results = []

            # go through each chunk result and add to combined results
            for chunk_idx, result in enumerate(chunk_results):
                if result is not None and len(result) > 0:
                    logging.info(f"Chunk {chunk_idx}: {len(result)} itineraries")
                    all_results.append(result)

                # idiot proofing
                else:
                    logging.warning(f"Chunk {chunk_idx}: No results returned")

            logging.info("Successfully combined results from all chunks")

            # check that all_results is not empty
            if all_results:
                combined_results_df = pd.concat(all_results, ignore_index=True)
                logging.info(f"Successfully calculated detailed itineraries for all origin-destination pairs"
                             f"; total itineraries: {len(combined_results_df)}")

                # writing combined results to cache folder
                output_filepath = os.path.join(self.cache_folder, f"{self.analysis_name}_detailed_itineraries.csv")
                combined_results_df.to_csv(output_filepath, index=False)
                logging.info(f"Written detailed itineraries to {output_filepath}")

                return combined_results_df

            else:
                logging.error("No results were returned from any chunk, cannot combine results")
                return None
        # in case of problems when running
        except Exception as e:
            logging.error(f"Error occurred during multiprocessing: {e}")
            raise

class TravelTimeMatrix:
    def __init__(self, analysis_name: str, r5_network: R5Network, origins_geojson_path: str,
                 destinations_geojson_path: str,
                 date: str, departure_time: str):
        self.analysis_name = analysis_name
        self.r5_network = r5_network
        self.origins_geojson_path = origins_geojson_path
        self.destinations_geojson_path = destinations_geojson_path
        self.date = date
        self.departure_time = departure_time

        # cache_folder
        self.cache_folder = os.path.join(r5_network.street_network.cache_folder.path, "detailed_itineraries",
                                         analysis_name)
        # make cache folder if doesn't exist
        os.makedirs(self.cache_folder, exist_ok=True)

        # format the departure time correctly
        self.itinerary_departure_time = datetime.strptime(f"{date} {departure_time}", "%m/%d/%Y %H:%M")

        # read in origins and destinations gdfs
        self.origins_gdf = gpd.read_file(self.origins_geojson_path)
        self.destinations_gdf = gpd.read_file(self.destinations_geojson_path)

        # number of processes to use
        self.num_processes = None

    @timer
    def calculate_detailed_itineraries(self):
        """
        Calculates detailed itineraries using R5py for all origin-destination pairs
        :return:
        """
        logging.info("Calculating detailed itineraries")

        # count of origins and destinations
        origins_count = len(self.origins_gdf)
        destinations_count = len(self.destinations_gdf)
        od_pairs_count = origins_count * destinations_count
        logging.info(f"Number of origins: {origins_count}, number of destinations: {destinations_count}"
                     f"; total OD pairs: {od_pairs_count}")

        # calculating ideal chunk size
        ideal_od_chunks = calculate_ideal_od_chunk_size(origins_count, destinations_count)
        origin_or_destination, number_of_chunks, ideal_chunk_size = ideal_od_chunks

        # splitting gdfs into chunks
        gdf_chunks = split_gdfs_into_chunks(self.origins_gdf, self.destinations_gdf, ideal_od_chunks)

        # origin and destination chunked gdfs
        origin_chunked_gdfs = gdf_chunks["origins"]
        destination_chunked_gdfs = gdf_chunks["destinations"]

        # setting up multiprocessing pool
        process_args = []

        # if chunking origins
        if origin_or_destination == "origin":
            for chunk_idx, origin_chunk in enumerate(origin_chunked_gdfs):
                # the arguments for the caclulate_od_with_routes_chunk function
                process_args.append((self.r5_network.transportation_network,
                                     self.r5_network.transport_modes,
                                     origin_chunk,
                                     destination_chunked_gdfs[0],
                                     self.itinerary_departure_time,
                                     True))

        # if chunking destinations insterd
        if origin_or_destination == "destination":
            for chunk_idx, destination_chunk in enumerate(destination_chunked_gdfs):
                # the arguments for the caclulate_od_with_routes_chunk function
                process_args.append((self.r5_network.transportation_network,
                                     self.r5_network.transport_modes,
                                     origin_chunked_gdfs[0],
                                     destination_chunk,
                                     self.itinerary_departure_time,
                                     True))

        # the number of processes to use (len(process_args) should always be <= number of cpus)
        num_processes = min(mp.cpu_count(), len(process_args))
        self.num_processes = num_processes
        logging.info(f"Using {num_processes} processes for multiprocessing")

        try:
            with mp.pool.ThreadPool(processes=num_processes) as pool:
                # using starmap to pass multiple arguments to the function
                chunk_results = pool.starmap(calculate_itineraries_with_chunks, process_args)
                # manually closing pool
                pool.close()
                pool.join()

            # combine the results from all chunks
            logging.info("Combining results from all chunks")
            all_results = []

            # go through each chunk result and add to combined results
            for chunk_idx, result in enumerate(chunk_results):
                if result is not None and len(result) > 0:
                    logging.info(f"Chunk {chunk_idx}: {len(result)} itineraries")
                    all_results.append(result)

                # idiot proofing
                else:
                    logging.warning(f"Chunk {chunk_idx}: No results returned")

            logging.info("Successfully combined results from all chunks")

            # check that all_results is not empty
            if all_results:
                combined_results_df = pd.concat(all_results, ignore_index=True)
                logging.info(f"Successfully calculated detailed itineraries for all origin-destination pairs"
                             f"; total itineraries: {len(combined_results_df)}")

                # writing combined results to cache folder
                output_filepath = os.path.join(self.cache_folder, f"{self.analysis_name}_detailed_itineraries.csv")
                combined_results_df.to_csv(output_filepath, index=False)
                logging.info(f"Written detailed itineraries to {output_filepath}")

                return combined_results_df

            else:
                logging.error("No results were returned from any chunk, cannot combine results")
                return None
        # in case of problems when running
        except Exception as e:
            logging.error(f"Error occurred during multiprocessing: {e}")
            raise

















