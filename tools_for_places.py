"""

"""
import time

# yes I know it's bad practice to use import * but in this case, I've made sure that it won't cause any problems
# (can safely map namespace of create_network_dataset_oop to this module because were developed in tandem)
from create_transportation_network import *
from general_tools import *

# standard library modules
import logging
import os

# logging setup
logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
# don't want to display debugging messages when running this script
logging.getLogger("requests").setLevel(logging.INFO)
logging.getLogger("urllib3").setLevel(logging.INFO)

class Place:
    """
    Class that represents a place and contains methods to create network datasets for it

    Parameters:
        arcgis_project (ArcProject): ArcGIS project object
        place_name (str): Name of place
        bound_box (tuple): Bounding box of place | (longitude_min, latitude_min, longitude_max, latitude_max)
        geographic_scope (str): Geographic scope of place | {"place_only"", "county", "msa", "csa", "specified"}
            specified means a list of places to include in the network dataset (must be well-formed place names that
            OSM will recognize).
        scenario_id (str): Scenario ID | if you would like to keep track of the same GeoDatabase/Network Dataset
            across multiple runs, then you should use the same scenario_id for each run! By default, will be
            a random (partially modified) base64 value.
        specified_places_to_include (list[str]): List of places to include in the network dataset if using
            "specified" geographic scope

    Methods:
        use_scope_for_place(geographic_scope="place_only") -> None: Sets the geographic scope for the place
        create_network_dataset_from_place(network_type="walk", use_elevation=False, full_reset=False,
                                          elevation_reset=False) -> None: Creates network dataset for place of specified
                                          type
        get_agencies_for_place() -> None: Gets agencies that serve the place (for transit network datasets)
        generate_isochrones_for_place() -> None: Generates isochrones for the place

    """

    def __init__(self, place_name: str | None = None,
                 bound_box: tuple[str | float, str | float, str | float, str | float] | None = None,
                 geographic_scope: str = "place_only",
                 scenario_id: str = None, specified_places_to_include: list[str] = None):

        if place_name is None and bound_box is None:
            raise ValueError("Must provide either a place or bounding box")
        # parameters
        self.place_name = place_name
        self.bound_box = bound_box
        self.geographic_scope = geographic_scope
        self.scenario_id = scenario_id
        self.specified_places_to_include = specified_places_to_include

        if self.scenario_id is None:
            self.scenario_id = generate_random_base64_value(1000000000)

        if self.bound_box:
            self.geographic_scope = "bbox"

        # important attributes not passed
        self.main_reference_place = ReferencePlace(place_name=self.place_name, bound_box=self.bound_box)
        self.snake_name = create_snake_name(self.main_reference_place)
        self.snake_name_with_scope = f"{self.snake_name}_{self.geographic_scope}"
        # cache folder for place
        self.cache_folder = CacheFolder(self.snake_name_with_scope)

        # list of reference places for creating networks
        self.reference_place_list = []

        # attributes to check whether certain things exist
        self.street_network_data_exists = False
        self.elevation_data_exists = False

        # set up cache folder when initializing instance if one doesn't already exist
        if not self.cache_folder.check_if_cache_folder_exists():
            self.cache_folder.set_up_cache_folder()

        # agencies that serve the place (will be set later by the get_agencies_for_place method)
        self.agencies_that_serve_place = None

        # always use scope for place
        self.use_scope_for_place(geographic_scope=geographic_scope)
        # always get list of states for place
        self.states_for_place = self.get_states_for_place_with_scope()

    def use_scope_for_place(self, geographic_scope="place_only"):

        # can set the geographic scope to something new
        self.geographic_scope = geographic_scope

        # if using bounding box
        if self.bound_box:
            # the list of ReferencePlaces to use in creating the network (just the bounding box)
            reference_place_list = [self.main_reference_place]

        # if using city limits and reference place
        elif self.place_name and self.geographic_scope == "place_only":
            # the list of ReferencePlaces to use in creating the network
            reference_place_list = [self.main_reference_place]

        # if using specified places
        elif self.place_name and self.geographic_scope == "specified":
            logging.info(f"Using the provided list of other places to include {self.specified_places_to_include}")
            # check that if specified is selected that specified_places_to_include is not None
            if self.specified_places_to_include is None:
                raise ValueError("Geographic scope is 'specified', but no specified places to include were provided")
            # for each place name in specified_places_to_include, create a ReferencePlace object and add it to the list
            reference_place_list = [ReferencePlace(place_name=place_name) for
                                    place_name in self.specified_places_to_include]
            # also need to add the original place to the list (in first position so that the network is seen as being
            # 'centered' on the main place)
            reference_place_list.insert(0, self.main_reference_place)

        # if using county, msa, or csa
        elif (self.place_name and
              (self.geographic_scope == "county" or self.geographic_scope == "msa" or self.geographic_scope == "csa")):
            reference_place_list = get_reference_places_for_scope(self.place_name, self.geographic_scope)

        else:
            raise ValueError("Geographic scope not recognized, please use "
                             "'city', 'county', 'msa', 'csa', or 'specified'")

        # now set the list
        self.reference_place_list = reference_place_list
        return reference_place_list

    def get_states_for_place_with_scope(self):
        """ Returns the list of states that the place (with geographic scope) is in

        :return: list[str] | list of state abbreviations that the place (with geographic scope) is in
        """
        states_for_place = set()
        # go through each reference place and get its state
        for reference_place in self.reference_place_list:
            # in case where state and country are provided
            if reference_place.place_name.count(",") == 2:
                this_state = reference_place.place_name.split(",")[1].strip()
                states_for_place.add(this_state)

            # in case where only county provided
            elif reference_place.place_name.count(",") == 1:
                this_state = reference_place.place_name.split(",")[1].strip()
                states_for_place.add(this_state)
        logging.info(f"Found {len(states_for_place)} states: {states_for_place}")
        return states_for_place


    def calculate_od_cost_matrix(self, analysis_name, origins_geojson_path, destinations_geojson_path,
                                 analysis_date, analysis_time, analysis_network_type="walk", output_csv_path=None,
                                 own_gtfs_folder_path=None):
        """
        Calculates the OD cost matrix for the given origins and destinations using the specified network dataset

        Parameters:
            :param analysis_name: str | name of the analysis
            :param origins_geojson_path: str | path to the GeoJSON file containing the origins
            :param destinations_geojson_path: str | path to the GeoJSON file containing the destinations
            :param analysis_date: str | date of the analysis in "MM/DD/YYYY" format
            :param analysis_time: str | time of the analysis in "HH:MM"
            :param analysis_network_type: str | type of network dataset to use {"walk", "bike", "drive", "transit"}
            :param output_csv_path: str | path to the output CSV file (if None, will use default path in cache folder)
            :param own_gtfs_folder_path: str | path to folder containing user-provided GTFS data (if None, will use
                TransitLand data
        Returns:
            None

        """
        logging.info(f"Calculating OD cost matrix for analysis '{analysis_name}' using {analysis_network_type} network")
        # am using a timer to give info back at the end"
        method_start_time = time.perf_counter()

        # set up street and transit networks
        this_street_network = StreetNetwork(geographic_scope=self.geographic_scope,
                                       reference_place_list=self.reference_place_list,
                                       states_to_include=self.states_for_place)
        this_street_network.get_osm_pbf_data()

        # in case using transit, make transit network
        if analysis_network_type == "transit":
            # check whether to use user-provided GTFS data or TransitLand
            if own_gtfs_folder_path is not None:
                logging.info(f"Using user-provided GTFS data from {own_gtfs_folder_path}")
                gtfs_data_paths = [gtfs_folder for gtfs_folder in os.listdir(own_gtfs_folder_path)]
            else:
                logging.info("Using GTFS data from TransitLand")
                gtfs_data_paths = None
            this_transit_network = TransitNetwork(geographic_scope=self.geographic_scope,
                                              reference_place_list=self.reference_place_list,
                                              own_gtfs_data_paths=gtfs_data_paths)
        else:
            this_transit_network = None

        # set up modes



        # now set up R5 network
        this_r5_network = R5Network(street_network=this_street_network, transit_network=this_transit_network,
                                    network_type=analysis_network_type)

        # calculate OD cost matrix
        logging.info("Setting up DetailedItineraries object for OD cost matrix calculation")
        analysis_detailed_itineraries = DetailedItineraries(analysis_name=analysis_name, r5_network=this_r5_network,
                                                            origins_geojson_path=origins_geojson_path,
                                                            destinations_geojson_path=destinations_geojson_path,
                                                            date=analysis_date, departure_time=analysis_time)

        analysis_detailed_itineraries.calculate_detailed_itineraries()

        # total run time
        method_end_time = time.perf_counter()
        total_run_time_seconds = method_end_time - method_start_time

        # give info about size of inputs, run time, threads used
        num_origins = len(analysis_detailed_itineraries.origins_gdf)
        num_destinations = len(analysis_detailed_itineraries.destinations_gdf)

        # threads used
        threads_used = analysis_detailed_itineraries.num_processes

        # chunk info
        ideal_chunks = calculate_ideal_od_chunk_size(num_origins, num_destinations)
        number_of_chunks = ideal_chunks[1]

        logging.info(f"Calculated OD cost matrix with {num_origins} origins and {num_destinations} destinations")
        logging.info(f"Completed {number_of_chunks} chunks using {threads_used} threads in "
                     f"{turn_seconds_into_minutes(total_run_time_seconds)}")

# # # # # # # # # # # # # # # # # # Testing Area :::: DO NOT REMOVE "if __name__ ..." # # # # # # # # # # # # # # # # #

if __name__ == "__main__":
    Chicago = Place(place_name="Chicago, IL", geographic_scope="csa")
    test_origins_path = r"C:\Users\aaron\PycharmProjects\network_datasets\upp_461_final_actual\taz_centroids.geojson"
    test_destinations_path = r"C:\Users\aaron\PycharmProjects\network_datasets\upp_461_final_actual\taz_centroids.geojson"
    test_departure_time = "09:00"
    test_date = "12/05/2025"
    test_network_type = "transit"

    Chicago.calculate_od_cost_matrix(analysis_name="Chicago_OD_Cost_Matrix_Test",
                                    origins_geojson_path=test_origins_path,
                                    destinations_geojson_path=test_destinations_path,
                                    analysis_date=test_date,
                                    analysis_time=test_departure_time,
                                    analysis_network_type=test_network_type)
