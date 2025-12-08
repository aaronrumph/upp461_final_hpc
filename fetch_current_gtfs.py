from tools_for_places import *
from create_transportation_network import *

"""
This script is what is actually going to generate the travel times for the transit network and driving network.
Because I know which data I am using, I can hardcode some of the parameters for the analysis here.
"""

# the path to the pbf sourced from SliceOSM (used to draw a bounding box around the area to make the network smaller)
pbf_path = None



# the list of modes to include in the transit network analysis
transit_analysis_modes = [r5py.TransportMode.WALK,
                          r5py.TransportMode.BUS,
                          r5py.TransportMode.RAIL,
                          r5py.TransportMode.SUBWAY,
                          r5py.TransportMode.TRAM,
                          r5py.TransportMode.FERRY]

# load the geojson for the cmap boundary polygon
cmap_boundary = gpd.read_file(
    r"C:\Users\aaron\OneDrive\Documents\ArcGIS\Projects\MyProject\Traffic_Analys_FeaturesToJSO.geojson")

# find the total bounding box for the cmap boundary
bbox = cmap_boundary.total_bounds

# query transit land api for all transit agencies in the cmap area
transitland_url = (f"https://transit.land/api/v2/rest/agencies?api_key="
                   f"{transit_land_api_key}&bbox={bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}")

transit_land_agencies_response = requests.get(transitland_url)
transit_agencies = transit_land_agencies_response.json()

print(transit_agencies)

# the gtfs folders to use for the transit network creation
current_gtfs_folders = []

# the TransitAgency objects created from the transit land
current_transit_agencies = []

# the dict of agencies to their gtfs zip file paths
current_gtfs_zip_folders = {}


temp_agencies_list = []
# going through the agency dicts provided by the api and using them as kwargs for TransitAgency object
for agency_data in transit_agencies["agencies"]:
    # fill out TransitAgency objects using the data from the API
    temp_agency = TransitAgency(**agency_data)
    # adding to the list of agencies for both (will modify no monroe later)
    current_transit_agencies.append(temp_agency)

# the paths where the gtfs data lives
current_gtfs_data_dir = os.path.join(Path(__file__).parent, "input_data", "gtfs_data", "current")


# now need to download the gtfs data for each agency
for agency in current_transit_agencies:
    # onestop_id (used to query for feed) is in feed_version["feed"]["onestop_id"]
    feed_version = agency.feed_version
    onestop_id = feed_version["feed"]["onestop_id"]
    transit_land_api_url = (f"https://transit.land/api/v2/rest/feeds/{onestop_id}/download_latest_feed_version"
                            f"?api_key={transit_land_api_key}")

    # standard API query
    transit_land_feed_response = requests.get(transit_land_api_url)
    transit_land_feed_response.raise_for_status()

    # the file path where the zipped gtfs data will be saved to (yes complicated, but best for organization)
    current_file_path = os.path.join(current_gtfs_data_dir, f"{onestop_id}.zip")

    # make directories if they don't already exist
    os.makedirs(os.path.dirname(current_file_path), exist_ok=True)


    # checking info about the feed version to see if still valid
    feed_version_query_response = requests.get(f"https://transit.land/api/v2/rest/feeds/{onestop_id}"
                                               f"?api_key={transit_land_api_key}")
    feed_version_query_response.raise_for_status()
    feed_version_query_response_json = feed_version_query_response.json()

    # when the feed is valid until
    latest_feed_valid_until = (feed_version_query_response_json["feeds"][0]["feed_versions"]
    [0]["latest_calendar_date"])
    valid_until_date_iso = isodate.parse_date(latest_feed_valid_until)
    current_date_iso = isodate.parse_date(datetime.now().strftime("%Y-%m-%d"))

    # only download if still valid
    if valid_until_date_iso > current_date_iso:
        # writing the content that was returned from transit land to a zip file in current folder and no monroe folder
        with open(current_file_path, "wb") as gtfs_zipped_file:
            gtfs_zipped_file.write(transit_land_feed_response.content)

    # adding to the dict of agency to gtfs zip file path
    current_gtfs_zip_folders[agency] = current_file_path

