from astropy.time import Time
import os
from astroquery.jplhorizons import Horizons
import pandas as pd
import numpy as np
import mlx.core as mx

#(id, name, mass relative to earth, distance from earth)
planets = ((1, 'Mercury', 0.0553, 0.61), (2, 'Venus', 0.815, 0.28), (4, 'Mars', 0.107, 0.52), (5, 'Jupiter', 318, 4.2), (6, 'Saturn', 95.2, 8.52), (7, 'Uranus', 14.5, 18.2), (7, 'Neptune', 17.1, 29.09))
start_date = "1970-01-01"
end_date = "2070-01-01"
module_dir = os.path.dirname(__file__)
data_dir = os.path.join(module_dir, 'data')
astro_constants = {}

def download_astro_data(planet_id, planet_name, data_dir):    
    ephemerides_output_file = os.path.join(data_dir, planet_name + '.csv')
    elements_output_file = os.path.join(data_dir, planet_name + '_elements.csv')

    if not os.path.exists(ephemerides_output_file):
        horizons_eph = Horizons(id=planet_id, location='@sun', epochs={'start':start_date, 'stop': end_date, 'step':'1d'})
        ephemerides = horizons_eph.ephemerides()        
        ephemerides.write(ephemerides_output_file, format='ascii.csv', overwrite=True)
        print(f"Downloaded {planet_name} ephemerides to {ephemerides_output_file}")   

    if not os.path.exists(elements_output_file):
        horizons_elements = Horizons(id=planet_id, location='@sun', epochs={'start':start_date, 'stop': '1970-01-02', 'step':'1d'})
        elements = horizons_elements.elements()
        elements.write(elements_output_file, format='ascii.csv', overwrite=True)
        print(f"Downloaded {planet_name} elements to {elements_output_file}")

    return (ephemerides_output_file, elements_output_file)


def convert_eph_date_string(date_str):
    return pd.to_datetime(date_str, format = "%Y-%b-%d %H:%M")

def convert_elm_date_string(date_str):
    return pd.to_datetime(date_str, format = "A.D. %Y-%b-%d %H:%M:%S.%f")

def initialize_astro_data():
    transform_eph_time = np.vectorize(convert_eph_date_string)
    transform_elm_time = np.vectorize(convert_elm_date_string)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    for planet in planets:
        planet_id, planet_name, mass, dist = planet
        ephemerides_output_file, elements_output_file = download_astro_data(planet_id, planet_name, data_dir)
        ephemerides_data = pd.read_csv(ephemerides_output_file)
        ephemerides_data["datetime_str"] = transform_eph_time(ephemerides_data["datetime_str"])
        ephemerides_data.set_index("datetime_str", inplace=True)

        elements_data = pd.read_csv(elements_output_file)
        elements_data["datetime_str"] = transform_elm_time(elements_data["datetime_str"])
        elements_data.set_index("datetime_str", inplace=True)
        
        astro_constants[planet_name] = {
            "λ": ephemerides_data['EclLon'],
            "c": mass / (dist*dist),
            "p": elements_data['P']['1970-01-01']
        }
    