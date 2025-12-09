# doppler_pkg/tle_manager.py
import os
import time
import requests
from skyfield.api import load, EarthSatellite
from .constants import TLE_FILENAME, TLE_URL

class TLEManager:
    """
    Module A: TLEManager (Data Handling)
    Responsibility: Fetch and manage orbital data.
    """
    def __init__(self, filename: str = TLE_FILENAME, url: str = TLE_URL):
        self.filename = filename
        self.url = url

    def _is_file_fresh(self) -> bool:
        """Check if local TLE file exists and is < 24 hours old."""
        if not os.path.exists(self.filename):
            return False
        file_age = time.time() - os.path.getmtime(self.filename)
        return file_age < 86400  # 24 hours in seconds

    def _download_tle(self):
        """Download fresh TLE data from CelesTrak."""
        print(f"Downloading fresh TLE data from {self.url}...")
        response = requests.get(self.url)
        response.raise_for_status()
        with open(self.filename, 'w') as f:
            f.write(response.text)
        print("Download complete.")

    def get_satellite(self, sat_name: str) -> EarthSatellite:
        """
        Parse the file to find a specific satellite object by name.
        Returns a Skyfield Satellite object.
        """
        if not self._is_file_fresh():
            self._download_tle()

        ts = load.timescale()
        satellites = load.tle_file(self.filename)
        by_name = {sat.name: sat for sat in satellites}
        
        if sat_name not in by_name:
            raise ValueError(f"Satellite '{sat_name}' not found in TLE file.")
            
        return by_name[sat_name]
