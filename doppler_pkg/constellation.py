import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class KeplerianElements:
    a: float  # Semi-major axis (m)
    e: float  # Eccentricity
    i: float  # Inclination (rad)
    omega: float  # Argument of perigee (rad)
    RAAN: float  # Right Ascension of Ascending Node (rad)
    M: float  # Mean Anomaly (rad)
    sat_id: int # Unique ID

class WalkerConstellation:
    """
    Generates Keplerian elements for a Walker Delta constellation.
    """
    def __init__(self, total_sats: int, planes: int, phasing: int, altitude: float, inclination_deg: float):
        self.T = total_sats
        self.P = planes
        self.F = phasing
        self.altitude = altitude
        self.inclination = np.radians(inclination_deg)
        
        self.sats_per_plane = self.T // self.P
        self.RE = 6378137.0  # Earth Radius (m)
        self.a = self.RE + self.altitude

    def generate_elements(self, start_id: int = 0) -> List[KeplerianElements]:
        elements = []
        id_counter = start_id
        
        for p in range(self.P):
            # RAAN for this plane
            raan = (2 * np.pi * p) / self.P
            
            for s in range(self.sats_per_plane):
                # Mean Anomaly for this satellite
                # Phasing factor F determines the relative phasing between planes
                # Delta M = 2*pi * (p * F / P)
                
                # Position in plane
                m_in_plane = (2 * np.pi * s) / self.sats_per_plane
                
                # Phasing offset
                m_offset = (2 * np.pi * p * self.F) / (self.T) # Walker Delta phasing formula: 2*pi * (P * F / T) ? No, usually 2*pi * F/P is the shift between adjacent planes.
                # Standard Walker notation i: T/P/F
                # RAAN(p) = 2pi * p / P
                # M(p, s) = 2pi * s / S + 2pi * F * p / T
                
                m = m_in_plane + (2 * np.pi * self.F * p) / self.T
                
                # Normalize M
                m = m % (2 * np.pi)
                
                elem = KeplerianElements(
                    a=self.a,
                    e=0.001, # Near circular
                    i=self.inclination,
                    omega=0.0, # Argument of perigee
                    RAAN=raan,
                    M=m,
                    sat_id=id_counter
                )
                elements.append(elem)
                id_counter += 1
                
        return elements

class LEOS9Constellation:
    """
    Generates the full LEO-S9 mixed constellation.
    441 Satellites total.
    3 Shells: 85deg, 55deg, 25deg.
    Assuming equal split: 147 sats per shell.
    """
    def __init__(self):
        self.shells = [
            # (Total, Planes, Phasing, Alt, Inc)
            # 147 sats, 7 planes (21 per plane), F=1 ?
            # Let's assume a standard configuration for high coverage
            {"T": 147, "P": 7, "F": 1, "alt": 800000, "inc": 85},
            {"T": 147, "P": 7, "F": 1, "alt": 800000, "inc": 55},
            {"T": 147, "P": 7, "F": 1, "alt": 800000, "inc": 25},
        ]

    def generate_all(self) -> List[KeplerianElements]:
        all_sats = []
        current_id = 0
        for shell_config in self.shells:
            walker = WalkerConstellation(
                shell_config["T"], 
                shell_config["P"], 
                shell_config["F"], 
                shell_config["alt"], 
                shell_config["inc"]
            )
            sats = walker.generate_elements(start_id=current_id)
            all_sats.extend(sats)
            current_id += len(sats)
        return all_sats

class StarlinkConstellation:
    """
    Generates Starlink Shell 1.
    1584 Satellites.
    Altitude: 550 km
    Inclination: 53 deg
    Planes: 72
    Sats per Plane: 22
    """
    def __init__(self):
        self.walker = WalkerConstellation(
            total_sats=1584,
            planes=72,
            phasing=1, 
            altitude=550000,
            inclination_deg=53.0
        )

    def generate_all(self) -> List[KeplerianElements]:
        return self.walker.generate_elements()
