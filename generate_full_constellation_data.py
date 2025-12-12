import numpy as np
import json
import math
import sys
import os

# Add current directory to path to import doppler_pkg
sys.path.append(os.getcwd())

from doppler_pkg.constellation import StarlinkConstellation
from doppler_pkg.orbit_propagator import NumericalPropagator
from doppler_pkg.constants import C, TX_FREQUENCY
from skyfield.api import load

def ecef_to_lla(x, y, z):
    # WGS84 constants
    a = 6378137.0
    e = 8.1819190842622e-2

    asq = a ** 2
    esq = e ** 2

    b = math.sqrt(asq * (1 - esq))
    bsq = b ** 2
    ep = math.sqrt((asq - bsq) / bsq)
    p = math.sqrt(x ** 2 + y ** 2)
    th = math.atan2(a * z, b * p)

    lon = math.atan2(y, x)
    lat = math.atan2((z + ep ** 2 * b * math.sin(th) ** 3), (p - esq * a * math.cos(th) ** 3))
    N = a / math.sqrt(1 - esq * math.sin(lat) ** 2)
    alt = p / math.cos(lat) - N

    # Convert to degrees
    lon = lon * 180 / math.pi
    lat = lat * 180 / math.pi

    return lat, lon, alt / 1000.0  # Altitude in km

def main():
    print("Initializing Full Constellation (Starlink Shell 1)...")
    constellation = StarlinkConstellation()
    sats_elements = constellation.generate_all()
    print(f"Generated {len(sats_elements)} satellites.")
    
    propagators = [NumericalPropagator(el) for el in sats_elements]
    
    # User position (Mumbai)
    user_lat = 19.0760
    user_lon = 72.8777
    RE = 6378137.0
    lat_rad = np.radians(user_lat)
    lon_rad = np.radians(user_lon)
    user_pos_ecef = np.array([
        RE * np.cos(lat_rad) * np.cos(lon_rad),
        RE * np.cos(lat_rad) * np.sin(lon_rad),
        RE * np.sin(lat_rad)
    ])
    
    duration = 60
    step = 1.0
    
    sat_data = {}
    
    print(f"Propagating for {duration} seconds...")
    
    # Initialize sat_data structure
    for el in sats_elements:
        sat_data[el.sat_id] = {
            'id': el.sat_id,
            'path': []
        }

    ts = load.timescale()
    
    for t_sec in range(0, duration, int(step)):
        if t_sec % 10 == 0:
            print(f"Time: {t_sec}/{duration}")
            
        for i, prop in enumerate(propagators):
            prop.step(step)
            
            r_eci = prop.state[:3]
            v_eci = prop.state[3:]
            
            # ECI to ECEF (simplified rotation for viz)
            # We need accurate rotation for Doppler, so we'll use the Skyfield logic if possible
            # or a simplified rotation based on earth rotation rate.
            # Reusing logic from leo_s9_sim.py for consistency
            
            delta_t_days = 69.184 / 86400.0
            jd_ut1 = prop.t_current_jd - delta_t_days
            Tu = jd_ut1 - 2451545.0
            theta = 2 * np.pi * (0.7790572732640 + 1.00273781191135448 * Tu)
            theta = theta % (2 * np.pi)
            
            c = np.cos(theta)
            s = np.sin(theta)
            R_z = np.array([
                [c, s, 0],
                [-s, c, 0],
                [0, 0, 1]
            ])
            
            sat_pos_ecef = R_z @ r_eci
            omega_e = np.array([0, 0, 7.2921159e-5])
            sat_vel_ecef = R_z @ v_eci - np.cross(omega_e, sat_pos_ecef)
            
            # Calculate Geometry
            diff = sat_pos_ecef - user_pos_ecef
            dist = np.linalg.norm(diff)
            u_los = diff / dist
            
            # Elevation (simplified)
            # We don't strictly need elevation for viz unless we want to color code visibility
            # But let's calculate it for the HUD
            
            # Up vector at user pos
            u_up = user_pos_ecef / np.linalg.norm(user_pos_ecef)
            el_rad = np.arcsin(np.dot(u_los, u_up))
            el_deg = np.degrees(el_rad)
            
            # Doppler
            range_rate = np.dot(sat_vel_ecef, u_los)
            doppler = - (range_rate / C) * TX_FREQUENCY
            
            # LLA for Globe.gl
            lat, lon, alt_km = ecef_to_lla(sat_pos_ecef[0], sat_pos_ecef[1], sat_pos_ecef[2])
            
            sat_data[sats_elements[i].sat_id]['path'].append({
                'time': t_sec,
                'lat': float(lat),
                'lng': float(lon),
                'alt': float(alt_km) / 6371.0,
                'real_alt_km': float(alt_km),
                'doppler': float(doppler),
                'range': float(dist),
                'el': float(el_deg)
            })

    print("Writing to viz/data.js...")
    sat_list = list(sat_data.values())
    
    # Reduce precision to save space
    for sat in sat_list:
        for p in sat['path']:
            p['lat'] = round(p['lat'], 3)
            p['lng'] = round(p['lng'], 3)
            p['alt'] = round(p['alt'], 4)
            p['real_alt_km'] = round(p['real_alt_km'], 1)
            p['doppler'] = round(p['doppler'], 1)
            p['range'] = round(p['range'], 1)
            p['el'] = round(p['el'], 1)

    with open('viz/data.js', 'w') as f:
        json_str = json.dumps(sat_list) # Minified
        f.write(f"const SATELLITE_DATA = {json_str};")
        
    print("Done.")

if __name__ == "__main__":
    main()
