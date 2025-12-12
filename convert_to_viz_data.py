import csv
import json
import math

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
    print("Reading leo_s9_results.csv...")
    
    satellites = {}
    
    with open('leo_s9_results.csv', 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        print("Processing data...")
        for row in reader:
            try:
                sat_id = int(row['sat_id'])
                time_step = int(row['time'])
                
                if sat_id not in satellites:
                    satellites[sat_id] = {
                        'id': sat_id,
                        'path': []
                    }
                
                sat_x = float(row['sat_x'])
                sat_y = float(row['sat_y'])
                sat_z = float(row['sat_z'])
                
                lat, lon, alt = ecef_to_lla(sat_x, sat_y, sat_z)
                
                satellites[sat_id]['path'].append({
                    'time': time_step,
                    'lat': float(lat),
                    'lng': float(lon),
                    'alt': float(alt) / 6371.0, # Globe.gl expects altitude relative to earth radius (approx)
                    'real_alt_km': float(alt),
                    'doppler': float(row['doppler']),
                    'range': float(row['true_range']),
                    'az': float(row['az']),
                    'el': float(row['el'])
                })
            except ValueError:
                continue # Skip bad rows

    # Convert to list
    sat_list = list(satellites.values())
    
    # Sort paths by time just in case
    for sat in sat_list:
        sat['path'].sort(key=lambda x: x['time'])

    print(f"Found {len(sat_list)} satellites.")
    
    # Write to JS file
    with open('viz/data.js', 'w') as f:
        json_str = json.dumps(sat_list, indent=2)
        f.write(f"const SATELLITE_DATA = {json_str};")
        
    print("Written to viz/data.js")

if __name__ == "__main__":
    main()
