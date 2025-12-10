import numpy as np
from skyfield.api import load, wgs84, EarthSatellite
from skyfield.positionlib import Geocentric
from skyfield.timelib import Time
from .constellation import KeplerianElements

# Constants
GM = 3.986004418e14  # Earth's Gravitational Parameter (m^3/s^2)
RE = 6378137.0       # Earth Radius (m)
J2 = 1.08262668e-3   # J2 Coefficient
CD = 2.2             # Drag Coefficient (Typical LEO)
CR = 1.8             # Radiation Pressure Coefficient
AREA_MASS_RATIO = 0.01 # m^2/kg (Guess for microsat)
P_SUN = 4.56e-6      # Solar Radiation Pressure at 1 AU (N/m^2)

class ForceModel:
    def __init__(self):
        self.ts = load.timescale()
        self.eph = load('de421.bsp') # For Sun position
        self.sun = self.eph['sun']
        self.earth = self.eph['earth']

    def compute_acceleration(self, t_jd: float, r: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Compute total acceleration in ECI frame.
        r: Position vector (m)
        v: Velocity vector (m/s)
        t_jd: Time in Julian Date
        """
        r_mag = np.linalg.norm(r)
        
        # 1. Monopole Gravity
        # a = -GM/r^3 * r
        a_grav = -GM / (r_mag**3) * r
        
        # 2. J2 Perturbation
        # Simplified J2 acceleration in ECI
        # z = r[2]
        # a_x = a_grav_x * (1 + 1.5*J2*(Re/r)^2 * (1 - 5*(z/r)^2))
        # a_y = a_grav_y * (1 + 1.5*J2*(Re/r)^2 * (1 - 5*(z/r)^2))
        # a_z = a_grav_z * (1 + 1.5*J2*(Re/r)^2 * (3 - 5*(z/r)^2))
        
        z2 = r[2]**2
        r2 = r_mag**2
        factor_j2 = 1.5 * J2 * (RE**2 / r2)
        tx = (1 - 5 * z2 / r2)
        tz = (3 - 5 * z2 / r2)
        
        a_j2 = np.zeros(3)
        a_j2[0] = a_grav[0] * factor_j2 * tx
        a_j2[1] = a_grav[1] * factor_j2 * tx
        a_j2[2] = a_grav[2] * factor_j2 * tz # Note: a_grav[2] already has the -GM/r^3
        
        # Correction: The formula is usually applied as a perturbation force added to monopole
        # a_J2_x = - (3/2) J2 (GM/r^2) (Re/r)^2 (1 - 5(z/r)^2) (x/r)
        #        = - (3/2) J2 (GM Re^2 / r^5) * x * (1 - 5(z/r)^2)
        
        pre_factor = -1.5 * J2 * GM * (RE**2) / (r_mag**5)
        a_j2_vec = np.zeros(3)
        a_j2_vec[0] = pre_factor * r[0] * (1 - 5 * z2 / r2)
        a_j2_vec[1] = pre_factor * r[1] * (1 - 5 * z2 / r2)
        a_j2_vec[2] = pre_factor * r[2] * (3 - 5 * z2 / r2)

        # 3. Atmospheric Drag
        # a_d = -0.5 * rho * v_rel^2 * (Cd * A / m) * unit(v_rel)
        # Simple exponential atmosphere model
        # rho = rho0 * exp(-(h - h0) / H)
        h = r_mag - RE
        rho = self._get_density(h)
        
        # Assuming Earth rotation for v_rel (approximate)
        # omega_earth = [0, 0, 7.2921159e-5]
        # v_atm = cross(omega_earth, r)
        # v_rel = v - v_atm
        omega_e = 7.2921159e-5
        v_rel = np.array([
            v[0] + omega_e * r[1],
            v[1] - omega_e * r[0],
            v[2]
        ])
        v_rel_mag = np.linalg.norm(v_rel)
        
        a_drag = -0.5 * rho * (v_rel_mag**2) * CD * AREA_MASS_RATIO * (v_rel / v_rel_mag)
        
        # 4. Solar Radiation Pressure
        # Need Sun vector
        # Using Skyfield to get sun position
        t = self.ts.tt_jd(t_jd)
        # Earth to Sun vector
        sun_pos = self.earth.at(t).observe(self.sun).position.m
        sun_vec = sun_pos - r # Satellite to Sun? No, Earth->Sun is roughly Sat->Sun for direction
        # Actually, we need Sat->Sun vector.
        # Sun position is usually given relative to Earth (GCRS).
        # So r_sun_rel_sat = r_sun_rel_earth - r_sat_rel_earth
        r_sun_sat = sun_pos - r
        dist_sun = np.linalg.norm(r_sun_sat)
        u_sun = r_sun_sat / dist_sun
        
        # Check shadow (Cylindrical shadow model)
        in_sun = self._check_visibility(r, sun_pos)
        
        a_srp = np.zeros(3)
        if in_sun:
            a_srp = -P_SUN * CR * AREA_MASS_RATIO * u_sun # Force is AWAY from sun?
            # Wait, radiation pushes AWAY from sun.
            # u_sun points TO sun.
            # So force is along -u_sun? No, force is along u_sun (away from sun? No, light travels from sun to sat)
            # Light direction: Sun -> Sat. Force direction: Sun -> Sat.
            # u_sun is Sat -> Sun.
            # So Force is along -u_sun.
            a_srp = -P_SUN * CR * AREA_MASS_RATIO * u_sun

        return a_grav + a_j2_vec + a_drag + a_srp

    def _get_density(self, h_m: float) -> float:
        # Simple exponential model
        # Harris-Priester or similar simplified
        if h_m > 1000000: return 0.0
        
        # Reference values (approximate)
        h0 = 800000 # 800 km
        rho0 = 1.170e-14 # kg/m^3 at 800km (very rough, varies wildly with solar cycle)
        H = 150000 # Scale height
        
        return rho0 * np.exp(-(h_m - h0) / H)

    def _check_visibility(self, r_sat: np.ndarray, r_sun: np.ndarray) -> bool:
        # Cylindrical shadow model
        # Projection of r_sat onto r_sun
        # s = dot(r_sat, r_sun) / |r_sun|
        # if s > 0 (sat is on day side relative to earth center plane): visible
        # if s < 0: check distance from axis
        
        # Actually, simpler:
        # Check if Earth blocks Sun->Sat line.
        # But for LEO, Earth shadow is significant.
        
        # Vector from Earth to Sat: r_sat
        # Vector from Earth to Sun: r_sun
        
        # Angle between them?
        # If angle is > 90 deg, might be in shadow.
        
        # Projection of Sat pos onto Sun-Earth line
        # L = -r_sat . unit(r_sun)
        # If L < 0 (sat is "in front" of Earth towards sun), it's lit.
        # If L > 0 (sat is "behind" Earth), check radial distance.
        
        u_sun = r_sun / np.linalg.norm(r_sun)
        L = np.dot(r_sat, u_sun) # Component along Sun axis
        
        if L > 0: # Towards Sun
            return True
            
        # Perpendicular distance
        D = np.linalg.norm(r_sat - L * u_sun)
        if D > RE:
            return True
            
        return False


class NumericalPropagator:
    def __init__(self, elements: KeplerianElements):
        self.elements = elements
        self.force_model = ForceModel()
        
        # Initialize state (r, v) from Keplerian elements
        self.t_current_jd = None
        self.state = None # [x, y, z, vx, vy, vz]
        self._init_state()

    def _init_state(self):
        # Convert Keplerian to Cartesian
        # Using Skyfield or manual?
        # Skyfield doesn't easily convert elements to state vectors without TLE.
        # I'll implement a simple Kepler to Cartesian converter.
        
        a = self.elements.a
        e = self.elements.e
        i = self.elements.i
        om = self.elements.omega
        raan = self.elements.RAAN
        M = self.elements.M
        
        # Solve Kepler's Equation for E
        # M = E - e sin E
        E = M
        for _ in range(10):
            E = M + e * np.sin(E)
            
        # Perifocal coordinates
        nu = 2 * np.arctan(np.sqrt((1+e)/(1-e)) * np.tan(E/2))
        r_c = a * (1 - e * np.cos(E))
        
        o = np.zeros(3)
        o[0] = r_c * np.cos(nu)
        o[1] = r_c * np.sin(nu)
        
        v_c = np.sqrt(GM * a) / r_c
        # Velocity in perifocal?
        # p = a(1-e^2)
        # h = sqrt(GM * p)
        # r_dot = (h/p) * e * sin(nu)
        # r_nu_dot = (h/r)
        
        # Easier:
        # x_dot = -sqrt(GM/a)/r * sin E ? No
        # x_dot = - (n * a^2 / r) * sin E
        # y_dot = (n * a^2 / r) * sqrt(1-e^2) * cos E
        n = np.sqrt(GM / a**3)
        x_dot = -(n * a**2 / r_c) * np.sin(E)
        y_dot = (n * a**2 / r_c) * np.sqrt(1 - e**2) * np.cos(E)
        
        vel_perifocal = np.array([x_dot, y_dot, 0])
        pos_perifocal = np.array([a * (np.cos(E) - e), a * np.sqrt(1-e**2) * np.sin(E), 0])
        
        # Rotation to ECI
        # R3(-RAAN) R1(-i) R3(-omega)
        
        # R_omega
        cw = np.cos(om)
        sw = np.sin(om)
        R_w = np.array([[cw, -sw, 0], [sw, cw, 0], [0, 0, 1]])
        
        # R_i
        ci = np.cos(i)
        si = np.sin(i)
        R_i = np.array([[1, 0, 0], [0, ci, -si], [0, si, ci]])
        
        # R_raan
        cO = np.cos(raan)
        sO = np.sin(raan)
        R_O = np.array([[cO, -sO, 0], [sO, cO, 0], [0, 0, 1]])
        
        R = R_O @ R_i @ R_w
        
        r_eci = R @ pos_perifocal
        v_eci = R @ vel_perifocal
        
        self.state = np.concatenate((r_eci, v_eci))
        
        # Set initial time (J2000 or now?)
        # Let's assume t=0 is J2000 for now, or user sets it.
        # We'll need a start time.
        ts = load.timescale()
        self.t_current_jd = ts.now().tt

    def step(self, dt: float):
        """
        Perform one RK4 step.
        dt: Time step in seconds.
        """
        # Convert dt to days for JD
        dt_days = dt / 86400.0
        
        y = self.state
        t = self.t_current_jd
        
        k1 = self._derivs(t, y)
        k2 = self._derivs(t + dt_days/2, y + k1 * dt/2)
        k3 = self._derivs(t + dt_days/2, y + k2 * dt/2)
        k4 = self._derivs(t + dt_days, y + k3 * dt)
        
        self.state = y + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        self.t_current_jd += dt_days

    def _derivs(self, t_jd, y):
        # y = [rx, ry, rz, vx, vy, vz]
        # dydt = [vx, vy, vz, ax, ay, az]
        r = y[:3]
        v = y[3:]
        
        a = self.force_model.compute_acceleration(t_jd, r, v)
        
        return np.concatenate((v, a))
