#Checked with source
class State:
    def __init__(self):
        self.pn = 0.      # inertial north position in meters
        self.pe = 0.      # inertial east position in meters
        self.altitude = 100.       # inertial altitude in meters
        self.phi = 0.     # roll angle in radians
        self.theta = 0.   # pitch angle in radians
        self.psi = 0.     # yaw angle in radians
        self.Va = 25.      # airspeed in meters/sec
        self.alpha = 0.   # angle of attack in radians
        self.beta = 0.    # sideslip angle in radians
        self.p = 0.       # roll rate in radians/sec
        self.q = 0.       # pitch rate in radians/sec
        self.r = 0.       # yaw rate in radians/sec
        self.Vg = 25.      # groundspeed in meters/sec
        self.gamma = 0.   # flight path angle in radians
        self.chi = 0.     # course angle in radians
        self.wn = 0.      # inertial windspeed in north direction in meters/sec
        self.we = 0.      # inertial windspeed in east direction in meters/sec

