class State:
    def __init__(self):
        self.pn = 0.0
        self.pe = 0.0
        self.pd = -10.0
        self.phi = 10.0  # roll angle (radians)
        self.theta = 2.0  # pitch angle (radians)
        self.psi = 2.0  # yaw angle (radians)
        self.Va = 25.0  # airspeed 
        self.alpha = 0.0  # angle of attack (radians)
        self.beta = 0.0  # sideslip angle (radians)
        self.p = 1.       # roll rate (rad/s)
        self.q = 0.       # pitch rate (rad/s)
        self.r = 2.       # yaw rate (rad/s)
        self.Vg = 25.      # groundspeed 
        self.gamma = 0.   # flight path angle (radians)
        self.chi = 0.     # course angle (radians)
        self.wn = 0.      # inertial windspeed in north direction 
        self.we = 0.      # inertial windspeed in east direction 

