"""
autopilot block for mavsim_python
    - Beard & McLain, PUP, 2012
    - Last Update:
        2/10/22 - RWB
"""
import numpy as np
from numpy import array, sin, cos, radians, concatenate, zeros, diag
from scipy.linalg import solve_continuous_are, inv
import Parameters.control_parameters as AP
from Tools.wrap import wrap
import Models.model_coef as M
from Message_types.state import State
from Message_types.delta import Delta

def saturate(input, low_limit, up_limit):
    if input <= low_limit:
        output = low_limit
    elif input >= up_limit:
        output = up_limit
    else:
        output = input
    return output


class Autopilot:
    def __init__(self, ts_control, trim_delta=None):
        self.Ts = ts_control
        self.trim_delta = trim_delta
        # initialize integrators and delay variables
        self.integratorCourse = 0
        self.integratorAltitude = 0
        self.integratorAirspeed = 0
        self.errorCourseD1 = 0
        self.errorAltitudeD1 = 0
        self.errorAirspeedD1 = 0
        # compute LQR gains
        
        #### TODO ######
        CrLat = array([[0, 0, 0, 0, 1.0]])
        AAlat = concatenate((
                    concatenate((M.A_lat, zeros((5,1))), axis=1),
                    concatenate((CrLat, zeros((1,1))), axis=1)),
                    axis=0)
        BBlat = concatenate((M.B_lat, zeros((1,2))), axis=0)
        Qlat = diag([0.001, 0.01, 0.1, 100, 1, 100]) # v, p, r, phi, chi, intChi
        Rlat = diag([0.5, 0.5]) # a, r
        Plat = solve_continuous_are(AAlat, BBlat, Qlat, Rlat)
        self.Klat = inv(Rlat) @ BBlat.T @ Plat
        # CrLon = array([[0, 0, 0, 0, 1.0], [1.0, 0, 0, 0, 0]])
        CrLon = array([[0, 0, 0, 0, 1.0], [1/AP.Va0, 1/AP.Va0, 0, 0, 0]])
        AAlon = concatenate((
                    concatenate((M.A_lon, zeros((5,2))), axis=1),
                    concatenate((CrLon, zeros((2,2))), axis=1)),
                    axis=0)
        BBlon = concatenate((M.B_lon, zeros((2, 2))), axis=0)
        Qlon = diag([10, 10, 0.001, 0.01, 10, 100, 10]) # u, w, q, theta, h, intH, intVa
        Rlon = diag([1, 1])  # e, t
        Plon = solve_continuous_are(AAlon, BBlon, Qlon, Rlon)
        # Plon = np.zeros((7,7))
        self.Klon = inv(Rlon) @ BBlon.T @ Plon
        # self.Klat = np.zeros((2,7))
        self.commanded_state = State()

    def update(self, cmd, state):
        # lateral autopilot
        errorAirspeed = state.Va - cmd.airspeed_command
        chi_c = wrap(cmd.course_command, state.chi)
        errorCourse = saturate(state.chi - chi_c, -radians(15), radians(15))
        self.integratorCourse = self.integratorCourse + (self.Ts/2.0)*(errorCourse + self.errorCourseD1)
        self.errorCourseD1 = errorCourse
        xLat = array([[errorAirspeed * sin(state.beta)],
                    [state.p],
                    [state.r],
                    [state.phi],
                    [errorCourse],
                    [self.integratorCourse]])
        tmp = -self.Klat @ xLat
        
        # Apply LQR computed deviations to trim values
        delta_a_trim = 0.0 if self.trim_delta is None else self.trim_delta.aileron
        delta_r_trim = 0.0 if self.trim_delta is None else self.trim_delta.rudder
        delta_a = saturate(tmp.item(0) + delta_a_trim, -radians(30), radians(30))
        delta_r = saturate(tmp.item(1) + delta_r_trim, -radians(30), radians(30))

        # longitudinal autopilot
        altitude_c = saturate(cmd.altitude_command,
                            state.altitude - 0.2*AP.altitude_zone,
                            state.altitude + 0.2*AP.altitude_zone)
        errorAltitude = state.altitude - altitude_c
        self.integratorAltitude = self.integratorAltitude + (self.Ts/2.0)*(errorAltitude + self.errorAltitudeD1)
        self.errorAltitudeD1 = errorAltitude
        self.integratorAirspeed = self.integratorAirspeed + (self.Ts/2.0)*(errorAirspeed + self.errorAirspeedD1)
        self.errorAirspeedD1 = errorAirspeed
        xLon = array([[errorAirspeed * cos(state.alpha)],
                    [errorAirspeed * sin(state.alpha)],
                    [state.q],
                    [state.theta],
                    [errorAltitude],
                    [self.integratorAltitude],
                    [self.integratorAirspeed]])
        tmp = -self.Klon @ xLon
        
        # Apply LQR computed deviations to trim values
        delta_e_trim = 0.0 if self.trim_delta is None else self.trim_delta.elevator
        delta_t_trim = 0.0 if self.trim_delta is None else self.trim_delta.throttle
        delta_e = saturate(tmp.item(0) + delta_e_trim, -radians(30), radians(30))
        delta_t = saturate(tmp.item(1) + delta_t_trim, 0.0, 5.0)

        # construct control outputs and commanded states
        delta = Delta(elevator=delta_e,
                        aileron=delta_a,
                        rudder=delta_r,
                        throttle=delta_t)
        self.commanded_state.altitude = cmd.altitude_command
        self.commanded_state.Va = cmd.airspeed_command
        return delta, self.commanded_state
