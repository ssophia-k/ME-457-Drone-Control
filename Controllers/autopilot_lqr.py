"""
autopilot block for mavsim_python
    - Beard & McLain, PUP, 2012
    - Last Update:
        2/10/22 - RWB
"""
import os, sys
# insert parent directory at beginning of python search path
from pathlib import Path
sys.path.insert(0,os.fspath(Path(__file__).parents[1]))

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
    def __init__(self, ts_control):
        self.Ts = ts_control
        self.trim_delta = Delta(
            elevator=M.u_trim.item(0),
            aileron=M.u_trim.item(1),
            rudder=M.u_trim.item(2),
            throttle=M.u_trim.item(3)
        )
        
        # initialize integrators and delay variables
        self.integratorCourse = 0
        self.integratorAltitude = 0
        self.integratorAirspeed = 0
        self.errorCourseD1 = 0
        self.errorAltitudeD1 = 0
        self.errorAirspeedD1 = 0
        
        # compute LQR gains
        HLat = array([[0, 0, 0, 0, 1.0]])
        AAlat = concatenate((
                    concatenate((M.A_lat, zeros((5,1))), axis=1),
                    concatenate((HLat, zeros((1,1))), axis=1)),
                    axis=0)
        BBlat = concatenate((M.B_lat, zeros((1,2))), axis=0)
        Qlat = diag([0.43055586330689405, 0.8161005484899314, 11.616778199952448, 21.86943030926892, 0.20995071471888255, 13.29816829974618]) # v, p, r, phi, chi, intChi
        Rlat = diag([0.003415326993571652, 4.936618432275214]) # a, r
        Plat = solve_continuous_are(AAlat, BBlat, Qlat, Rlat)
        self.Klat = inv(Rlat) @ BBlat.T @ Plat
        
        HLon = array([[0, 0, 0, 0, 1.0], [1/AP.Va0, 1/AP.Va0, 0, 0, 0]])
        AAlon = concatenate((
                    concatenate((M.A_lon, zeros((5,2))), axis=1),
                    concatenate((HLon, zeros((2,2))), axis=1)),
                    axis=0)
        BBlon = concatenate((M.B_lon, zeros((2, 2))), axis=0)
        Qlon = diag([8.857719780145729, 4.35253257393339, 2.1150472679802315, 12.315787768195039, 1.2344041283431726, 63.27903318413501, 4.828766697227217]) # u, w, q, theta, h, intH, intVa
        Rlon = diag([4.568284033194546, 1.5963397595437596])  # e, t
        Plon = solve_continuous_are(AAlon, BBlon, Qlon, Rlon)
        self.Klon = inv(Rlon) @ BBlon.T @ Plon
        
        self.commanded_state = State()

    def update(self, cmd, state):
        # lateral autopilot
        errorAirspeed = state.Va - cmd.airspeed_command
        
        # Properly handle course error with wrapping
        chi_c = wrap(cmd.course_command, state.chi)
        raw_error_course = wrap(state.chi - chi_c, 0)
        errorCourse = saturate(raw_error_course, -radians(15), radians(15))
        
        # Update course integrator
        self.integratorCourse = self.integratorCourse + (self.Ts/2.0)*(errorCourse + self.errorCourseD1)
        self.errorCourseD1 = errorCourse
        
        # Lateral state vector
        xLat = array([[errorAirspeed * sin(state.beta)],
                    [state.p],
                    [state.r],
                    [state.phi],
                    [errorCourse],
                    [self.integratorCourse]])
        tmp = -self.Klat @ xLat
        
        # Apply LQR computed deviations to trim values
        delta_a_trim = self.trim_delta.aileron
        delta_r_trim = self.trim_delta.rudder
        delta_a = saturate(tmp.item(0) + delta_a_trim, -radians(30), radians(30))
        delta_r = saturate(tmp.item(1) + delta_r_trim, -radians(30), radians(30))

        # longitudinal autopilot
        altitude_c = saturate(cmd.altitude_command,
                            state.altitude - 0.2*AP.altitude_zone,
                            state.altitude + 0.2*AP.altitude_zone)
        errorAltitude = state.altitude - altitude_c
        
        # Update altitude and airspeed integrators
        self.integratorAltitude = self.integratorAltitude + (self.Ts/2.0)*(errorAltitude + self.errorAltitudeD1)
        self.errorAltitudeD1 = errorAltitude
        self.integratorAirspeed = self.integratorAirspeed + (self.Ts/2.0)*(errorAirspeed + self.errorAirspeedD1)
        self.errorAirspeedD1 = errorAirspeed
        
        # Longitudinal state vector
        xLon = array([[errorAirspeed * cos(state.alpha)],
                    [errorAirspeed * sin(state.alpha)],
                    [state.q],
                    [state.theta],
                    [errorAltitude],
                    [self.integratorAltitude],
                    [self.integratorAirspeed]])
        tmp = -self.Klon @ xLon
        
        # Apply LQR computed deviations to trim values
        delta_e_trim = self.trim_delta.elevator
        delta_t_trim = self.trim_delta.throttle
        delta_e = saturate(tmp.item(0) + delta_e_trim, -radians(30), radians(30))
        delta_t = saturate(tmp.item(1) + delta_t_trim, 0.0, 1.0)

        # construct control outputs and commanded states
        delta = Delta(elevator=delta_e,
                      aileron=delta_a,
                      rudder=delta_r,
                      throttle=delta_t)
                      
        # Update commanded states
        self.commanded_state.altitude = cmd.altitude_command
        self.commanded_state.Va = cmd.airspeed_command
        self.commanded_state.phi = cmd.phi_feedforward
        self.commanded_state.theta = M.theta_trim
        self.commanded_state.chi = cmd.course_command
        
        return delta, self.commanded_state