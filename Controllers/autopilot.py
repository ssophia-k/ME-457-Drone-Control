"""
autopilot block for mavsim_python
    - Beard & McLain, PUP, 2012
    - Last Update:
        2/6/2019 - RWB
"""
import os, sys
# insert parent directory at beginning of python search path
from pathlib import Path
sys.path.insert(0,os.fspath(Path(__file__).parents[1]))
import numpy as np
import Parameters.control_parameters as AP
from Tools.transfer_function import TransferFunction
from Tools.wrap import wrap
from Controllers.pi_control import PIControl
from Controllers.pd_control_with_rate import PDControlWithRate
from Controllers.tf_control import TFControl
from Message_types.state import State
from Message_types.delta import Delta

"""
class Autopilot:
    def __init__(self, ts_control):
        # instantiate lateral-directional controllers
        self.roll_from_aileron = PDControlWithRate(
                        kp=AP.roll_kp,
                        kd=AP.roll_kd,
                        limit=np.radians(45))
        self.course_from_roll = PIControl(
                        kp=AP.course_kp,
                        ki=AP.course_ki,
                        Ts=ts_control,
                        limit=np.radians(30))
        #self.yaw_damper = TransferFunction(
        #                num=np.array([[AP.yaw_damper_kr, 0]]),
        #                den=np.array([[1, AP.yaw_damper_p_wo]]),
        #                Ts=ts_control)
        self.yaw_damper = TFControl(
                         k=AP.yaw_damper_kr,
                         n0=0.0,
                         n1=1.0,
                         d0=AP.yaw_damper_p_wo,
                         d1=1,
                         Ts=ts_control)

        # instantiate longitudinal controllers
        self.pitch_from_elevator = PDControlWithRate(
                        kp=AP.pitch_kp,
                        kd=AP.pitch_kd,
                        limit=np.radians(45))
        self.altitude_from_pitch = PIControl(
                        kp=AP.altitude_kp,
                        ki=AP.altitude_ki,
                        Ts=ts_control,
                        limit=np.radians(30))
        self.airspeed_from_throttle = PIControl(
                        kp=AP.airspeed_throttle_kp,
                        ki=AP.airspeed_throttle_ki,
                        Ts=ts_control,
                        limit=1.0)
        self.commanded_state = State()

    def update(self, cmd, state):
	
	#### TODO #####
        # lateral autopilot, updates our command inputs based on current state
        #chi_c is course command, phi_c is roll command, delta_a is aileron command, delta_r is rudder command
        chi_c = wrap(cmd.course_command, state.chi)
        phi_c = self.saturate(cmd.phi_feedforward +self.course_from_roll.update(chi_c, state.chi), 
                              -np.radians(30), 
                              np.radians(30))
        delta_a = self.roll_from_aileron.update(phi_c, state.phi, state.p)
        delta_r = self.yaw_damper.update(state.r)


        # longitudinal autopilot
        #h_c is altitude command, need to saturate
        h_c = self.saturate(cmd.altitude_command,
                            state.h - AP.altitude_zone,
                            state.h + AP.altitude_zone)
        theta_c = self.altitude_from_pitch.update(h_c, state.h)
        delta_e = self.pitch_from_elevator.update(theta_c, state.theta, state.q)
        delta_t = self.saturate(self.airspeed_from_throttle.update(cmd.airspeed_command, state.Va),
                                0.0, 1.0) 

        # construct control outputs and commanded states
        delta = Delta(elevator=0,
                         aileron=0,
                         rudder=0,
                         throttle=0)
        self.commanded_state.altitude = 0
        self.commanded_state.Va = 0
        self.commanded_state.phi = 0
        self.commanded_state.theta = 0
        self.commanded_state.chi = 0
        return delta, self.commanded_state

    def saturate(self, input, low_limit, up_limit):
        if input <= low_limit:
            output = low_limit
        elif input >= up_limit:
            output = up_limit
        else:
            output = input
        return output
"""


class Autopilot:
    def __init__(self, ts_control):
        # instantiate lateral controllers
        self.roll_from_aileron = PDControlWithRate(
                        kp=AP.roll_kp,
                        kd=AP.roll_kd,
                        limit=np.radians(45))
        self.course_from_roll = PIControl(
                        kp=AP.course_kp,
                        ki=AP.course_ki,
                        Ts=ts_control,
                        limit=np.radians(30))
        self.yaw_damper = TransferFunction(
                        num=np.array([[AP.yaw_damper_kr, 0]]),
                        den=np.array([[1, AP.yaw_damper_p_wo]]),
                        Ts=ts_control)

        # instantiate lateral controllers
        self.pitch_from_elevator = PDControlWithRate(
                        kp=AP.pitch_kp,
                        kd=AP.pitch_kd,
                        limit=np.radians(45))
        self.altitude_from_pitch = PIControl(
                        kp=AP.altitude_kp,
                        ki=AP.altitude_ki,
                        Ts=ts_control,
                        limit=np.radians(30))
        self.airspeed_from_throttle = PIControl(
                        kp=AP.airspeed_throttle_kp,
                        ki=AP.airspeed_throttle_ki,
                        Ts=ts_control,
                        limit=1.0)
        self.commanded_state = State()

    def update(self, cmd, state):

        # lateral autopilot
        chi_c = wrap (cmd.course_command , state.chi )
        phi_c_unsaturated = self.course_from_roll.update(chi_c, state.chi)
        phi_c_limit = np.pi / 4
        phi_c = self.saturate(phi_c_unsaturated, -phi_c_limit, phi_c_limit)
        delta_a = self.roll_from_aileron.update(phi_c, state.phi, state.p)
        delta_r = self.yaw_damper.update(state.r)

        # longitudinal autopilot
        # saturate the altitude command
        h_c = self.saturate(cmd.altitude_command, state.altitude - AP.altitude_zone, state.altitude + AP.altitude_zone)
        theta_c = self.altitude_from_pitch.update(h_c, state.altitude)
        delta_e = self.pitch_from_elevator.update(theta_c, state.theta, state.q)
        delta_t_unsat = self.airspeed_from_throttle.update(cmd.airspeed_command, state.Va)
        delta_t = self.saturate(delta_t_unsat, 0, 1.0)

        # construct output and commanded states
        delta = Delta(elevator=delta_e,
                         aileron=delta_a,
                         rudder=delta_r,
                         throttle=delta_t)
        self.commanded_state.altitude = cmd.altitude_command
        self.commanded_state.Va = cmd.airspeed_command
        self.commanded_state.phi = phi_c
        self.commanded_state.theta = theta_c
        self.commanded_state.chi = cmd.course_command
        return delta, self.commanded_state

    def saturate(self, input, low_limit, up_limit):
        if input <= low_limit:
            output = low_limit
        elif input >= up_limit:
            output = up_limit
        else:
            output = input
        return output