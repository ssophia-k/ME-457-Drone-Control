"""
mavDynamics 
    - this file implements the dynamic equations of motion for MAV
"""

import numpy as np
import parameters as MAV
from Message_types.state import State
from Tools.rotations import quaternion_to_rotation, quaternion_to_euler


class Dynamics:

    def __init__(self, Ts):
        self._ts_simulation = Ts
        # set initial states based on parameter file
        # _state is the 13x1 internal state of the aircraft that is being propagated:
        # _state = [pn, pe, pd, u, v, w, e0, e1, e2, e3, p, q, r]
        # We will also need a variety of other elements that are functions of the _state and the wind.
        # self.true_state is a 19x1 vector that is estimated and used by the autopilot to control the aircraft:
        # true_state = [pn, pe, h, Va, alpha, beta, phi, theta, chi, p, q, r, Vg, wn, we, psi]
        self._state = np.array([
            [MAV.pn_0],   #(0)
            [MAV.pe_0],   #(1)
            [MAV.pd_0],   #(2)
            [MAV.u_0],    #(3)
            [MAV.v_0],    #(4)  
            [MAV.w_0],    #(5)
            [MAV.e_0],    #(6)
            [MAV.e_1],    #(7)
            [MAV.e_2],    #(8)
            [MAV.e_3],    #(9)
            [MAV.p_0],    #(10)  
            [MAV.q_0],    #(11) 
            [MAV.r_0],    #(12)
        ])
        self.true_state = State()
        

    ##############################
    #Public Functions
    def update(self, forces_moments):
        '''
            Integrate the differential equations defining dynamics, update sensors
            delta = (delta_a, delta_e, delta_r, delta_t) are the control inputs
            wind is the wind vector in inertial coordinates
            Ts is the time step between function calls.
        '''
        self._rk4_step(forces_moments)
        # update the message class for the true state
        self._update_true_state()
    
    def external_set_state(self, new_state):
        self._state = new_state

    ##############################
    #Private Functions
    def _rk4_step(self, forces_moments):
        time_step = self._ts_simulation
        k1 = self._f(self._state, forces_moments)
        k2 = self._f(self._state + time_step/2.*k1, forces_moments)
        k3 = self._f(self._state + time_step/2.*k2, forces_moments)
        k4 = self._f(self._state + time_step*k3, forces_moments)
        self._state += time_step/6 * (k1 + 2*k2 + 2*k3 + k4)

        # normalize the quaternion
        e0 = self._state.item(6)
        e1 = self._state.item(7)
        e2 = self._state.item(8)
        e3 = self._state.item(9)
        normE = (e0**2+e1**2+e2**2+e3**2)**0.5
        self._state[6][0] = self._state.item(6)/normE
        self._state[7][0] = self._state.item(7)/normE
        self._state[8][0] = self._state.item(8)/normE
        self._state[9][0] = self._state.item(9)/normE


    def _f(self, state, forces_moments):
        """
        for the dynamics xdot = f(x, u), returns f(x, u)
        """
        pn, pe, pd, u, v, w, e0, e1, e2, e3, p, q, r = state[0:13]
        f_x, f_y, f_z, l, m, n = forces_moments.flatten()


        # Position Kinematics
        pos_dot = quaternion_to_rotation(np.array([e0, e1, e2, e3])) @ np.array([u, v, w])
        pn_dot = pos_dot[0]
        pe_dot = pos_dot[1]
        pd_dot = pos_dot[2]

        # Position Dynamics
        u_dot = r*v - q*w + f_x/MAV.mass
        v_dot = p*w - r*u + f_y/MAV.mass
        w_dot = q*u - p*v + f_z/MAV.mass

        #Rotational kinematics
        e0_dot = 0.5 * (-p*e1 - q*e2 - r*e3)
        e1_dot = 0.5 * ( p*e0 + r*e2 - q*e3)
        e2_dot = 0.5 * ( q*e0 - r*e1 + p*e3)
        e3_dot = 0.5 * ( r*e0 + q*e1 - p*e2)

        #Rotational dynamics
        gamma = (MAV.J_x*MAV.J_z) - (MAV.J_xz**2)
        gamma_1 = (MAV.J_xz*(MAV.J_x - MAV.J_y + MAV.J_z))/gamma
        gamma_2 = (MAV.J_z*(MAV.J_z - MAV.J_y) + MAV.J_xz**2)/gamma
        gamma_3 = MAV.J_z/gamma
        gamma_4 = MAV.J_xz/gamma
        gamma_5 = (MAV.J_z - MAV.J_x)/MAV.J_y
        gamma_6 = MAV.J_xz/MAV.J_y
        gamma_7 = ((MAV.J_x - MAV.J_y)*MAV.J_x + MAV.J_xz**2)/gamma
        gamma_8 = MAV.J_x/gamma

        angular_acceleration_body = np.array([
        gamma_1*p*q - gamma_2*q*r + gamma_3*l + gamma_4*n,
        gamma_5*p*r - gamma_6*((p**2) - (r**2)) + m/MAV.J_y,
        gamma_7*p*q - gamma_1*q*r + gamma_4*l + gamma_8*n
        ])

        p_dot = angular_acceleration_body[0]
        q_dot = angular_acceleration_body[1]
        r_dot = angular_acceleration_body[2]

        #Return x-dot
        return np.array([pn_dot, pe_dot, pd_dot, u_dot, v_dot, w_dot, e0_dot, e1_dot, e2_dot, e3_dot, p_dot, q_dot, r_dot])

    def _update_true_state(self):
        # update the class structure for the true state:
        #   [pn, pe, h, Va, alpha, beta, phi, theta, chi, p, q, r, Vg, wn, we, psi, gyro_bx, gyro_by, gyro_bz]
        phi, theta, psi = quaternion_to_euler(self._state[6:10])
        self.true_state.pn = self._state.item(0)
        self.true_state.pe = self._state.item(1)
        self.true_state.altitude = -self._state.item(2)
        self.true_state.Va = 0
        self.true_state.alpha = 0
        self.true_state.beta = 0
        self.true_state.phi = phi
        self.true_state.theta = theta
        self.true_state.psi = psi
        self.true_state.Vg = 0
        self.true_state.gamma = 0
        self.true_state.chi = 0
        self.true_state.p = self._state.item(10)
        self.true_state.q = self._state.item(11)
        self.true_state.r = self._state.item(12)
        self.true_state.wn = 0
        self.true_state.we = 0
