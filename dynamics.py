import numpy as np
import parameters as MAV
from state import State


class Dynamics:

    def __init__(self, Ts):
        self._ts_simulation = Ts
        self._state = np.array([
            [MAV.pn_0],  
            [MAV.pe_0],   
            [MAV.pd_0],   
            [MAV.u_0],     
            [MAV.v_0],     
            [MAV.w_0],   
            [MAV.p_0],      
            [MAV.q_0],      
            [MAV.r_0],      
            [MAV.phi_0],   
            [MAV.theta_0], 
            [MAV.psi_0],   
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
        dt = self._ts_simulation
        k1 = self._f(self._state.flatten(), forces_moments)
        k2 = self._f((self._state.flatten() + dt/2 * k1), forces_moments)
        k3 = self._f((self._state.flatten() + dt/2 * k2), forces_moments)
        k4 = self._f((self._state.flatten() + dt * k3), forces_moments)

        self._state = self._state.flatten() + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        self._state = self._state.reshape((12, 1))  # Ensure _state retains (12,1) shape


    def _f(self, state, forces_moments):
        pn, pe, pd, u, v, w, p, q, r, phi, theta, psi, = state.flatten()
        f_x, f_y, f_z, l, m, n = forces_moments.flatten()
    
        c_theta = np.cos(theta)
        s_theta = np.sin(theta)
        c_phi = np.cos(phi)
        s_phi = np.sin(phi)
        c_psi = np.cos(psi)
        s_psi = np.sin(psi)

        pn_dot = (c_theta*c_psi)*u + (s_phi*s_theta*c_psi - c_phi*s_psi)*v + (c_phi*s_theta*c_psi + s_phi*s_psi)*w
        pe_dot = (c_theta*s_psi)*u + (s_phi*s_theta*s_psi + c_phi*c_psi)*v + (c_phi*s_theta*s_psi - s_phi*c_psi)*w
        pd_dot = -(s_theta)*u + (s_phi*c_theta)*v + (c_phi*c_theta)*w

        u_dot = r*v - q*w + f_x/MAV.mass
        v_dot = p*w - r*u + f_y/MAV.mass
        w_dot = q*u - p*v + f_z/MAV.mass

        angular_velocity_body = np.array([p, q, r])

        angular_matrix = np.array([
        [1, s_phi*np.tan(theta), c_phi*np.tan(theta)],
        [0, c_phi, -s_phi],
        [0, s_phi/c_theta, c_phi/c_theta ]
        ])

        angular_acceleration_euler = angular_matrix @ angular_velocity_body

        phi_dot = angular_acceleration_euler[0]
        theta_dot = angular_acceleration_euler[1]
        psi_dot = angular_acceleration_euler[2]

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

        return np.array([pn_dot, pe_dot, pd_dot, u_dot, v_dot, w_dot, p_dot, q_dot, r_dot, phi_dot, theta_dot, psi_dot])

    def _update_true_state(self):
        self.true_state.pn = self._state.item(0)
        self.true_state.pe = self._state.item(1)
        self.true_state.pd = -self._state.item(2)
        self.true_state.Va = 0
        self.true_state.alpha = 0
        self.true_state.beta = 0
        self.true_state.phi = self._state.item(9)
        self.true_state.theta = self._state.item(10)
        self.true_state.psi = self._state.item(11)
        self.true_state.p = self._state.item(6)
        self.true_state.q = self._state.item(7)
        self.true_state.r = self._state.item(8)


