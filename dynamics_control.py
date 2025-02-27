"""
mavDynamics 
    - this file implements the dynamic equations of motion for MAV
    - use unit quaternion for the attitude state
    
mavsim_python
    - Beard & McLain, PUP, 2012
    - Update history:  
        2/24/2020 - RWB
"""
import numpy as np
from dynamics import Dynamics as DynamicsForces
# load message types
from state import State
from delta import Delta
import parameters as MAV


class MavDynamics(DynamicsForces):
    def __init__(self, Ts):
        super().__init__(Ts)
        # store wind data for fast recall since it is used at various points in simulation
        self._wind = np.array([[0.], [0.], [0.]])  # wind in NED frame in meters/sec
        # store forces to avoid recalculation in the sensors function
        self._forces = np.array([[0.], [0.], [0.]])
        self._Va = MAV.u_0
        self._alpha = 0
        self._beta = 0
        # update velocity data and forces and moments
        self._update_velocity_data()
        self._forces_moments(delta=Delta())
        # update the message class for the true state
        self._update_true_state()


    ###################################
    # public functions
    def update(self, delta, wind):
        '''
            Integrate the differential equations defining dynamics, update sensors
            delta = (delta_a, delta_e, delta_r, delta_t) are the control inputs
            wind is the wind vector in inertial coordinates
            Ts is the time step between function calls.
        '''
        # get forces and moments acting on rigid bod
        forces_moments = self._forces_moments(delta)
        super()._rk4_step(forces_moments)
        # update the airspeed, angle of attack, and side slip angles using new state
        self._update_velocity_data(wind)
        # update the message class for the true state
        self._update_true_state()

    ###################################
    # private functions
    def _update_velocity_data(self, wind=np.zeros((6, 1))):
        steady_state = wind[0:3]  # in NED
        gust = wind[3:6]  # in body frame

        phi = self._state.item(9)
        theta = self._state.item(10)
        psi = self._state.item(11)

        # rotation matrix from body to NED
        R = np.array([
            [np.cos(theta) * np.cos(psi), np.cos(theta) * np.sin(psi), -np.sin(theta)],
            [np.sin(phi) * np.sin(theta) * np.cos(psi) - np.cos(phi) * np.sin(psi), 
             np.sin(phi) * np.sin(theta) * np.sin(psi) + np.cos(phi) * np.cos(psi), 
             np.sin(phi) * np.cos(theta)],
            [np.cos(phi) * np.sin(theta) * np.cos(psi) + np.sin(phi) * np.sin(psi), 
             np.cos(phi) * np.sin(theta) * np.sin(psi) - np.sin(phi) * np.cos(psi), 
             np.cos(phi) * np.cos(theta)]])  

        # convert steady-state wind vector from NED to body frame
        wind_body_steady = R.T @ steady_state

        # add the gust
        wind_body = wind_body_steady + gust

        # convert total wind to NED frame  
        self._wind = R @ wind_body             

        u, v, w = self._state[3:6]  # velocity in body frame

        # velocity vector relative to air in body frame
        ur = u - wind_body[0]
        vr = v - wind_body[1]
        wr = w - wind_body[2]

        # compute airspeed, Angle of Attack, and sideslip angle
        self._Va = np.sqrt(ur**2 + vr**2 + wr**2)
        self._alpha = np.arctan2(wr, ur)
        self._beta = np.arcsin(vr / self._Va)

    def _forces_moments(self, delta):
        """
        return the forces on the UAV based on the state, wind, and control surfaces
        :param delta: np.matrix(delta_a, delta_e, delta_r, delta_t)
        :return: Forces and Moments on the UAV np.matrix(Fx, Fy, Fz, Ml, Mn, Mm)
        """
        phi, theta, psi = self._state[6:9]
        p, q, r = self._state[9:12]

        delta_a = delta.aileron
        delta_e = delta.elevator
        delta_r = delta.rudder
        delta_t = delta.throttle

        # compute gravitational forces
        fg_ned = np.array([[0], [0], [-MAV.mass * MAV.gravity]])
        R = np.array([
            [np.cos(theta) * np.cos(psi), np.cos(theta) * np.sin(psi), -np.sin(theta)],
            [np.sin(phi) * np.sin(theta) * np.cos(psi) - np.cos(phi) * np.sin(psi), 
             np.sin(phi) * np.sin(theta) * np.sin(psi) + np.cos(phi) * np.cos(psi), 
             np.sin(phi) * np.cos(theta)],
            [np.cos(phi) * np.sin(theta) * np.cos(psi) + np.sin(phi) * np.sin(psi), 
             np.cos(phi) * np.sin(theta) * np.sin(psi) - np.sin(phi) * np.cos(psi), 
             np.cos(phi) * np.cos(theta)]
        ])  # rotation matrix from body to NED
        fg_body = R.T @ fg_ned
        fg_x, fg_y, fg_z = fg_body.flatten()

        # compute Lift and Drag coefficients
        C_L = MAV.C_L_0 + MAV.C_L_alpha * self._alpha
        C_D = MAV.C_D_0 + MAV.C_D_alpha * self._alpha
        C_X = -C_D*np.cos(self._alpha) + C_L*np.sin(self._alpha)
        C_X_q = -MAV.C_D_q*np.cos(self._alpha) + MAV.C_L_q*np.sin(self._alpha)
        C_X_delta_e = -MAV.C_D_delta_e*np.cos(self._alpha) + MAV.C_L_delta_e*np.sin(self._alpha)
        C_Z = -C_D*np.sin(self._alpha) - C_L*np.cos(self._alpha)
        C_Z_q = -MAV.C_D_q*np.sin(self._alpha) - MAV.C_L_q*np.cos(self._alpha)
        C_Z_delta_e = -MAV.C_D_delta_e*np.sin(self._alpha) - MAV.C_L_delta_e*np.cos(self._alpha)

        # propeller thrust and torque
        thrust_prop, torque_prop = self._motor_thrust_torque(self._Va, delta_t)

        # compute forces in body frame
        f_x = fg_x + 0.5*MAV.rho*self._Va**2*MAV.S_wing* ( C_X+C_X_q*(MAV.c*q)/(2*self._Va)+C_X_delta_e*delta_e) + thrust_prop
        f_y = fg_y+ 0.5*MAV.rho*self._Va**2*MAV.S_wing* ( MAV.C_Y_0 + MAV.C_Y_beta*self._beta + MAV.C_Y_p*(MAV.b*p)/(2*self._Va) + MAV.C_Y_r*(MAV.b*r)/(2*self._Va) + MAV.C_Y_delta_a*delta_a + MAV.C_Y_delta_r*delta_r)
        f_z = fg_z + 0.5*MAV.rho*self._Va**2*MAV.S_wing* ( C_Z + C_Z_q*(MAV.c*q)/(2*self._Va) + C_Z_delta_e*delta_e)

        # compute torques in body frame
        l = 0.5 * MAV.rho * self._Va**2 * MAV.S_wing * ( 
            MAV.b * (MAV.C_ell_0 + MAV.C_ell_beta + MAV.C_ell_p * (MAV.b * p) / (2 * self._Va) 
                    + MAV.C_ell_r * (MAV.b * r) / (2 * self._Va) + MAV.C_ell_delta_a * delta_a 
                    + MAV.C_ell_delta_r * delta_r) ) + torque_prop
        m = 0.5 * MAV.rho * self._Va**2 * MAV.S_wing * (
            MAV.c * (MAV.C_m_0 + MAV.C_m_alpha * self._alpha + MAV.C_m_q * (MAV.c * q) / (2 * self._Va) 
                    + MAV.C_m_delta_e * delta_e))
        n = 0.5 * MAV.rho * self._Va**2 * MAV.S_wing * (
            MAV.b * (MAV.C_n_0 + MAV.C_n_beta * self._beta + MAV.C_n_p * (MAV.b * p) / (2 * self._Va) 
                    + MAV.C_n_r * (MAV.b * r) / (2 * self._Va) + MAV.C_n_delta_a * delta_a 
                    + MAV.C_n_delta_r * delta_r))
   
        forces_moments = np.array([[f_x, f_y, f_z, l, m, n]]).T
        return forces_moments

    def _motor_thrust_torque(self, Va, delta_t):
        # compute thrust and torque due to propeller
        # map delta_t throttle command(0 to 1) into motor input voltage
        v_in = MAV.V_max * delta_t

        # angular speed of propeller
        prop_a = MAV.rho * MAV.D_prop**5 / ((2 * np.pi)**2) * MAV.C_Q0
        prop_b = (MAV.rho * MAV.D_prop**4 / (2 * np.pi)) * MAV.C_Q1 * Va + MAV.KQ**2 / MAV.R_motor
        prop_c = MAV.rho * MAV.D_prop**3 * MAV.C_Q2 * Va**2 - (MAV.KQ * v_in / MAV.R_motor) + MAV.KQ * MAV.i0
        omega_p = (-prop_b + np.sqrt(prop_b**2 - 4 * prop_a * prop_c)) / (2 * prop_a)

        # thrust and torque due to propeller
        C_T = MAV.C_T0 + MAV.C_T1 * (2 * np.pi * omega_p / Va) + MAV.C_T2 * (2 * np.pi * omega_p / Va)**2
        C_Q = MAV.C_Q0 + MAV.C_Q1 * (2 * np.pi * omega_p / Va) + MAV.C_Q2 * (2 * np.pi * omega_p / Va)**2
        thrust_prop = MAV.rho * (omega_p / (2 * np.pi))**2 * MAV.D_prop**4 * C_T
        torque_prop = MAV.rho * (omega_p / (2 * np.pi))**2 * MAV.D_prop**5 * C_Q

        return thrust_prop, torque_prop

    def _update_true_state(self):
        self.true_state.pn = self._state.item(0)
        self.true_state.pe = self._state.item(1)
        self.true_state.pd = -self._state.item(2)
        self.true_state.Va = self._Va
        self.true_state.alpha = self._alpha
        self.true_state.beta = self._beta
        self.true_state.phi = self._state.item(9)
        self.true_state.theta = self._state.item(10)
        self.true_state.psi = self._state.item(11)
        self.true_state.p = self._state.item(6)
        self.true_state.q = self._state.item(7)
        self.true_state.r = self._state.item(8)
