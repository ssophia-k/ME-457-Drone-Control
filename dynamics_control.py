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
from Message_types.state import State
from Message_types.delta import Delta
import parameters as MAV
from Tools.rotations import quaternion_to_rotation, quaternion_to_euler

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

        # convert wind vector from world to body frame and add gust
        wind_body_frame = quaternion_to_rotation(self._state[6:10]) @ steady_state + gust

        # velocity vector relative to the airmass
        v_air = self._state[3:6] - wind_body_frame
        ur = v_air.item(0)
        vr = v_air.item(1)
        wr = v_air.item(2)

        # compute airspeed, Angle of Attack, and sideslip angle
        self._Va = np.linalg.norm(v_air)  # np.sqrt(ur**2+vr**2+wr**2)
        if ur == 0:
            self._alpha = 0
        else:
            self._alpha = np.arctan(wr / ur)
        if self._Va == 0:
            self._beta = 0
        else:
            self._beta = np.arcsin(vr / self._Va)


    def _forces_moments(self, delta):
        """
        return the forces on the UAV based on the state, wind, and control surfaces
        :param delta: np.matrix(delta_a, delta_e, delta_r, delta_t)
        :return: Forces and Moments on the UAV np.matrix(Fx, Fy, Fz, Ml, Mn, Mm)
        """
        phi, theta, psi = quaternion_to_euler(self._state[6:10])
        p, q, r = self._state[10:13]

        delta_a = delta.aileron
        delta_e = delta.elevator
        delta_r = delta.rudder
        delta_t = delta.throttle

        # compute gravitational forces
        fg_x = -MAV.mass * MAV.gravity * np.sin(theta)
        fg_y = MAV.mass * MAV.gravity * np.cos(theta) * np.sin(phi)
        fg_z = MAV.mass * MAV.gravity * np.cos(theta) * np.cos(phi)

        # compute Lift and Drag coefficients
        #C_L = MAV.C_L_0 + MAV.C_L_alpha * self._alpha
        #C_D = MAV.C_D_0 + MAV.C_D_alpha * self._alpha
        sigma = (1 + np.exp(-MAV.M * (self._alpha - MAV.alpha0)) + np.exp(MAV.M * (self._alpha + MAV.alpha0))) / ((1 + np.exp(-MAV.M * (self._alpha - MAV.alpha0))) * (1 + np.exp(MAV.M * (self._alpha + MAV.alpha0))))
        C_L = (1 - sigma) * (MAV.C_L_0 + MAV.C_L_alpha * self._alpha) + sigma * (2 * np.sign(self._alpha) * np.sin(self._alpha)**2 * np.cos(self._alpha))
        C_D = MAV.C_D_p + (MAV.C_L_0 + MAV.C_L_alpha * self._alpha)**2 / (np.pi * MAV.e * MAV.AR)
        C_X = -C_D*np.cos(self._alpha) + C_L*np.sin(self._alpha)
        C_X_q = -MAV.C_D_q*np.cos(self._alpha) + MAV.C_L_q*np.sin(self._alpha)
        C_X_delta_e = -MAV.C_D_delta_e*np.cos(self._alpha) + MAV.C_L_delta_e*np.sin(self._alpha)
        C_Z = -C_D*np.sin(self._alpha) - C_L*np.cos(self._alpha)
        C_Z_q = -MAV.C_D_q*np.sin(self._alpha) - MAV.C_L_q*np.cos(self._alpha)
        C_Z_delta_e = -MAV.C_D_delta_e*np.sin(self._alpha) - MAV.C_L_delta_e*np.cos(self._alpha)

        # propeller thrust and torque
        thrust_prop, torque_prop = self._motor_thrust_torque(self._Va, delta_t)

        # compute forces in body frame
        f_x = fg_x + 0.5*MAV.rho*self._Va**2*MAV.S_wing* ( C_X + C_X_q*(MAV.c*q)/(2*self._Va) + C_X_delta_e*delta_e) + thrust_prop
        f_y = fg_y+ 0.5*MAV.rho*self._Va**2*MAV.S_wing* ( MAV.C_Y_0 + MAV.C_Y_beta*self._beta + MAV.C_Y_p*(MAV.b*p)/(2*self._Va) + MAV.C_Y_r*(MAV.b*r)/(2*self._Va) + MAV.C_Y_delta_a*delta_a + MAV.C_Y_delta_r*delta_r)
        f_z = fg_z + 0.5*MAV.rho*self._Va**2*MAV.S_wing* ( C_Z + C_Z_q*(MAV.c*q)/(2*self._Va) + C_Z_delta_e*delta_e)

        # compute torques in body frame
        l = 0.5 * MAV.rho * self._Va**2 * MAV.S_wing * ( 
            MAV.b * (MAV.C_ell_0 + MAV.C_ell_beta*self._beta + MAV.C_ell_p * (MAV.b * p) / (2 * self._Va) 
                    + MAV.C_ell_r * (MAV.b * r) / (2 * self._Va) + MAV.C_ell_delta_a * delta_a 
                    + MAV.C_ell_delta_r * delta_r) ) - torque_prop
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
        a =(MAV.rho * MAV.D_prop**5 * MAV.C_Q0) / ((2 * np.pi)**2) 
        b = (MAV.rho * MAV.D_prop**4 / (2 * np.pi)) * MAV.C_Q1 * Va + MAV.KQ**2 / MAV.R_motor
        c = MAV.rho * MAV.D_prop**3 * MAV.C_Q2 * Va**2 - (MAV.KQ * v_in / MAV.R_motor) + MAV.KQ * MAV.i0
        omega_p = (-b + np.sqrt(b**2 - 4 * a * c)) / (2. * a)

        # thrust and torque due to propeller
        J = 2 * np.pi * Va / (omega_p * MAV.D_prop)
        C_T = MAV.C_T0 + MAV.C_T1 * J + MAV.C_T2 * J**2
        C_Q = MAV.C_Q0 + MAV.C_Q1 * J + MAV.C_Q2 * J**2
        n = omega_p / (2 * np.pi)
        thrust_prop = MAV.rho * n**2 * MAV.D_prop**4 * C_T
        torque_prop = MAV.rho * n**2 * MAV.D_prop**5 * C_Q

        return thrust_prop, torque_prop

    def _update_true_state(self):
        # rewrite this function because we now have more information
        phi, theta, psi = quaternion_to_euler(self._state[6:10])
        pdot = quaternion_to_rotation(self._state[6:10]) @ self._state[3:6]
        self.true_state.pn = self._state.item(0)
        self.true_state.pe = self._state.item(1)
        self.true_state.altitude = -self._state.item(2)
        self.true_state.Va = self._Va
        self.true_state.alpha = self._alpha
        self.true_state.beta = self._beta
        self.true_state.phi = phi
        self.true_state.theta = theta
        self.true_state.psi = psi
        self.true_state.Vg = np.linalg.norm(pdot)
        self.true_state.gamma = np.arcsin(pdot.item(2) / self.true_state.Vg)
        self.true_state.chi = np.arctan2(pdot.item(1), pdot.item(0))
        self.true_state.p = self._state.item(10)
        self.true_state.q = self._state.item(11)
        self.true_state.r = self._state.item(12)
        self.true_state.wn = self._wind.item(0)
        self.true_state.we = self._wind.item(1)
