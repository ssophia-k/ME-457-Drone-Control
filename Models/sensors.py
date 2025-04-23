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
from Message_types.sensors import MsgSensors
import Parameters.parameters as MAV
import Parameters.sensor_parameters as SENSOR
from Models.dynamics_control import MavDynamics as MavDynamicsNoSensors
from Tools.rotations import quaternion_to_rotation, quaternion_to_euler, euler_to_rotation

class MavDynamics(MavDynamicsNoSensors):
    def __init__(self, Ts):
        super().__init__(Ts)
        # initialize the sensors message
        self._sensors = MsgSensors()
        # random walk parameters for GPS
        self._gps_eta_n = 0.
        self._gps_eta_e = 0.
        self._gps_eta_h = 0.
        # timer so that gps only updates every ts_gps seconds
        self._t_gps = 999.  # large value ensures gps updates at initial time.

    def sensors(self):
        "Return value of sensors on MAV: gyros, accels, absolute_pressure, dynamic_pressure, GPS"
       
        # simulate rate gyros(units are rad / sec)
        self._sensors.gyro_x = self._state.item(10) + SENSOR.gyro_x_bias + np.random.normal(0, SENSOR.gyro_sigma)
        self._sensors.gyro_y = self._state.item(11) + SENSOR.gyro_y_bias + np.random.normal(0, SENSOR.gyro_sigma)
        self._sensors.gyro_z = self._state.item(12) + SENSOR.gyro_z_bias + np.random.normal(0, SENSOR.gyro_sigma)

        # simulate accelerometers(units of g)
        gravity = 9.81
        forces_body = self._forces_moments()[0:3]
        phi, theta, psi = quaternion_to_euler(self._state[6:10])
        R = euler_to_rotation(phi, theta, psi)
        g_vector = np.array([[0.0], [0.0], [gravity]])
        f_gravity = R.T @ g_vector
        accel = forces_body / MAV.mass - f_gravity
        self._sensors.accel_x = accel.item(0) / gravity + np.random.normal(0, SENSOR.accel_sigma)
        self._sensors.accel_y = accel.item(1) / gravity + np.random.normal(0, SENSOR.accel_sigma)
        self._sensors.accel_z = accel.item(2) / gravity + np.random.normal(0, SENSOR.accel_sigma)

        # simulate magnetometers
        # magnetic field in provo has magnetic declination of 12.5 degrees
        # and magnetic inclination of 66 degrees
        mag_declination = np.radians(12.5)
        mag_inclination = np.radians(66.0)
        mag_strength = 1.0
        mag_field = mag_strength * np.array([
            [np.cos(mag_inclination) * np.cos(mag_declination)],
            [np.cos(mag_inclination) * np.sin(mag_declination)],
            [np.sin(mag_inclination)]
        ])
        # convert to body frame
        R = euler_to_rotation(phi, theta, psi)
        mag_body = R.T @ mag_field
        self._sensors.mag_x = mag_body.item(0) + np.random.normal(0, SENSOR.mag_sigma)
        self._sensors.mag_y = mag_body.item(1) + np.random.normal(0, SENSOR.mag_sigma)
        self._sensors.mag_z = mag_body.item(2) + np.random.normal(0, SENSOR.mag_sigma)

        # simulate pressure sensors
        rho = MAV.rho
        h = -self._state.item(2)
        self._sensors.abs_pressure = rho * gravity * h + np.random.normal(0, SENSOR.abs_pres_sigma)
        # differential pressure
        self._sensors.diff_pressure = 0.5 * rho * self._Va**2 + np.random.normal(0, SENSOR.diff_pres_sigma)
        
        # simulate GPS sensor
        if self._t_gps >= SENSOR.ts_gps:
            self._gps_eta_n = np.exp(-SENSOR.gps_beta * SENSOR.ts_gps) * self._gps_eta_n + np.random.normal(0, SENSOR.gps_n_sigma)
            self._gps_eta_e = np.exp(-SENSOR.gps_beta * SENSOR.ts_gps) * self._gps_eta_e + np.random.normal(0, SENSOR.gps_e_sigma)
            self._gps_eta_h = np.exp(-SENSOR.gps_beta * SENSOR.ts_gps) * self._gps_eta_h + np.random.normal(0, SENSOR.gps_h_sigma)
            self._sensors.gps_n = self._state.item(0) + self._gps_eta_n
            self._sensors.gps_e = self._state.item(1) + self._gps_eta_e
            self._sensors.gps_h = -self._state.item(2) + self._gps_eta_h

            pdot = quaternion_to_rotation(self._state[6:10]) @ self._state[3:6]
            Vg = np.linalg.norm(pdot)
            self._sensors.gps_Vg = Vg + np.random.normal(0, SENSOR.gps_Vg_sigma)

            chi = np.arctan2(pdot.item(1), pdot.item(0))
            self._sensors.gps_course = chi + np.random.normal(0, SENSOR.gps_course_sigma)
            self._t_gps = 0.
        else:
            self._t_gps += self._ts_simulation
        return self._sensors

    def external_set_state(self, new_state):
        self._state = new_state

    def _update_true_state(self):
        # update the class structure for the true state:
        #   [pn, pe, h, Va, alpha, beta, phi, theta, chi, p, q, r, Vg, wn, we, psi, gyro_bx, gyro_by, gyro_bz]
        phi, theta, psi = quaternion_to_euler(self._state[6:10])
        pdot = quaternion_to_rotation(self._state[6:10]) @ self._state[3:6]
        self.true_state.north = self._state.item(0)
        self.true_state.east = self._state.item(1)
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
        self.true_state.bx = SENSOR.gyro_x_bias
        self.true_state.by = SENSOR.gyro_y_bias
        self.true_state.bz = SENSOR.gyro_z_bias
        # self.true_state.camera_az = self._state.item(13)
        # self.true_state.camera_el = self._state.item(14)