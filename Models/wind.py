"""
Class to determine wind velocity at any given moment,
calculates a steady wind speed and uses a stochastic
process to represent wind gusts. (Follows section 4.4 in uav book)
"""
from Tools.transfer_function import TransferFunction
import numpy as np


class WindSimulation:
    def __init__(self, Ts, gust_flag = False, steady_state = np.array([[0., 0., 0.]]).T):
        # steady state wind defined in the inertial frame, gust defined in the body frame
        self._steady_state = steady_state
        V_a0 = 25  # m/s

        if gust_flag:
            # Dryden gust model parameters
            # Low Altitude (50m), Light Turbulence
            L_u = L_v = 200
            L_w = 50
            sigma_u=sigma_v = 1.06
            sigma_w = 0.7

            # Low Altitude (50m), Moderate Turbulence
            #L_u = L_v = 200
            #L_w = 50
            #sigma_u=sigma_v = 2.12
            #sigma_w = 1.4

            # Med Altitude (600m), Light Turbulence
            #L_u = L_v = 533
            #L_w = 533
            #sigma_u=sigma_v = 1.5
            #sigma_w = 1.5

            # Med Altitude (600m), Moderate Turbulence
            #L_u = L_v = 533
            #L_w = 533
            #sigma_u=sigma_v = 3.0
            #sigma_w = 3.0

            # Dryden transfer functions
            self.u_w = TransferFunction(num=np.array([[sigma_u * np.sqrt(2.0 * V_a0 / (np.pi * L_u))]]), den=np.array([[1,V_a0/L_u]]),Ts=Ts)
            self.v_w = TransferFunction(num=np.array([[sigma_v * np.sqrt(3.0 * V_a0 / (np.pi * L_v)),sigma_v * np.sqrt(3.0 * V_a0 / (np.pi * L_v)) * (V_a0 / (np.sqrt(3.0) * L_v))]]), den=np.array([[1,2 * (V_a0 / L_v), (V_a0 / L_v) ** 2]]),Ts=Ts)
            self.w_w = TransferFunction(num=np.array([[sigma_w * np.sqrt(3.0 * V_a0 / (np.pi * L_w)),sigma_w * np.sqrt(3.0 * V_a0 / (np.pi * L_w)) * (V_a0 / (np.sqrt(3.0) * L_w))]]), den=np.array([[1,2 * (V_a0 / L_w), (V_a0 / L_w) ** 2]]),Ts=Ts)
            self._Ts = Ts
        
        else:
            self.u_w = TransferFunction(num=np.array([[0]]), den=np.array([[1,1]]),Ts=Ts)
            self.v_w = TransferFunction(num=np.array([[0,0]]), den=np.array([[1,1,1]]),Ts=Ts)
            self.w_w = TransferFunction(num=np.array([[0,0]]), den=np.array([[1,1,1]]),Ts=Ts)
            self._Ts = Ts


    def update(self):
        # returns a six vector.
        #   The first three elements are the steady state wind in the inertial frame
        #   The second three elements are the gust in the body frame
        gust = np.array([[self.u_w.update(np.random.randn())],
                         [self.v_w.update(np.random.randn())],
                         [self.w_w.update(np.random.randn())]])
        return np.concatenate(( self._steady_state, gust ))
