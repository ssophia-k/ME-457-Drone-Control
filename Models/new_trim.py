"""
compute_trim 
    - Chapter 5 assignment for Beard & McLain, PUP, 2012
    - Update history:  
        12/29/2018 - RWB
"""

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from scipy.optimize import minimize
from Tools.rotations import euler_to_quaternion, quaternion_to_euler
from Message_types.delta import Delta
import time
from Models.dynamics_control import MavDynamics
import Parameters.simulation_parameters as SIM



def compute_trim(mav, Va, gamma):
    # define initial state and input
    
    # set the initial conditions of the optimization
    e0 = euler_to_quaternion(0, gamma, 0)
    state0 = np.array([[0],  # pn
                   [0],  # pe
                   [-120],  # pd
                   [Va],  # u
                   [0.], # v
                   [0.], # w
                   [e0.item(0)],  # e0
                   [e0.item(1)],  # e1
                   [e0.item(2)],  # e2
                   [e0.item(3)],  # e3
                   [0.], # p
                   [0.], # q
                   [0.]  # r
                   ])
    delta0 = np.array([[-0.02],  # elevator
                       [0],  # aileron
                       [0],  # rudder
                       [0.3]]) # throttle
    x0 = np.concatenate((state0, delta0), axis=0).flatten()
    # define equality constraints
    cons = ({'type': 'eq',
            'fun': lambda x: np.array([
                    x[3]**2 + x[4]**2 + x[5]**2 - Va**2, # magnitude of velocity
                    # vector is Va
                    x[4], # v=0, force side velocity to be zero
                    x[6]**2 + x[7]**2 + x[8]**2 + x[9]**2 - 1., # quaternion is unit length
                    x[7], # e1=0 - forcing e1=e3=0 ensures zero roll and zero yaw in trim
                    x[9], # e3=0
                    x[10], # p=0 - angular rates should all be zero
                    x[11], # q=0
                    x[12], # r=0
                    ]),

            'jac': lambda x: np.array([
                    [0., 0., 0., 2*x[3], 2*x[4], 2*x[5], 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 2*x[6], 2*x[7], 2*x[8], 2*x[9], 0., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                    ])
            })
    # solve the minimization problem to find the trim states and inputs

    res = minimize(trim_objective_fun, x0, method='SLSQP', args=(mav, Va, gamma),
                   constraints=cons, 
                   options={'ftol': 1e-10, 'disp': True})
    # extract trim state and input and return
    trim_state = np.array([res.x[0:13]]).T
    trim_input = Delta(elevator=res.x.item(13),
                          aileron=res.x.item(14),
                          rudder=res.x.item(15),
                          throttle=res.x.item(16))
    trim_input.print()
    print('trim_state=', trim_state.T)
    return trim_state, trim_input


def trim_objective_fun(x, mav, Va, gamma):
    # Extract state and control inputs from the optimization vector
    state = np.array([x[0:13]]).T
    delta = Delta(elevator=x.item(13),
                 aileron=x.item(14),
                 rudder=x.item(15),
                 throttle=x.item(16))
    
    # Calculate quaternion values and extract psi
    phi = 0.0  # desired roll angle (0 for straight flight)
    theta = gamma  # desired pitch angle (equal to flight path angle for straight flight)
    psi = 0.0  # desired heading angle
    
    # Calculate desired state derivative for steady flight
    desired_trim_state_dot = np.array([
        [Va * np.cos(theta) * np.cos(psi)],  # pn_dot
        [Va * np.cos(theta) * np.sin(psi)],  # pe_dot
        [-Va * np.sin(theta)],               # pd_dot (negative for NED coordinates)
        [0.0],                               # u_dot
        [0.0],                               # v_dot
        [0.0],                               # w_dot
        [0.0],                               # e0_dot
        [0.0],                               # e1_dot
        [0.0],                               # e2_dot
        [0.0],                               # e3_dot
        [0.0],                               # p_dot
        [0.0],                               # q_dot
        [0.0]                                # r_dot
    ])
    
    # Calculate quaternion derivatives if needed
    # (This would be more complex, matching how quaternions change with angular rates)
    # For straight flight with zero angular rates, quaternion derivatives should be zero
    
    # Set the state for force/moment calculations
    mav._state = state
    mav._update_velocity_data()
    forces_moments = mav._forces_moments(delta)
    f = mav._f(state, forces_moments)
    
    # Calculate error between desired and actual state derivatives
    # Excluding the first three position derivatives (which aren't constrained)
    tmp = desired_trim_state_dot - f
    J = np.linalg.norm(tmp[2:13])**2.0
    return J


# Initialize the MAV model
mav = MavDynamics(SIM.ts_simulation)

# Define trim conditions
Va = 25.0  # desired airspeed (m/s)
gamma = 0.0  # desired flight path angle (radians), 0 for level flight

# Compute trim conditions
print("Computing trim conditions...")
trim_state, trim_input = compute_trim(mav, Va, gamma)