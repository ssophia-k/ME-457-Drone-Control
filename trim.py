"""
compute_trim 
    - Chapter 5 assignment for Beard & McLain, PUP, 2012
    - Update history:  
        12/29/2018 - RWB
"""
import numpy as np
from scipy.optimize import minimize
from Tools.rotations import euler_to_quaternion
from Message_types.delta import Delta
import time

def compute_trim(mav, Va, gamma):
    # define initial state and input
    
    # set the initial conditions of the optimization
    phi_0 = 0.0
    theta_0 = gamma
    psi_0 = 0.0
    state0 = np.array([[0],  # pn
                   [0],  # pe
                   [-10],  # pd
                   [Va],  # u
                   [0.], # v
                   [0.], # w
                   [phi_0],  # phi_0
                   [theta_0],  # theta_0
                   [psi_0],  # psi_0
                   [0.], # p
                   [0.], # q
                   [0.]  # r
                   ])
    delta0 = np.array([[0],  # elevator
                       [0],  # aileron
                       [0],  # rudder
                       [0]]) # throttle
    x0 = np.concatenate((state0, delta0), axis=0).flatten()
    # define equality constraints
    cons = ({'type': 'eq',
             'fun': lambda x: np.array([
                                x[3]**2 + x[4]**2 + x[5]**2 - Va**2,  # velocity magnitude = Va
                                x[4],  # v = 0 (force no sideslip)
                                x[9],  # p = 0 (no roll rate)
                                x[10], # q = 0 (no pitch rate)
                                x[11], # r = 0 (no yaw rate)
                                ]),
             'jac': lambda x: np.array([
                                [0., 0., 0., 2*x[3], 2*x[4], 2*x[5], 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                                ])
             })
    # solve the minimization problem to find the trim states and inputs

    res = minimize(trim_objective_fun, x0, method='SLSQP', args=(mav, Va, gamma),
                   constraints=cons, 
                   options={'ftol': 1e-10, 'disp': True})
    # extract trim state and input and return
    trim_state = np.array([res.x[0:12]]).T
    trim_input = Delta(elevator=res.x.item(12),
                          aileron=res.x.item(13),
                          rudder=res.x.item(14),
                          throttle=res.x.item(15))
    trim_input.print()
    print('trim_state=', trim_state.T)
    return trim_state, trim_input


def trim_objective_fun(x, mav, Va, gamma):
    state = x[0:12]
    delta = Delta(elevator=x.item(12),
                     aileron=x.item(13),
                     rudder=x.item(14),
                     throttle=x.item(15))
    
    desired_trim_state_dot = np.array([[0., 0., Va*np.sin(gamma), 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]).T

    mav._state = state
    mav._update_velocity_data()
    forces_moments = mav._forces_moments(delta)
    f = mav._f(state, forces_moments)
    tmp = desired_trim_state_dot - f
    J = np.linalg.norm(tmp[2:12])**2.0
    return J