import os, sys
# insert parent directory at beginning of python search path
from pathlib import Path
sys.path.insert(0,os.fspath(Path(__file__).parents[1]))
import numpy as np
from Message_types.delta import Delta
from Models.dynamics_control import MavDynamics


# Build mavsim object and set time step
Ts   = 0.02
mav  = MavDynamics(Ts)


# Trim conditions from Luchtenburg's file
x_trim = np.array([
    -0.000000, -0.000000, -100.000000,
     24.971443,   0.000000,   1.194576,
      0.993827,   0.000000,   0.110938,
       0.000000,  0.000000,   0.000000,
       0.000000
]).reshape((13,1))

u_trim = Delta(
    elevator = float(-0.118662),
    aileron  = float( 0.009775),
    rudder   = float(-0.001611),
    throttle = float( 0.857721)
)


# Initialize mav’s internal state with the trim conditions
mav._state = x_trim.copy()
mav._update_velocity_data()


# Helper function to compute the full state derivative, x_dot = f(x,u)
def full_state_derivative(x, u):
    # ensure x is (13×1)
    x_col = x.reshape((13,1))
    mav._state = x_col
    mav._update_velocity_data()
    forces_moments = mav._forces_moments(u)
    # returns a flat (13,) array
    return mav._f(x_col, forces_moments).flatten()


# Finite‑difference to build A (13×13) and B (13×4)
x0  = x_trim.flatten()      # (13,)
eps = 1e-6                  # small perturbation for finite difference
n   = x0.size               # 13
m   = 4                     # number of controls

A = np.zeros((n,n))
B = np.zeros((n,m))


# Fill A
for i in range(n):
    dx = np.zeros(n); dx[i] = eps
    f_plus  = full_state_derivative(x0 + dx, u_trim)
    f_minus = full_state_derivative(x0 - dx, u_trim)
    A[:,i] = (f_plus - f_minus) / (2*eps)


# Fill B
controls = ['elevator','aileron','rudder','throttle']
# Make a mutable copy of the baseline control dict
base = {
    'elevator': u_trim.elevator,
    'aileron':  u_trim.aileron,
    'rudder':   u_trim.rudder,
    'throttle': u_trim.throttle
}
for j,name in enumerate(controls):
    # bump +eps / -eps
    base[name] += eps
    u_plus  = Delta(**base)
    base[name] -= 2*eps
    u_minus = Delta(**base)
    # restore base[name] for next iteration
    base[name] += eps

    f_plus  = full_state_derivative(x0, u_plus)
    f_minus = full_state_derivative(x0, u_minus)
    B[:,j] = (f_plus - f_minus) / (2*eps)


# A and B are the full 13×13 and 13×4 Jacobians
print("A shape:", A.shape)
print("B shape:", B.shape)

print("A:\n", A)
print("B:\n", B)
