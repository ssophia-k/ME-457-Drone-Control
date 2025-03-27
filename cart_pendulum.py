import numpy as np
import control as ct
import matplotlib.pyplot as plt

# Parameters
M = 0.5
m = 0.2
b = 0.1
I = 0.006
g = 9.8
l = 0.3

# State space
den = I*(M+m)+M*m*l**2
A = np.array([
    [0,      1,              0,           0],
    [0, -(I+m*l**2)*b/den,  (m**2*g*l**2)/den,  0],
    [0,      0,              0,           1],
    [0, -(m*l*b)/den,       m*g*l*(M+m)/den,  0]
    ])
B = np.array([
    [0],
    [(I+m*l**2)/den],
    [0],
    [m*l/den]
    ])
C = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0]
    ])
D = np.array([
    [0],
    [0]
    ])

sys = ct.ss(A, B, C, D)
print(ct.pole(sys))

ctrb = ct.ctrb(A, B)
rank = np.linalg.matrix_rank(ctrb)
print(rank)

# LQR design
Q = np.diag([10, 0.1, 100, 1])
R = np.array([[1]])
K, S, E = ct.lqr(A, B, Q, R)

print("LQR gain matrix K:", K)
print("Closed-loop poles:", E)

# Closed-loop system
A_cl = A - B @ K
sys_cl = ct.ss(A_cl, B, C, D)

ct.step_response(sys_cl)

# Reference input (step of 0.2)
r = 0.2  # desired cart position

# Compute feedforward gain Nbar
# This ensures steady-state tracking of reference
# Nbar = -1 / (C (A - BK)^-1 B)
# Only care about tracking the cart position (first output)
C_ref = np.array([[1, 0, 0, 0]])
Nbar = -1.0 / (C_ref @ np.linalg.inv(A - B @ K) @ B)

# Closed-loop system with reference input
A_cl = A - B @ K
B_cl = B * Nbar  # scaled input to get reference tracking
sys_cl_ref = ct.ss(A_cl, B_cl, C, D)

# Simulate step response to reference input of 0.2
t = np.linspace(0, 10, 500)
t_out, y_out = ct.forced_response(sys_cl_ref, T=t, U=np.ones_like(t) * r)

# Plot
plt.figure()
plt.plot(t_out, y_out[0], label='Cart Position (m)')
plt.plot(t_out, y_out[1], label='Pendulum Angle (rad)')
plt.title("LQR Response to Step Reference (0.2 m)")
plt.xlabel("Time (s)")
plt.ylabel("Output")
plt.grid(True)
plt.legend()
plt.show()

# Reconstruct the full state trajectory (not just outputs)
# Simulate with state output
_, y_states = ct.forced_response(ct.ss(A_cl, B_cl, np.eye(4), np.zeros((4,1))), T=t, U=np.ones_like(t) * r)

# Compute control input: u(t) = -K x(t)
u = -K @ y_states

# Plot all state variables
plt.figure()
plt.plot(t, y_states[0], label='x (Cart Position) [m]')
plt.plot(t, y_states[1], label="x' (Cart Velocity) [m/s]")
plt.plot(t, y_states[2], label='θ (Pendulum Angle) [rad]')
plt.plot(t, y_states[3], label="θ' (Angular Velocity) [rad/s]")
plt.title("State Trajectories under LQR Control")
plt.xlabel("Time (s)")
plt.ylabel("State Values")
plt.grid(True)
plt.legend()
plt.show()

# Plot control input
plt.figure()
plt.plot(t, u.flatten(), label='Control Input u(t)')
plt.title("Control Input Over Time (u = -Kx)")
plt.xlabel("Time (s)")
plt.ylabel("Force (N)")
plt.grid(True)
plt.legend()
plt.show()
