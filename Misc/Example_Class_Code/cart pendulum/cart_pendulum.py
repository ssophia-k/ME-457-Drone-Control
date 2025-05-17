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

# Value problem parameters
# First guess is based on intuition - want to drive x and theta to 0 the most
q1 = 100    #x
q2 = 0.1   #x-dot
q3 = 10   #theta
q4 = 0.1   #theta-dot
r = 1      #force input

# State space
# x = [x, x_dot, theta, theta_dot]
# u = [F]
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
    [0, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 0]
    ])
D = np.array([
    [0],
    [0],
    [0],
    [0]
    ])

# Set up state space system, get poles
sys = ct.ss(A, B, C, D)
print(ct.poles(sys)) #unstable

# See if system is controllable
ctrb = ct.ctrb(A,B) #[B, AB, A^2B, A^3B]
rank = np.linalg.matrix_rank(ctrb)
print(rank) #rank = 4, full rank, so controllable

# Set up LQR controller via ACE solver
Q = np.diag([q1, q2, q3, q4])
R = np.array([[r]])
K, S, E = ct.lqr(A, B, Q, R)  # correctly unpack K

# Now implement the closed loop system with the LQR controller
A_cl = A - B @ K
sys_cl = ct.ss(A_cl, B, np.eye(4), D)
print(ct.poles(sys_cl)) #stable

# Simulate the system with step response where v = 0.2 * unit step
t = np.linspace(0, 10, 1000)
t, y = ct.step_response(sys_cl, T=t)
y *= 0.2  # scale the step response to match the input
y = np.squeeze(y)

# Plot the results
plt.figure()
plt.plot(t, y[0], label='x')
plt.plot(t, y[1], label='x_dot')
plt.plot(t, y[2], label='theta')
plt.plot(t, y[3], label='theta_dot')
plt.xlabel('Time [s]')
plt.ylabel('States')
plt.title('State Response')
plt.legend()
plt.grid()
plt.show()

# Compute control input: u(t) = -K * x(t)
u = -K @ y
u = np.squeeze(u)  # Make u a 1D array

# Plot the control input
plt.figure()
plt.plot(t, u, label='Control Input')
plt.xlabel('Time [s]')
plt.ylabel('Control Input')
plt.title('Control Input vs Time')
plt.legend()
plt.grid()
plt.show()



