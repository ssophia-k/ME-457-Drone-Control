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
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1]
    ])
D = np.array([
    [0],
    [0],
    [0],
    [0]
    ])

sys = ct.ss(A, B, C, D)
poles = ct.poles(sys)
# print(poles)

co = ct.ctrb(A,B)
nc = np.linalg.matrix_rank(co)
# print(nc)

q1 = 1
q2 = 0.1
q3 = 1
q4 = 0.1
r = 1
Q = np.diag([q1, q2, q3, q4])
R = np.array([[r]])

K,_,_ = ct.lqr(A, B, Q, R)

sys_cl = ct.ss(A-B@K, B, C, D)
poles_cl = ct.poles(sys_cl)

T, yout = ct.step_response(sys_cl)
yout=np.squeeze(yout)
u = -K @ yout

plt.plot(T, yout[0], label='x')
plt.plot(T, yout[1], label='theta')
plt.plot(T, yout[2], label='x_dot')
plt.plot(T, yout[3], label='theta_dot')
plt.plot(T, u[0,:], label='u')
plt.legend()
plt.grid(True)
plt.show()