import numpy as np
import matplotlib.pyplot as plt
import integrators as intg

#params
mass = 11. #kg
J_x = 0.8244 #kg m^2
J_y = 1.135
J_z = 1.759
J_xz = 0.1204
S_wing = 0.55
b = 2.8956
c = 0.18994
S_prop = 0.2027
rho = 1.2682
e = 0.9
AR = (b**2) / S_wing
gravity = 9.81

#Drag/longitudinal coefficients
C_L_0 = 0.23
C_D_0 = 0.0424
C_m_0 = 0.0135
C_L_alpha = 5.61
C_D_alpha = 0.132
C_m_alpha = -2.74
C_L_q = 7.95
C_D_q = 0.0
C_m_q = -38.21
C_L_delta_e = 0.13
C_D_delta_e = 0.0135
C_m_delta_e = -0.99
M = 50.0
alpha0 = 0.47
epsilon = 0.16
C_D_p = 0.043

#Drag/lateral coefficients
C_Y_0 = 0.0
C_ell_0 = 0.0
C_n_0 = 0.0
C_Y_beta = -0.98
C_ell_beta = -0.13
C_n_beta = 0.073
C_Y_p = 0.0
C_ell_p = -0.51
C_n_p = 0.069
C_Y_r = 0.0
C_ell_r = 0.25
C_n_r = -0.095
C_Y_delta_a = 0.075
C_ell_delta_a = 0.17
C_n_delta_a = -0.011
C_Y_delta_r = 0.19
C_ell_delta_r = 0.0024
C_n_delta_r = -0.069

#ic
pn_0 = 0 #position
pe_0 = 0
pd_0 = 0
u_0 = 10 #velocity
v_0 = 0
w_0 = 0
phi_0 = 0 #orientation
theta_0 = 0
psi_0 = 0
p_0 = 0 #angular velocity in body frame
q_0 = 0
r_0 = 0

#wind
u_r = u_0 -1
v_r = v_0 -1
w_r = w_0 -1

#angle of attack and sideslip[] angles
alpha = np.arctan(w_r/u_r)
beta = np.arcsin(v_r / np.sqrt(u_r**2 + v_r**2 + w_r**2))

#wind params
rho = 1.268 #kg/m^3
V_a = np.sqrt(u_r**2 + v_r**2 + w_r**2)
C_L = C_L_0 + C_L_alpha * alpha
C_D = C_D_0 + C_D_alpha * alpha
C_X = -C_D*np.cos(alpha) + C_L*np.sin(alpha)
C_X_q = -C_D_q*np.cos(alpha) + C_L_q*np.sin(alpha)
C_X_delta_e = -C_D_delta_e*np.cos(alpha) + C_L_delta_e*np.sin(alpha)
C_Z = -C_D*np.sin(alpha) - C_L*np.cos(alpha)
C_Z_q = -C_D_q*np.sin(alpha) - C_L_q*np.cos(alpha)
C_Z_delta_e = -C_D_delta_e*np.sin(alpha) - C_L_delta_e*np.cos(alpha)

#applied
f_x = -mass*gravity*np.sin(theta_0) + 0.5*rho*V_a**2*S_wing* ( C_X+C_X_q*(c*q_0)/(2*V_a) )
f_y = mass*gravity*np.cos(theta_0)*np.sin(phi_0)+ 0.5*rho*V_a**2*S_wing* ( C_Y_0 + C_Y_beta*beta + C_Y_p*(b*p_0)/(2*V_a) + C_Y_r*(b*r_0)/(2*V_a))
f_z = mass*gravity*np.cos(theta_0)*np.cos(phi_0) + 0.5*rho*V_a**2*S_wing* ( C_Z + C_Z_q*(c*q_0)/(2*V_a))
l = 0
m = 0
n = 0

dt = 0.001
num_steps = 5000

state = np.array([pn_0, pe_0, pd_0, u_0, v_0, w_0, phi_0, theta_0, psi_0, p_0, q_0, r_0])
input = np.array([f_x, f_y, f_z, l, m, n])

def f(t, state, input):
    pn, pe, pd, u, v, w, phi, theta, psi, p, q, r = state
    f_x, f_y, f_z, l, m, n = input
    
    c_theta = np.cos(theta)
    s_theta = np.sin(theta)
    c_phi = np.cos(phi)
    s_phi = np.sin(phi)
    c_psi = np.cos(psi)
    s_psi = np.sin(psi)

    pn_dot = (c_theta*c_psi)*u + (s_phi*s_theta*c_psi - c_phi*s_psi)*v + (c_phi*s_theta*c_psi + s_phi*s_psi)*w
    pe_dot = (c_theta*s_psi)*u + (s_phi*s_theta*s_psi + c_phi*c_psi)*v + (c_phi*s_theta*s_psi - s_phi*c_psi)*w
    pd_dot = -(s_theta)*u + (s_phi*c_theta)*v + (c_phi*c_theta)*w

    u_dot = r*v - q*w + f_x/mass
    v_dot = p*w - r*u + f_y/mass
    w_dot = q*u - p*v + f_z/mass

    angular_velocity_body = np.array([p, q, r])

    angular_matrix = np.array([
        [1, s_phi*np.tan(theta), c_phi*np.tan(theta)],
        [0, c_phi, -s_phi],
        [0, s_phi/c_theta, c_phi/c_theta ]
    ])

    angular_acceleration_euler = angular_matrix @ angular_velocity_body

    phi_dot = angular_acceleration_euler[0]
    theta_dot = angular_acceleration_euler[1]
    psi_dot = angular_acceleration_euler[2]

    gamma = (J_x*J_z) - (J_xz**2)
    gamma_1 = (J_xz*(J_x - J_y + J_z))/gamma
    gamma_2 = (J_z*(J_z - J_y) + J_xz**2)/gamma
    gamma_3 = J_z/gamma
    gamma_4 = J_xz/gamma
    gamma_5 = (J_z - J_x)/J_y
    gamma_6 = J_xz/J_y
    gamma_7 = ((J_x - J_y)*J_x + J_xz**2)/gamma
    gamma_8 = J_x/gamma

    angular_acceleration_body = np.array([
        gamma_1*p*q - gamma_2*q*r + gamma_3*l + gamma_4*n,
        gamma_5*p*r - gamma_6*((p**2) - (r**2)) + m/J_y,
        gamma_7*p*q - gamma_1*q*r + gamma_4*l + gamma_8*n
    ])

    p_dot = angular_acceleration_body[0]
    q_dot = angular_acceleration_body[1]
    r_dot = angular_acceleration_body[2]

    return np.array([pn_dot, pe_dot, pd_dot, u_dot, v_dot, w_dot, phi_dot, theta_dot, psi_dot, p_dot, q_dot, r_dot])

x_RK4 = state

RK4 = intg.RK4(dt, f)

t_history = [0]
x_RK4_history = [x_RK4]

t = 0.0
for i in range(num_steps):
    x_RK4 = RK4.step(t, x_RK4, input)
    t = (i+1) * dt
    t_history.append(t)
    x_RK4_history.append(x_RK4)

x_RK4_history = np.array(x_RK4_history)

fig, axs = plt.subplots(4, 1, figsize=(15, 8))

axs[0].plot(t_history, x_RK4_history[:, 0], label='pn')
axs[0].plot(t_history, x_RK4_history[:, 1], label='pe')
axs[0].plot(t_history, x_RK4_history[:, 2], label='pd')
axs[0].set_title('Position')
axs[0].legend()

axs[1].plot(t_history, x_RK4_history[:, 3], label='u')
axs[1].plot(t_history, x_RK4_history[:, 4], label='v')
axs[1].plot(t_history, x_RK4_history[:, 5], label='w')
axs[1].set_title('Velocity')
axs[1].legend()

axs[2].plot(t_history, x_RK4_history[:, 6], label='phi')
axs[2].plot(t_history, x_RK4_history[:, 7], label='theta')
axs[2].plot(t_history, x_RK4_history[:, 8], label='psi')
axs[2].set_title('Angular Position')
axs[2].legend()

axs[3].plot(t_history, x_RK4_history[:, 9], label='p')
axs[3].plot(t_history, x_RK4_history[:, 10], label='q')
axs[3].plot(t_history, x_RK4_history[:, 11], label='r')
axs[3].set_title('Angular Velocity')
axs[3].legend()

plt.tight_layout()
plt.show()