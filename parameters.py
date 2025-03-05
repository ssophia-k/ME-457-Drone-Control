import numpy as np

#Initial Conditions
pn_0 = 0 #position, north
pe_0 = 0 #east
pd_0 = -1 #down
u_0 = 10 #velocity, north
v_0 = 0 #east
w_0 = 0 #down
phi_0 = 0 #orientation, roll
theta_0 = 0 #pitch
psi_0 = 0 #yaw
p_0 = 0 #angular velocity in body frame, roll
q_0 = 0 #pitch
r_0 = 0 #yaw
Va_0 = np.sqrt(u_0**2+v_0**2+w_0**2)

#Physical Parameters
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
gravity = 9.81
e = 0.9
AR = (b**2) / S_wing

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

#   Propeller thrust / torque parameters (see addendum by McLain)
######################################################################################
# Prop parameters
D_prop = 20*(0.0254)     # prop diameter in m

# Motor parameters
KV_rpm_per_volt = 145.                            # Motor speed constant from datasheet in RPM/V
KV = (1. / KV_rpm_per_volt) * 60. / (2. * np.pi)  # Back-emf constant, KV in V-s/rad
KQ = KV                                           # Motor torque constant, KQ in N-m/A
R_motor = 0.042              # ohms
i0 = 1.5                     # no-load (zero-torque) current (A)


# Inputs
ncells = 12.
V_max = 3.7 * ncells  # max voltage for specified number of battery cells

# Coeffiecients from prop_data fit
C_Q2 = -0.01664
C_Q1 = 0.004970
C_Q0 = 0.005230
C_T2 = -0.1079
C_T1 = -0.06044
C_T0 = 0.09357

