
import os, sys
# insert parent directory at beginning of python search path
from pathlib import Path
sys.path.insert(0,os.fspath(Path(__file__).parents[1]))

import numpy as np
import matplotlib.pyplot as plt
from Models.dynamics_control import MavDynamics
from Message_types.delta import Delta
from mpl_toolkits.mplot3d import Axes3D
from Tools.rotations import quaternion_to_euler
from Models.wind import WindSimulation

#We currently have some issues wrt scalar math overflow
#If this remains an issue, we can decrease the dt value.
#Fix coming soon.
dt = 0.001
num_steps = 5000
t = np.linspace(0, dt*num_steps, num_steps)

#MAV dynamics object
MAV = MavDynamics(Ts=dt)

#Wind Simulation Object
WIND_SIM = WindSimulation(Ts = dt, gust_flag=True, steady_state = np.array([[0., 0., 5.]]).T)

#Control input
delta = Delta(elevator=0.0, aileron=0.0, rudder=0.0, throttle=0.05) 

#Wind vector (ss and gust), zero for now
wind = np.zeros((6, 1))

#Array to store state history
# 13 state variables (north, east, down, u, v, w, e0, e1, e2, e3, p, q, r)
state_history = np.zeros((num_steps, 13))  
wind_history = np.zeros((num_steps, 6))

#Integrate it up!
for i in range(num_steps):
    wind = WIND_SIM.update()
    MAV.update(delta, wind)

    # Store all state variables in one row
    state_history[i, :] = MAV._state[:13, 0] 
    wind_history[i, :] = wind[:6, 0]   

north = state_history[:, 0]
east = state_history[:, 1]
down = state_history[:, 2]
u = state_history[:, 3]
v = state_history[:, 4]
w = state_history[:, 5]
e0 = state_history[:, 6]
e1 = state_history[:, 7]
e2 = state_history[:, 8]
e3= state_history[:, 9]
p = state_history[:, 10]
q = state_history[:, 11]
r = state_history[:, 12]

euler_angles = np.array([quaternion_to_euler(q) for q in state_history[:, 6:10]])
phi, theta, psi = euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2]

fig3d = plt.figure()
ax = fig3d.add_subplot(111, projection='3d')
ax.plot(north, east, down, label='Aircraft Position')
ax.set_xlabel('North (m)')
ax.set_ylabel('East (m)')
ax.set_zlabel('Down (m)')
ax.set_title('3D Flight Path')
ax.legend()
plt.show()

print(phi)

fig, axs = plt.subplots(4, 1, figsize=(15, 8))

axs[0].plot(t, north, label='pn')
axs[0].plot(t, east, label='pe')
axs[0].plot(t, down, label='pd')
axs[0].set_title('Position')
axs[0].legend()

axs[1].plot(t, u, label='u')
axs[1].plot(t, v, label='v')
axs[1].plot(t, w, label='w')
axs[1].set_title('Velocity')
axs[1].legend()

axs[2].plot(t, phi, label='phi')
axs[2].plot(t, theta, label='theta')
axs[2].plot(t, psi, label='psi')
axs[2].set_title('Angular Position')
axs[2].legend()

axs[3].plot(t, p, label='p')
axs[3].plot(t, q, label='q')
axs[3].plot(t, r, label='r')
axs[3].set_title('Angular Velocity')
axs[3].legend()

plt.tight_layout()
plt.show()


#Plot wind
fig, axs = plt.subplots(2, 1, figsize=(15, 8))

#Components
axs[0].plot(t, wind_history[:, 0], label='u_s')
axs[0].plot(t, wind_history[:, 1], label='v_s')
axs[0].plot(t, wind_history[:, 2], label='w_s')
axs[0].plot(t, wind_history[:, 3], label='u_g')
axs[0].plot(t, wind_history[:, 4], label='v_g')
axs[0].plot(t, wind_history[:, 5], label='w_g')
axs[0].set_title('Wind')
axs[0].legend()

#Magnitude
axs[1].plot(t, np.sqrt((wind_history[:, 0]+wind_history[:, 3])**2 + (wind_history[:, 1]+wind_history[:, 4])**2 + (wind_history[:, 2]+wind_history[:, 5])**2), label='Wind Magnitude')
axs[1].set_title('Wind Magnitude')
axs[1].legend()

plt.tight_layout()
plt.show()
