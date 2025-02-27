import numpy as np
import matplotlib.pyplot as plt
from dynamics_control import MavDynamics
from delta import Delta
from mpl_toolkits.mplot3d import Axes3D

#We currently have some issues wrt scalar math overflow
#If this remains an issue, we can decrease the dt value.
#Fix coming soon.
dt = 0.001
num_steps = 5000
t = np.linspace(0, dt*num_steps, num_steps)

#MAV dynamics object
MAV = MavDynamics(Ts=dt)

#Control input
delta = Delta(elevator=0.0, aileron=0.0, rudder=0.0, throttle=0.0) 

#Wind vector (ss and gust), zero for now
wind = np.zeros((6, 1))

#Array to store state history
# 9 state variables (north, east, down, u, v, w, p, q, r, phi, theta, psi)
state_history = np.zeros((num_steps, 12))  

#Integrate it up!
for i in range(num_steps):
    MAV.update(delta, wind)

    # Store all state variables in one row
    state_history[i, :] = MAV._state[:12, 0]  

north = state_history[:, 0]
east = state_history[:, 1]
down = state_history[:, 2]
u = state_history[:, 3]
v = state_history[:, 4]
w = state_history[:, 5]
p = state_history[:, 6]
q = state_history[:, 7]
r = state_history[:, 8]
phi = state_history[:, 9]
theta = state_history[:, 10]
psi = state_history[:, 11]

fig3d = plt.figure()
ax = fig3d.add_subplot(111, projection='3d')
ax.plot(north, east, down, label='Aircraft Position')
ax.set_xlabel('North (m)')
ax.set_ylabel('East (m)')
ax.set_zlabel('Down (m)')
ax.set_title('3D Flight Path')
ax.legend()
plt.show()


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