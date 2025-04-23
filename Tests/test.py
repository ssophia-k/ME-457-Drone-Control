import os, sys
from pathlib import Path
sys.path.insert(0, os.fspath(Path(__file__).parents[1]))

import numpy as np
import Parameters.simulation_parameters as SIM
from Tools.signals import Signals
from Models.dynamics_control import MavDynamics
from Models.wind import WindSimulation
from Controllers.autopilot import Autopilot
from Message_types.autopilot import MsgAutopilot
import matplotlib.pyplot as plt
import Models.model_coef as M
from Tools.rotations import quaternion_to_euler
from scipy.interpolate import interp1d

# initialize elements of the architecture
wind = WindSimulation(SIM.ts_simulation)
mav = MavDynamics(SIM.ts_simulation)

# Set initial state to trim state
mav._state = M.x_trim
autopilot = Autopilot(SIM.ts_simulation)

# autopilot commands
commands = MsgAutopilot()
Va_command = Signals(dc_offset=25.0,
                     amplitude=3.0,
                     start_time=2.0,
                     frequency=0.01)
altitude_command = Signals(dc_offset=100.0,
                           amplitude=20.0,
                           start_time=0.0,
                           frequency=0.02)
course_command = Signals(dc_offset=np.radians(180),
                         amplitude=np.radians(45),
                         start_time=5.0,
                         frequency=0.015)

# initialize storage for plotting
time_history = []
Va_history = []
altitude_history = []
course_history = []
delta_e_history = []
delta_a_history = []
delta_r_history = []
delta_t_history = []

# initialize storage for command signals
Va_command_history = []
altitude_command_history = []
course_command_history = []

# Initialize arrays to store state and wind history
num_steps = 10000
state_history = np.zeros((num_steps, 13))
wind_history = np.zeros((num_steps, 6))

# Simulation loop
sim_time = SIM.start_time
end_time = 200
while sim_time < end_time:
    # autopilot commands
    current_Va_command = Va_command.square(sim_time)
    current_course_command = course_command.square(sim_time)
    current_altitude_command = altitude_command.square(sim_time)
    
    commands.airspeed_command = current_Va_command
    commands.course_command = current_course_command
    commands.altitude_command = current_altitude_command

    # autopilot update
    estimated_state = mav.true_state
    delta, commanded_state = autopilot.update(commands, estimated_state)

    # wind update
    current_wind = wind.update()

    # MAV dynamics update
    mav.update(delta, current_wind)

    # Store state and wind history
    time_history.append(sim_time)
    Va_history.append(mav.true_state.Va)
    altitude_history.append(-mav.true_state.altitude)
    course_history.append(mav.true_state.chi)
    delta_e_history.append(delta.elevator)
    delta_a_history.append(delta.aileron)
    delta_r_history.append(delta.rudder)
    delta_t_history.append(delta.throttle)

    # Store command signals
    Va_command_history.append(current_Va_command)
    altitude_command_history.append(-current_altitude_command)  # Negative to match plotting convention
    course_command_history.append(current_course_command)

    # Store MAV state and wind history
    state_history[int(sim_time/SIM.ts_simulation), :] = mav._state[:13, 0]
    wind_history[int(sim_time/SIM.ts_simulation), :] = current_wind[:6, 0]

    # Increment time
    sim_time += SIM.ts_simulation

# Extract state data
north = state_history[:, 0]
east = state_history[:, 1]
down = state_history[:, 2]

# Function to interpolate zero values
def interpolate_zeros(data):
    # Identify indices where the data is zero
    zero_indices = np.where(data == 0)[0]
    
    # If all values are zero, we need to interpolate
    if len(zero_indices) > 0 and len(zero_indices) == len(data):
        return data  # Return as is if the whole array is zero
    
    # Interpolation: Ignore zero values for interpolation
    non_zero_indices = np.where(data != 0)[0]
    
    # Create an interpolation function for non-zero data
    interpolator = interp1d(non_zero_indices, data[non_zero_indices], kind='linear', fill_value='extrapolate')
    
    # Apply the interpolator to all data points
    interpolated_data = interpolator(np.arange(len(data)))
    
    return interpolated_data

# Interpolate positions if all zero
north = interpolate_zeros(north)
east = interpolate_zeros(east)
down = interpolate_zeros(down)
u = state_history[:, 3]
v = state_history[:, 4]
w = state_history[:, 5]
e0 = state_history[:, 6]
e1 = state_history[:, 7]
e2 = state_history[:, 8]
e3 = state_history[:, 9]
p = state_history[:, 10]
q = state_history[:, 11]
r = state_history[:, 12]

# Convert quaternion to euler angles
euler_angles = np.array([quaternion_to_euler(q) for q in state_history[:, 6:10]])
phi, theta, psi = euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2]

time_history = time_history[:-1]
Va_history = Va_history[:-1]
altitude_history = altitude_history[:-1]
course_history = course_history[:-1]
delta_e_history = delta_e_history[:-1]
delta_a_history = delta_a_history[:-1]
delta_r_history = delta_r_history[:-1]
delta_t_history = delta_t_history[:-1]
Va_command_history = Va_command_history[:-1]
altitude_command_history = altitude_command_history[:-1]
course_command_history = course_command_history[:-1]

# Unwrap angles for plotting
course_history_unwrapped = np.unwrap(course_history)
course_command_history_unwrapped = np.unwrap(course_command_history)

# Plot the data
plt.figure(figsize=(12, 10))

# Airspeed plot
plt.subplot(4, 1, 1)
plt.plot(time_history, Va_history, label='Airspeed (Va)')
plt.plot(time_history, Va_command_history, 'r--', label='Command')
plt.ylabel('Va (m/s)')
plt.grid()
plt.legend()

# Altitude plot
plt.subplot(4, 1, 2)
plt.plot(time_history, altitude_history, label='Altitude')
plt.plot(time_history, altitude_command_history, 'r--', label='Command')
plt.ylabel('Altitude (m)')
plt.grid()
plt.legend()

# Course plot
plt.subplot(4, 1, 3)
plt.plot(time_history, course_history_unwrapped, label='Course angle (rad)')
plt.plot(time_history, course_command_history_unwrapped, 'r--', label='Command')
plt.ylabel('Course (rad)')
plt.grid()
plt.legend()

# Control Inputs plot
plt.subplot(4, 1, 4)
plt.plot(time_history, delta_e_history, label='Elevator')
plt.plot(time_history, delta_a_history, label='Aileron')
plt.plot(time_history, delta_r_history, label='Rudder')
plt.plot(time_history, delta_t_history, label='Throttle')
plt.xlabel('Time (s)')
plt.ylabel('Control Inputs')
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()

fig, axs = plt.subplots(4, 1, figsize=(15, 8))

axs[0].plot(time_history, north, label='pn')
axs[0].plot(time_history, east, label='pe')
axs[0].plot(time_history, down, label='pd')
axs[0].set_title('Position')
axs[0].legend()

axs[1].plot(time_history, u, label='u')
axs[1].plot(time_history, v, label='v')
axs[1].plot(time_history, w, label='w')
axs[1].set_title('Velocity')
axs[1].legend()

axs[2].plot(time_history, phi, label='phi')
axs[2].plot(time_history, theta, label='theta')
axs[2].plot(time_history, psi, label='psi')
axs[2].set_title('Angular Position')
axs[2].legend()

axs[3].plot(time_history, p, label='p')
axs[3].plot(time_history, q, label='q')
axs[3].plot(time_history, r, label='r')
axs[3].set_title('Angular Velocity')
axs[3].legend()

plt.tight_layout()
plt.show()

# 3D Plot of Aircraft Position
fig3d = plt.figure()
ax = fig3d.add_subplot(111, projection='3d')
ax.plot(north, east, down, label='Aircraft Position')
x_min, x_max = np.min(north), np.max(north)
y_min, y_max = np.min(east), np.max(east)
xy_max_range = max(x_max - x_min, y_max - y_min) / 2.0

x_mid = (x_max + x_min) / 2.0
y_mid = (y_max + y_min) / 2.0

ax.set_xlim(x_mid - xy_max_range, x_mid + xy_max_range)
ax.set_ylim(y_mid - xy_max_range, y_mid + xy_max_range)
ax.set_xlabel('North (m)')
ax.set_ylabel('East (m)')
ax.set_zlabel('Down (m)')
ax.set_title('3D Flight Path')
ax.legend()
plt.show()

# Plot wind components and magnitude
fig, axs = plt.subplots(2, 1, figsize=(15, 8))

# Wind Components plot
axs[0].plot(time_history, wind_history[:, 0], label='u_s')
axs[0].plot(time_history, wind_history[:, 1], label='v_s')
axs[0].plot(time_history, wind_history[:, 2], label='w_s')
axs[0].plot(time_history, wind_history[:, 3], label='u_g')
axs[0].plot(time_history, wind_history[:, 4], label='v_g')
axs[0].plot(time_history, wind_history[:, 5], label='w_g')
axs[0].set_title('Wind Components')
axs[0].legend()

# Wind Magnitude plot
axs[1].plot(time_history, np.sqrt((wind_history[:, 0]+wind_history[:, 3])**2 + 
                                  (wind_history[:, 1]+wind_history[:, 4])**2 + 
                                  (wind_history[:, 2]+wind_history[:, 5])**2), label='Wind Magnitude')
axs[1].set_title('Wind Magnitude')
axs[1].legend()

plt.tight_layout()
plt.show()