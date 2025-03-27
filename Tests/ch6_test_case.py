"""
mavsim_python
    - Chapter 6 assignment for Beard & McLain, PUP, 2012
    - Last Update:
        2/5/2019 - RWB
        2/24/2020 - RWB
        1/5/2023 - David L. Christiansen
        7/13/2023 - RWB
"""

import os, sys
from pathlib import Path
sys.path.insert(0,os.fspath(Path(__file__).parents[1]))

import numpy as np
import Parameters.simulation_parameters as SIM
from Tools.signals import Signals
from Models.dynamics_control import MavDynamics
from Models.wind import WindSimulation
from Controllers.autopilot import Autopilot
import matplotlib.pyplot as plt

# initialize elements of the architecture
wind = WindSimulation(SIM.ts_simulation)
mav = MavDynamics(SIM.ts_simulation)
autopilot = Autopilot(SIM.ts_simulation)

# autopilot commands
from Message_types.autopilot import Autopilot
commands = Autopilot()
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

# initialize the simulation time
sim_time = SIM.start_time
end_time = 300

# main simulation loop
while sim_time < end_time:

    # autopilot commands
    commands.airspeed_command = Va_command.square(sim_time)
    commands.course_command = course_command.square(sim_time)
    commands.altitude_command = altitude_command.square(sim_time)

    # autopilot
    estimated_state = mav.true_state
    delta, commanded_state = autopilot.update(commands, estimated_state)

    # physical system
    current_wind = wind.update()
    mav.update(delta, current_wind)

    # store data for plotting
    time_history.append(sim_time)
    Va_history.append(mav.true_state.Va)
    altitude_history.append(-mav.true_state.altitude)
    course_history.append(np.degrees(mav.true_state.chi))
    delta_e_history.append(delta.elevator)
    delta_a_history.append(delta.aileron)
    delta_r_history.append(delta.rudder)
    delta_t_history.append(delta.throttle)

    # increment time
    sim_time += SIM.ts_simulation

# plot the data
plt.figure()

# airspeed
plt.subplot(4, 1, 1)
plt.plot(time_history, Va_history, label='Airspeed (Va)')
plt.ylabel('Va (m/s)')
plt.grid()
plt.legend()

# altitude
plt.subplot(4, 1, 2)
plt.plot(time_history, altitude_history, label='Altitude')
plt.ylabel('Altitude (m)')
plt.grid()
plt.legend()

# course
plt.subplot(4, 1, 3)
plt.plot(time_history, course_history, label='Course angle (deg)')
plt.ylabel('Course (deg)')
plt.grid()
plt.legend()

# control inputs
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
