"""
mavsim_python
    - Chapter 8 assignment for Beard & McLain, PUP, 2012
    - Last Update:
        2/21/2019 - RWB
        2/24/2020 - RWB
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import copy
import Parameters.simulation_parameters as SIM

from viewers.mav_viewer import MavViewer
from viewers.data_viewer import DataViewer
from viewers.sensor_viewer import SensorViewer
from Models.wind import WindSimulation
from Controllers.autopilot_lqr import Autopilot
from Models.sensors import MavDynamics
from Estimators.observer import Observer
from Tools.signals import Signals
from PyQt5 import QtWidgets
import Models.model_coef as M

app = QtWidgets.QApplication([])


# initialize the visualization
VIDEO = False  # True==write video, False==don't write video
mav_view = MavViewer(app)  # initialize the mav viewer
data_view = DataViewer(app)  # initialize view of data plots
sensor_view = SensorViewer(app)
if VIDEO is True:
    from viewers.video_writer import VideoWriter
    video = VideoWriter(video_name="chap8_video.avi",
                        bounding_box=(0, 0, 1000, 1000),
                        output_rate=SIM.ts_video)

# initialize elements of the architecture
wind = WindSimulation(SIM.ts_simulation)
mav = MavDynamics(SIM.ts_simulation)
autopilot = Autopilot(SIM.ts_simulation)
mav._state = M.x_trim
initial_measurements = copy.deepcopy(mav.sensors())
observer = Observer(SIM.ts_simulation, initial_measurements)

# autopilot commands
from Message_types.autopilot import MsgAutopilot
commands = MsgAutopilot()
Va_command = Signals(dc_offset=25.0,
                     amplitude=3.0,
                     start_time=2.0,
                     frequency = 0.01)
h_command = Signals(dc_offset=100.0,
                    amplitude=10.0,
                    start_time=0.0,
                    frequency=0.02)
chi_command = Signals(dc_offset=np.radians(180),
                      amplitude=np.radians(45),
                      start_time=5.0,
                      frequency=0.015)

# initialize the simulation time
sim_time = SIM.start_time

# main simulation loop
print("Press Command-Q to exit...")
while sim_time < SIM.end_time:

    app.processEvents()

    # -------autopilot commands-------------
    commands.airspeed_command = Va_command.square(sim_time)
    commands.course_command = chi_command.square(sim_time)
    commands.altitude_command = h_command.square(sim_time)

    # -------autopilot-------------
    measurements = mav.sensors()  # get sensor measurements
    estimated_state = observer.update(measurements)  # estimate states from measurements
    # delta, commanded_state = autopilot.update(commands, estimated_state)
    delta, commanded_state = autopilot.update(commands, mav.true_state)

    # -------physical system-------------
    current_wind = wind.update()  # get the new wind vector
    mav.update(delta, current_wind)  # propagate the MAV dynamics

    # -------update viewer-------------
    mav_view.update(mav.true_state)  # plot body of MAV
    data_view.update(mav.true_state,  # true states
                     estimated_state,  # estimated states
                     commanded_state,  # commanded states
                     delta)  # input to aircraft
    sensor_view.update(measurements)  # input to aircraft
    
    if VIDEO is True:
        video.update(sim_time)

    # -------increment time-------------
    sim_time += SIM.ts_simulation

if VIDEO is True:
    video.close()



