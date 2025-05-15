import numpy as np
import matplotlib.pyplot as plt
import os, sys
# insert parent directory at beginning of python search path
from pathlib import Path
sys.path.insert(0,os.fspath(Path(__file__).parents[1]))
from Models.dynamics_control import MavDynamics
from Message_types.delta import Delta
from mpl_toolkits.mplot3d import Axes3D
from Models.wind import WindSimulation
from Models.trim import compute_trim
from Tools.rotations import quaternion_to_euler

#We currently have some issues wrt scalar math overflow
#If this remains an issue, we can decrease the dt value.
#Fix coming soon.
dt = 0.001
num_steps = 50000
t = np.linspace(0, dt*num_steps, num_steps)

#MAV dynamics object
MAV = MavDynamics(Ts=dt)


#Trim Conditions
Va = 25.0
gamma = 0.2
trim_state, trim_input = compute_trim(MAV, Va, gamma)

