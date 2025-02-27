import matplotlib.pyplot as plt
import numpy as np

def plot_trajectory(north, east, down):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(north, east, down, label='Aircraft Position')
    ax.set_xlabel('North (m)')
    ax.set_ylabel('East (m)')
    ax.set_zlabel('Down (m)')
    ax.set_title('3D Flight Path')
    ax.legend()
    plt.show()
