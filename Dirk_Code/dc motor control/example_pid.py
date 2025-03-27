import numpy as np
import matplotlib.pyplot as plt
import Parameters.parameters as P
from integrators import get_integrator
from pid import PIDControl



class Controller:
    def __init__(self):
        self.kp = P.kp 
        self.ki = P.ki
        self.kd = P.kd
        self.emax = P.emax
        self.sigma = P.sigma
        self.Ts = P.Ts
        self.controller = PIDControl(self.kp, self.ki, self.kd, self.emax, self.sigma, self.Ts)
        self._state = 0 #u
        
    def update(self, r, y):
        self._state = self.controller.PID(r, y)
        return self._state
        

#represents the motor, state is omega, input is voltage
class System:
    def __init__(self):
        self.RK4 = get_integrator(P.Ts, System.f, "RK4")
        self._state = np.array([
            [0], #omega
        ])        
    
    def update(self, u):
        self._state = self.RK4.step(0, self._state, u)
        return self._state

    def f(t, y, u):
        #tau * y-dot + y = K * u
        y_dot = (-1/P.tau) * y + (P.K/P.tau) * u
        return y_dot

# Init system and feedback controller
system = System()
controller = Controller()

# Simulate step response
t_history = [0]
y_history = [0]
u_history = [0]

r = 1
y = 0
t = 0
for i in range(P.nsteps):
    u = controller.update(r, y) 
    y = system.update(u)[0, 0]
    t += P.Ts

    t_history.append(t)
    y_history.append(y)
    u_history.append(u)

# Plot response y due to step change in r
plt.figure()
plt.plot(t_history, y_history)
plt.xlabel('Time [s]')
plt.ylabel('Omega [rad/s]')
plt.title('Step response of DC motor')
plt.grid()
plt.show()


# Plot actuation signal
plt.figure()
plt.plot(t_history, u_history)
plt.xlabel('Time [s]')
plt.ylabel('Voltage [V]')
plt.title('Actuation signal')
plt.grid()
plt.show()
