import matplotlib.pyplot as plt
import numpy as np
import parameters as P
from integrators import get_integrator
from pid import PIDControl


class Controller:
    def __init__(self):
        self.emax = P.emax
        self.zeta = P.zeta
        self.kp = P.kp
        self.wn = P.wn
        self.ki = P.ki
        self.kd = P.kd
        self.sigma = P.sigma
        self.Ts = P.Ts
        
        self.pid = PIDControl(self.kp, self.ki, self.kd, self.emax, self.sigma, self.Ts)

    def update(self, r, y):
        u = self.pid.PID(r, y)
        
        return u
    
class System:
    def __init__(self):
        self.K = P.K
        self.tau = P.tau
        self.Ts = P.Ts
        self.x = 0

        def motor_dynamics(t, x, u):
            return (-1/self.tau) * x + (self.K/self.tau) * u

        self.integrator = get_integrator(self.Ts, motor_dynamics, integrator="RK4")      
    
   
    def update(self, u):
        self.x = self.integrator.step(0, self.x, u)
        return self.x

# Init system and feedback controller
system = System()
controller = Controller()

# Simulate step response
t_history = [0]
y_history = [0]
u_history = [0]

r = P.rstep
y = 0
t = 0
for i in range(P.nsteps):
    u = controller.update(r, y) 
    y = system.update(u) 
    t += P.Ts

    t_history.append(t)
    y_history.append(y)
    u_history.append(u)


# Plot response y due to step change in r
plt.figure()
plt.plot(t_history, y_history, label="System Output (y)")
plt.axhline(r, color='r', linestyle="--", label="Reference (r)")
plt.xlabel("Time (s)")
plt.ylabel("Output")
plt.title("Step Response of PID-Controlled System")
plt.legend()
plt.grid()

# Plot control signal
plt.figure()
plt.plot(t_history, u_history, label="Control Input (u)")
plt.xlabel("Time (s)")
plt.ylabel("Control Signal (V)")
plt.title("Control Signal Over Time")
plt.legend()
plt.grid()

plt.show()
