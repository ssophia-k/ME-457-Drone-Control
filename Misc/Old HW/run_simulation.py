import numpy as np
import matplotlib.pyplot as plt
import integrators as intg
from scipy.integrate import solve_ivp


# Nonlinear state space form:
#  xdot = f(t, x, u), 
#   t: time
#   x: state vector
#   u: input

b = 0.25
m = 1
k = 1

#Initial conditions
x_0 = 1
x_dot_0 = 0


#Numerical solutions
def f(t, x, u):
    x_double_dot = -b/m * x[0] + -k/m * x[1] + u/m
    x_dot = x[0]
    return np.array([x_double_dot,x_dot])
        
t = 0
x_euler = np.array([x_dot_0, x_0])
x_heun = np.array([x_dot_0, x_0])
x_RK4 = np.array([x_dot_0, x_0])
u = 0
dt = 0.1; n = 500

euler = intg.Euler(dt, f)
heun = intg.Heun(dt, f)
RK4 = intg.RK4(dt, f)

t_history = [0]
x_euler_history = [x_euler]
x_heun_history = [x_heun]
x_RK4_history = [x_RK4]
for i in range(n):
    x_euler = euler.step(t, x_euler, u)
    x_heun = heun.step(t, x_heun, u)
    x_RK4 = RK4.step(t,x_RK4,u)
    t = (i+1) * dt
    t_history.append(t)
    x_euler_history.append(x_euler)
    x_heun_history.append(x_heun)
    x_RK4_history.append(x_RK4)


#analytical solution
# e^(-zeta * omega_n) * ( Asin(omega*t) + Bcos(omega*t) )
t_anal = np.linspace(0, t_history[-1], len(t_history))
omega_d = np.sqrt(4*m*k - b**2) / (2*m)
z = b / (2*m)
B = x_0 
A = (x_dot_0 + z * B) / omega_d
x_analytical = np.exp(-z*t_anal) * (B * np.cos(omega_d * t_anal) +  A * np.sin(omega_d * t_anal))
v_analytical = (-z*np.exp(-z*t_anal) * (A*np.sin(omega_d*t_anal) + B*np.cos(omega_d*t_anal))) + (np.exp(-z*t_anal) * (A*omega_d*np.cos(omega_d*t_anal) - B*omega_d*np.sin(omega_d*t_anal)))


#plots
intg.__doc__
fig, axs = plt.subplots(2)
axs[0].plot(t_anal, x_analytical)
axs[0].plot(t_history, np.array(x_euler_history)[:,1], alpha = 0.6)
axs[0].plot(t_history, np.array(x_heun_history)[:,1], alpha = 0.6)
axs[0].plot(t_history, np.array(x_RK4_history)[:,1], alpha = 0.6)
axs[0].set_title("Numerical and Analytical Integration Methods, X(t)")
axs[0].legend(['Analytical','Euler', 'Heun', 'RK4'])

axs[1].plot(t_anal, np.array(x_euler_history)[:,1] - x_analytical, alpha = 0.6)
axs[1].plot(t_anal, np.array(x_heun_history)[:,1] - x_analytical, alpha = 0.6)
axs[1].plot(t_anal, np.array(x_RK4_history)[:,1] - x_analytical, alpha = 0.6)
axs[1].set_title("Numerical Integration Methods Error")
axs[1].legend(['Euler error', 'Heun error', 'RK4 error'])

fig2, axs2 = plt.subplots(2)
axs2[0].plot(t_anal, v_analytical)
axs2[0].plot(t_history, np.array(x_euler_history)[:,0], alpha = 0.6)
axs2[0].plot(t_history, np.array(x_heun_history)[:,0], alpha = 0.6)
axs2[0].plot(t_history, np.array(x_RK4_history)[:,0], alpha = 0.6)
axs2[0].set_title("Numerical and Analytical Integration Methods, V(t)")
axs2[0].legend(['Analytical','Euler', 'Heun', 'RK4'])

axs2[1].plot(t_anal, np.array(x_euler_history)[:,0] - v_analytical, alpha = 0.6)
axs2[1].plot(t_anal, np.array(x_heun_history)[:,0] - v_analytical, alpha = 0.6)
axs2[1].plot(t_anal, np.array(x_RK4_history)[:,0] - v_analytical, alpha = 0.6)
axs2[1].set_title("Numerical Integration Methods Error")
axs2[1].legend(['Euler error', 'Heun error', 'RK4 error'])

plt.show()








