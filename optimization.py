from scipy.optimize import minimize
import numpy as np

def objective(x):
    return x[0]**2 + x[1]**2

x0 = np.array([[5], [2]]) #Initial conditions
cons = ({'type': 'eq', 'fun': lambda x: x[0] + x[1] - 1})