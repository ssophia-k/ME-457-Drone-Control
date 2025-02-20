import numpy as np

psi = np.radians(2)
theta = np.radians(10)
phi = np.radians(20)

A = np.array([
    [np.cos(theta) * np.cos(psi),
     np.sin(phi) * np.sin(theta) * np.cos(psi) - np.cos(phi) * np.sin(psi),
     np.cos(phi) * np.sin(theta) * np.cos(psi) + np.sin(phi) * np.sin(psi)],
    
    [np.cos(theta) * np.sin(psi),
     np.sin(phi) * np.sin(theta) * np.sin(psi) + np.cos(phi) * np.cos(psi),
     np.cos(phi) * np.sin(theta) * np.sin(psi) - np.sin(phi) * np.cos(psi)],
    
    [-np.sin(theta),
     np.sin(phi) * np.cos(theta),
     np.cos(phi) * np.cos(theta)]
])

v = np.array([15, 1, 0.5])

print(np.dot(A, v))


