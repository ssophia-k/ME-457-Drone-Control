class Integrator:
    """Integrator for a system of first-order ordinary differential equations
    of the form \dot x = f(t, x, u).
    """
    def __init__(self, dt, f):
        self.dt = dt
        self.f = f

    def step(self, t, x, u):
        raise NotImplementedError

class Euler(Integrator):
    def step(self, t, x, u):
        return x + self.dt * self.f(t, x, u)

class Heun(Integrator):
    def step(self, t, x, u):
        intg = Euler(self.dt, self.f)
        xe = intg.step(t, x, u) # Euler predictor step
        return x + 0.5*self.dt * (self.f(t, x, u) + self.f(t+self.dt, xe, u))

class RK4(Integrator):
    def step(self, t, x, u):
        k1 = self.dt * self.f(t, x, u)
        k2 = self.dt * self.f(t+self.dt/2, x+k1/2, u)
        k3 = self.dt * self.f(t+self.dt/2, x+k2/2, u)
        k4 = self.dt * self.f(t+self.dt, x+k3, u)
        return x + (k1 + 2*k2 + 2*k3 + k4)/6
