"""
Strictly Positive Genetic Algorithm Optimization for UAV Autopilot LQR Gains
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, os.fspath(Path(__file__).parents[1]))
import Parameters.control_parameters as AP

import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
from scipy.linalg import solve_continuous_are, inv

# Import necessary simulation components
import Parameters.simulation_parameters as SIM
from Tools.signals import Signals
from Models.dynamics_control import MavDynamics
from Models.wind import WindSimulation
from Controllers.autopilot_lqr import Autopilot
from Message_types.autopilot import MsgAutopilot
import Models.model_coef as M

def run_simulation(Qlat_params, Qlon_params, Rlat_params, Rlon_params):
    """
    Run simulation with given LQR gain parameters and compute performance metrics
    """
    # Ensure all parameters are positive
    Qlat_params = np.maximum(np.abs(Qlat_params), 0.01)
    Qlon_params = np.maximum(np.abs(Qlon_params), 0.01)
    Rlat_params = np.maximum(np.abs(Rlat_params), 0.01)
    Rlon_params = np.maximum(np.abs(Rlon_params), 0.01)

    # Reset simulation components
    wind = WindSimulation(SIM.ts_simulation)
    mav = MavDynamics(SIM.ts_simulation)
    mav._state = M.x_trim

    # Create autopilot with custom LQR gains
    class CustomAutopilot(Autopilot):
        def __init__(self, ts_control, Qlat_params, Qlon_params, Rlat_params, Rlon_params):
            # Initialize with default constructor
            super().__init__(ts_control)
            
            # Ensure positive values and create diag matrices
            Qlat = np.diag(Qlat_params)
            Qlon = np.diag(Qlon_params)
            Rlat = np.diag(Rlat_params)
            Rlon = np.diag(Rlon_params)
            
            # Recompute lateral gains using solve_continuous_are
            CrLat = np.array([[0, 0, 0, 0, 1.0]])
            AAlat = np.concatenate((
                        np.concatenate((M.A_lat, np.zeros((5,1))), axis=1),
                        np.concatenate((CrLat, np.zeros((1,1))), axis=1)),
                        axis=0)
            BBlat = np.concatenate((M.B_lat, np.zeros((1,2))), axis=0)
            
            try:
                # Use solve_continuous_are for more robust solution
                Plat = solve_continuous_are(AAlat, BBlat, Qlat, Rlat)
                self.Klat = inv(Rlat) @ BBlat.T @ Plat
            except Exception as e:
                print(f"Lateral gain computation error: {e}")
                # Use a reasonable default if computation fails
                self.Klat = np.array([
                    [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                    [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
                ])
            
            # Recompute longitudinal gains
            CrLon = np.array([[0, 0, 0, 0, 1.0], [1/AP.Va0, 1/AP.Va0, 0, 0, 0]])
            AAlon = np.concatenate((
                        np.concatenate((M.A_lon, np.zeros((5,2))), axis=1),
                        np.concatenate((CrLon, np.zeros((2,2))), axis=1)),
                        axis=0)
            BBlon = np.concatenate((M.B_lon, np.zeros((2, 2))), axis=0)
            
            try:
                # Use solve_continuous_are for more robust solution
                Plon = solve_continuous_are(AAlon, BBlon, Qlon, Rlon)
                self.Klon = inv(Rlon) @ BBlon.T @ Plon
            except Exception as e:
                print(f"Longitudinal gain computation error: {e}")
                # Use a reasonable default if computation fails
                self.Klon = np.array([
                    [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                    [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
                ])

    # Simulation commands
    commands = MsgAutopilot()
    Va_command = Signals(dc_offset=25.0, amplitude=3.0, start_time=2.0, frequency=0.01)
    altitude_command = Signals(dc_offset=100.0, amplitude=20.0, start_time=0.0, frequency=0.02)
    course_command = Signals(dc_offset=np.radians(180), amplitude=np.radians(45), start_time=5.0, frequency=0.015)

    # Performance tracking
    Va_errors = []
    altitude_errors = []
    course_errors = []

    # Initialize autopilot with custom gains
    autopilot = CustomAutopilot(SIM.ts_simulation, Qlat_params, Qlon_params, Rlat_params, Rlon_params)

    # Simulation loop
    sim_time = SIM.start_time
    end_time = 75  # Reduced simulation time for faster optimization

    while sim_time < end_time:
        # Generate commands
        current_Va_command = Va_command.square(sim_time)
        current_course_command = course_command.square(sim_time)
        current_altitude_command = altitude_command.square(sim_time)
        
        commands.airspeed_command = current_Va_command
        commands.course_command = current_course_command
        commands.altitude_command = current_altitude_command

        # Run autopilot
        estimated_state = mav.true_state
        delta, commanded_state = autopilot.update(commands, estimated_state)

        # Update physical system
        current_wind = wind.update()
        mav.update(delta, current_wind)

        # Compute tracking errors with weights
        Va_errors.append(abs(mav.true_state.Va - current_Va_command))
        altitude_errors.append(abs(mav.true_state.altitude - current_altitude_command))
        course_errors.append(abs(mav.true_state.chi - current_course_command))

        # Increment time
        sim_time += SIM.ts_simulation

    # Compute weighted performance metrics
    performance_score = (
        np.mean(Va_errors) + 
        np.mean(altitude_errors) + 
        np.mean(course_errors)
    )

    return performance_score

def genetic_optimize():
    # Create fitness and individual classes
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    # Genetic Algorithm toolbox
    toolbox = base.Toolbox()

    # Logarithmic parameter generation for more uniform exploration of positive values
    def positive_log_uniform(low=0.01, high=100):
        return abs(np.exp(np.random.uniform(np.log(low), np.log(high))))

    # Parameter generation for Qlat (6 parameters)
    for i in range(6):
        toolbox.register(f"attr_qlat_{i}", positive_log_uniform)
    
    # Parameter generation for Qlon (7 parameters)
    for i in range(7):
        toolbox.register(f"attr_qlon_{i}", positive_log_uniform)

    # Parameter generation for Rlat (2 parameters)
    for i in range(2):
        toolbox.register(f"attr_rlat_{i}", positive_log_uniform, low=0.01, high=10)
    
    # Parameter generation for Rlon (2 parameters)
    for i in range(2):
        toolbox.register(f"attr_rlon_{i}", positive_log_uniform, low=0.01, high=10)

    # Individual and population creation
    def create_individual():
        return creator.Individual([
            abs(toolbox.__dict__[f'attr_qlat_{i}']()) for i in range(6)] + 
            [abs(toolbox.__dict__[f'attr_qlon_{i}']()) for i in range(7)] +
            [abs(toolbox.__dict__[f'attr_rlat_{i}']()) for i in range(2)] +
            [abs(toolbox.__dict__[f'attr_rlon_{i}']()) for i in range(2)]
        )

    toolbox.register("individual", create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Evaluation function
    def evaluate(individual):
        Qlat_params = individual[:6]
        Qlon_params = individual[6:13]
        Rlat_params = individual[13:15]
        Rlon_params = individual[15:17]
        return run_simulation(Qlat_params, Qlon_params, Rlat_params, Rlon_params),

    # Genetic operators
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.5, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Genetic Algorithm parameters
    population_size = 20
    num_generations = 10
    crossover_prob = 0.7
    mutation_prob = 0.2

    # Initial population
    population = toolbox.population(n=population_size)

    # Statistics tracking
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # Run the genetic algorithm
    population, logbook = algorithms.eaSimple(
        population, 
        toolbox, 
        cxpb=crossover_prob, 
        mutpb=mutation_prob, 
        ngen=num_generations, 
        stats=stats, 
        verbose=True
    )

    # Find the best individual
    best_ind = tools.selBest(population, k=1)[0]
    best_Qlat = best_ind[:6]
    best_Qlon = best_ind[6:13]
    best_Rlat = best_ind[13:15]
    best_Rlon = best_ind[15:17]

    print("\nBest Individual:")
    print("Qlat:", best_Qlat)
    print("Qlon:", best_Qlon)
    print("Rlat:", best_Rlat)
    print("Rlon:", best_Rlon)
    print("Fitness:", best_ind.fitness.values[0])

    return best_Qlat, best_Qlon, best_Rlat, best_Rlon

# Run the optimization
if __name__ == "__main__":
    import time
    start_time = time.time()
    best_Qlat, best_Qlon, best_Rlat, best_Rlon = genetic_optimize()
    end_time = time.time()
    print(f"Optimization took {end_time - start_time:.2f} seconds")