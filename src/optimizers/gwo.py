# src/optimizers/gwo.py
import numpy as np
import random
from tqdm import tqdm

class GreyWolfOptimizer:
    """
    Standard Grey Wolf Optimizer.
    """
    def __init__(self, fitness_function, dim, num_wolves, max_iter, lower_bound, upper_bound):
        """
        Initializes the GWO algorithm.

        Args:
            fitness_function (function): The objective function to minimize. It must accept
                                         a 1D NumPy array (a wolf's position) and return a scalar fitness value.
            dim (int): The number of dimensions (variables) in the problem.
            num_wolves (int): The number of search agents (wolves).
            max_iter (int): The maximum number of iterations.
            lower_bound (float or list/array): The lower bound for each dimension.
            upper_bound (float or list/array): The upper bound for each dimension.
        """
        self.fitness_func = fitness_function
        self.dim = dim
        self.num_wolves = num_wolves
        self.max_iter = max_iter

        # Handle bounds if they are single values
        if isinstance(lower_bound, (int, float)):
            self.lb = np.full(dim, lower_bound)
        else:
            self.lb = np.array(lower_bound)
            
        if isinstance(upper_bound, (int, float)):
            self.ub = np.full(dim, upper_bound)
        else:
            self.ub = np.array(upper_bound)
            
        # Initialize wolf positions and fitness
        self.positions = np.zeros((self.num_wolves, self.dim))
        self.fitness = np.full(self.num_wolves, np.inf)

        # Track the leaders
        self.alpha_pos = np.zeros(self.dim)
        self.alpha_fitness = np.inf
        self.beta_pos = np.zeros(self.dim)
        self.beta_fitness = np.inf
        self.delta_pos = np.zeros(self.dim)
        self.delta_fitness = np.inf

        self.convergence_curve = np.zeros(self.max_iter)

    def _initialize_wolves(self):
        """Initializes the positions of the wolves within the search space."""
        for i in range(self.num_wolves):
            self.positions[i, :] = self.lb + (self.ub - self.lb) * np.random.rand(self.dim)
            self.fitness[i] = self.fitness_func(self.positions[i, :])

    def _update_leaders(self):
        """Updates the alpha, beta, and delta wolves based on current fitness."""
        # Create a sorted list of wolves by fitness
        sorted_indices = np.argsort(self.fitness)
        
        self.alpha_pos = self.positions[sorted_indices[0], :].copy()
        self.alpha_fitness = self.fitness[sorted_indices[0]]
        
        self.beta_pos = self.positions[sorted_indices[1], :].copy()
        self.beta_fitness = self.fitness[sorted_indices[1]]
        
        self.delta_pos = self.positions[sorted_indices[2], :].copy()
        self.delta_fitness = self.fitness[sorted_indices[2]]

    def optimize(self):
        """
        Runs the main GWO optimization loop.
        
        Returns:
            tuple: A tuple containing the best fitness value and the best position found.
        """
        self._initialize_wolves()
        self._update_leaders()

        for t in tqdm(range(self.max_iter), desc="GWO Optimization"):
            # The 'a' parameter decreases linearly from 2 to 0
            a = 2 - t * (2 / self.max_iter)

            for i in range(self.num_wolves):
                # Update position based on alpha, beta, and delta wolves
                # --- Alpha wolf influence ---
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = np.abs(C1 * self.alpha_pos - self.positions[i, :])
                X1 = self.alpha_pos - A1 * D_alpha

                # --- Beta wolf influence ---
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = np.abs(C2 * self.beta_pos - self.positions[i, :])
                X2 = self.beta_pos - A2 * D_beta

                # --- Delta wolf influence ---
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = np.abs(C3 * self.delta_pos - self.positions[i, :])
                X3 = self.delta_pos - A3 * D_delta

                # --- Calculate the new position ---
                new_position = (X1 + X2 + X3) / 3.0

                # Clip the position to stay within bounds
                new_position = np.clip(new_position, self.lb, self.ub)

                # Calculate fitness of the new position
                new_fitness = self.fitness_func(new_position)

                # Update if the new position is better
                if new_fitness < self.fitness[i]:
                    self.positions[i, :] = new_position
                    self.fitness[i] = new_fitness

            # Update the leaders and record the best fitness for this iteration
            self._update_leaders()
            self.convergence_curve[t] = self.alpha_fitness
        
        return self.alpha_fitness, self.alpha_pos