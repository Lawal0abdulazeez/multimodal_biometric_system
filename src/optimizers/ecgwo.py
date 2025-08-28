# src/optimizers/ecgwo.py
import numpy as np
from tqdm import tqdm
from .gwo import GreyWolfOptimizer # Import the base class

class EnhancedChaoticGWO(GreyWolfOptimizer):
    """
    Enhanced Chaotic Grey Wolf Optimizer.
    Inherits from the standard GWO and modifies the main loop
    to use a chaotic tent map for the 'a' parameter.
    """
    def __init__(self, fitness_function, dim, num_wolves, max_iter, lower_bound, upper_bound, chaos_map_type='tent'):
        """
        Initializes the ECGWO algorithm.
        
        Args:
            chaos_map_type (str): The type of chaotic map to use ('tent' is the primary one for this research).
        """
        super().__init__(fitness_function, dim, num_wolves, max_iter, lower_bound, upper_bound)
        self.chaos_map_type = chaos_map_type
        
        # Generate the chaotic sequence for 'a'
        self.chaotic_a_sequence = self._generate_chaotic_sequence()

    def _tent_map(self, x, mu=0.7):
        """The tent map function."""
        if x < mu:
            return x / mu
        else:
            return (1 - x) / (1 - mu)

    def _generate_chaotic_sequence(self):
        """Generates a chaotic sequence for the 'a' parameter."""
        sequence = np.zeros(self.max_iter)
        # Initialize with a random value, avoiding common fixed points like 0 or 0.5
        sequence[0] = np.random.uniform(0.01, 0.99)
        
        for t in range(1, self.max_iter):
            if self.chaos_map_type == 'tent':
                sequence[t] = self._tent_map(sequence[t-1])
            else:
                # Can add other maps like logistic map here if needed
                raise NotImplementedError("Only 'tent' map is implemented.")
                
        # Scale the chaotic sequence (from [0,1]) to the desired range [2,0] for 'a'
        return 2 * (1 - sequence)

    def optimize(self):
        """
        Runs the main ECGWO optimization loop.
        This method OVERRIDES the parent GWO's optimize method.
        """
        self._initialize_wolves()
        self._update_leaders()

        for t in tqdm(range(self.max_iter), desc="ECGWO Optimization"):
            # --- Key Difference ---
            # Get the 'a' value from our pre-computed chaotic sequence
            a = self.chaotic_a_sequence[t]
            
            # The rest of the loop is identical to the standard GWO
            for i in range(self.num_wolves):
                # Update position based on alpha, beta, and delta wolves
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = np.abs(C1 * self.alpha_pos - self.positions[i, :])
                X1 = self.alpha_pos - A1 * D_alpha

                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = np.abs(C2 * self.beta_pos - self.positions[i, :])
                X2 = self.beta_pos - A2 * D_beta

                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = np.abs(C3 * self.delta_pos - self.positions[i, :])
                X3 = self.delta_pos - A3 * D_delta

                new_position = (X1 + X2 + X3) / 3.0
                new_position = np.clip(new_position, self.lb, self.ub)

                new_fitness = self.fitness_func(new_position)

                if new_fitness < self.fitness[i]:
                    self.positions[i, :] = new_position
                    self.fitness[i] = new_fitness

            self._update_leaders()
            self.convergence_curve[t] = self.alpha_fitness
        
        return self.alpha_fitness, self.alpha_pos