import random
import math
from algorithm.algorithm import Algorithm
import numpy as np


# --- 1. Swap Operator and Swap Sequence ---

def apply_swap_operator(tour, i, j):
    """
    Applies the Swap Operator SO(i, j) to a tour.
    This function swaps the cities at indices i and j in the tour list.
    Note: The paper uses 1-based indexing for cities, but Python uses 0-based.
    We operate on the indices of the tour list directly.
    """
    new_tour = tour[:]  # Create a copy
    new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
    return new_tour


def apply_swap_sequence(tour, swap_sequence):
    """
    Applies a sequence of Swap Operators to a tour.
    The swap_sequence is a list of tuples, where each tuple (i, j) represents a Swap Operator.
    """
    new_tour = tour[:]
    for i, j in swap_sequence:
        new_tour = apply_swap_operator(new_tour, i, j)
    return new_tour


def generate_random_swap_sequence(num_cities, num_swaps):
    """
    Generates a random Swap Sequence.
    """
    swap_sequence = []
    for _ in range(num_swaps):
        # Generate two distinct random indices to swap
        i, j = random.sample(range(num_cities), 2)
        swap_sequence.append((i, j))
    return swap_sequence


# --- 2. Discrete Artificial Bee Colony (DABC) Algorithm ---

class DABC:
    """
    Implements the Discrete Artificial Bee Colony algorithm for the TSP.
    """

    def __init__(self, cities, colony_size, n_se, limit, max_cycles, fitness_function):
        self.cities = cities
        self.num_cities = len(cities)
        self.colony_size = colony_size
        self.num_employed = self.colony_size // 2
        self.num_onlooker = self.colony_size // 2
        self.n_se = n_se  # Number of Swap Operators in a Swap Sequence
        self.limit = limit  # Abandonment limit
        self.max_cycles = max_cycles
        self.fitness_function = fitness_function

        # --- Bee Colony Data ---
        # Each food source is a tour (a permutation of city indices)
        self.food_sources = [self._generate_random_tour() for _ in range(self.num_employed)]
        self.fitness_values = [self.calculate_tour_fitness(tour, self.cities) for tour in self.food_sources]

        # Trial counters for each food source (for scout phase)
        self.trial_counters = [0] * self.num_employed

        # Keep track of the best solution found so far
        self.best_tour = self.food_sources[np.argmax(self.fitness_values)]
        self.best_fitness = max(self.fitness_values)
        self.best_tour_length = self.get_tour_length(self.best_tour, self.cities)

        # For plotting convergence
        self.convergence_curve = []

    @staticmethod
    def calculate_distance(city1, city2):
        """Calculates the Euclidean distance between two cities."""
        return math.sqrt((city1[0] - city2[0]) ** 2 + (city1[1] - city2[1]) ** 2)

    def calculate_tour_fitness(self, tour, cities):
        # Avoid division by zero if distance is 0
        return 1 / self.fitness_function(tour)

    def get_tour_length(self, tour, cities):
        """Calculates the raw tour length (distance)."""
        return 1 / self.calculate_tour_fitness(tour, cities) if self. calculate_tour_fitness(tour, cities) != float('inf') else 0

    def _generate_random_tour(self):
        """Generates a random tour (a shuffled list of city indices)."""
        tour = list(range(self.num_cities))
        random.shuffle(tour)
        return tour

    def execute(self):
        """Executes the main loop of the DABC algorithm."""
        print("DABC algorithm started...")
        for cycle in range(self.max_cycles):
            self.employed_bee_phase()
            self.onlooker_bee_phase()
            self.scout_bee_phase()
            self.memorize_best_solution()

            # Store data for convergence plot
            self.convergence_curve.append(self.best_tour_length)

            if (cycle + 1) % 100 == 0:
                print(f"Cycle {cycle + 1}/{self.max_cycles} | Best Tour Length: {self.best_tour_length:.2f}")

        print("DABC algorithm finished.")
        return self.best_tour, self.best_tour_length

    def employed_bee_phase(self):
        """
        The phase where employed bees search for new food sources (tours)
        around their current source.
        """
        for i in range(self.num_employed):
            current_tour = self.food_sources[i]

            # Generate a new candidate tour using a Swap Sequence
            swap_sequence = generate_random_swap_sequence(self.num_cities, self.n_se)
            candidate_tour = apply_swap_sequence(current_tour, swap_sequence)

            # Evaluate the new tour
            candidate_fitness = self.calculate_tour_fitness(candidate_tour, self.cities)

            # Greedy selection: if the new tour is better, update the bee's memory
            if candidate_fitness > self.fitness_values[i]:
                self.food_sources[i] = candidate_tour
                self.fitness_values[i] = candidate_fitness
                self.trial_counters[i] = 0  # Reset trial counter
            else:
                self.trial_counters[i] += 1  # Increment trial counter

    def onlooker_bee_phase(self):
        """
        The phase where onlooker bees select a food source based on its quality (fitness)
        and then search for a better one in its neighborhood.
        """
        # Calculate selection probabilities using roulette wheel
        total_fitness = sum(self.fitness_values)
        probabilities = [f / total_fitness for f in
                         self.fitness_values] if total_fitness > 0 else [1 / self.num_employed] * self.num_employed

        for _ in range(self.num_onlooker):
            # Select a food source based on probability
            selected_index = np.random.choice(range(self.num_employed), p=probabilities)

            current_tour = self.food_sources[selected_index]

            # Generate a candidate tour using a Swap Sequence
            swap_sequence = generate_random_swap_sequence(self.num_cities, self.n_se)
            candidate_tour = apply_swap_sequence(current_tour, swap_sequence)

            # Evaluate the new tour
            candidate_fitness = self.calculate_tour_fitness(candidate_tour, self.cities)

            # Greedy selection
            if candidate_fitness > self.fitness_values[selected_index]:
                self.food_sources[selected_index] = candidate_tour
                self.fitness_values[selected_index] = candidate_fitness
                self.trial_counters[selected_index] = 0
            else:
                self.trial_counters[selected_index] += 1

    def scout_bee_phase(self):
        """
        The phase where an employed bee whose food source has been exhausted
        (exceeded the 'limit') becomes a scout and finds a new random food source.
        """
        for i in range(self.num_employed):
            if self.trial_counters[i] > self.limit:
                # Abandon the old food source and become a scout
                # The scout finds a new random food source (tour)
                self.food_sources[i] = self._generate_random_tour()
                self.fitness_values[i] = self.calculate_tour_fitness(self.food_sources[i], self.cities)
                self.trial_counters[i] = 0  # Reset the trial counter for the new source

    def memorize_best_solution(self):
        """Finds and stores the best solution found in the current cycle."""
        current_best_fitness = max(self.fitness_values)
        if current_best_fitness > self.best_fitness:
            self.best_fitness = current_best_fitness
            best_index = np.argmax(self.fitness_values)
            self.best_tour = self.food_sources[best_index]
            self.best_tour_length = self.get_tour_length(self.best_tour, self.cities)

class DABC_TSP(Algorithm):
    def __init__(self, agent_length, dataset_name):
        super().__init__(agent_length, dataset_name)

    def run(self, cities, colony_size=50, n_se=10, limit=20, max_cycles=1000):
        algorithm = DABC(cities, colony_size, n_se, limit, max_cycles, self.tsp_fitness_function)
        return algorithm.execute()
