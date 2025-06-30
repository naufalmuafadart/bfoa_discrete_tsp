from algorithm.algorithm import Algorithm
import random
import math
import numpy as np


class State:
    def __init__(self, solution_length, fitness_function, list_poi, is_maximizing=True):
        self.solution_length = solution_length
        self.fitness_function = fitness_function
        self.is_maximizing = is_maximizing
        self.solution = None
        self.fitness_value = None
        self.list_poi = list_poi

        self.generate_random_solution(list_poi)
        self.calculate_fitness()

    def generate_random_solution(self, list_poi):
        """Generate random initial solution"""
        shuffled_list = list_poi.copy()
        random.shuffle(shuffled_list)
        self.solution = shuffled_list
        self.calculate_fitness()

    def calculate_fitness(self):
        """Calculate fitness value of current solution"""
        self.fitness_value = self.fitness_function(self.solution)

    def set_solution(self, solution):
        """Set new solution and calculate its fitness"""
        self.solution = solution
        self.calculate_fitness()

    def copy(self):
        """Create a deep copy of current state"""
        new_state = State(self.solution_length, self.fitness_function, self.list_poi, self.is_maximizing)
        new_state.solution = self.solution.copy()
        new_state.fitness_value = self.fitness_value
        return new_state


class SimulatedAnnealing:
    def __init__(self, solution_length, fitness, max_iter, N, list_poi, is_maximizing=True):
        self.solution_length = solution_length
        self.fitness_function = fitness
        self.max_iter = max_iter
        self.list_poi = list_poi
        self.is_maximizing = is_maximizing

        # SA specific parameters
        self.initial_temperature = 100.0
        self.final_temperature = 80
        # self.final_temperature = 1e-8
        self.alpha = 0.95  # cooling rate

        # Current state and best solution tracking
        self.current_state = None
        self.best_solution = None
        self.best_fitness = float('-inf') if is_maximizing else float('inf')

        self.initialize()

    def initialize(self):
        """Initialize the algorithm with random solution"""
        self.current_state = State(
            self.solution_length,
            self.fitness_function,
            self.list_poi,
            self.is_maximizing
        )
        self.update_best_solution()

    def update_best_solution(self):
        """Update best solution if current solution is better"""
        if self.is_better(self.current_state.fitness_value, self.best_fitness):
            self.best_solution = self.current_state.solution[:]
            self.best_fitness = self.current_state.fitness_value

    def is_better(self, value1, value2):
        """Check if value1 is better than value2 based on optimization direction"""
        if self.is_maximizing:
            return value1 > value2
        return value1 < value2

    def acceptance_probability(self, current_energy, new_energy, temperature):
        """Calculate probability of accepting worse solution"""
        if self.is_better(new_energy, current_energy):
            return 1.0

        # For minimization, we use (current - new)
        # For maximization, we use (new - current)
        delta = current_energy - new_energy if not self.is_maximizing else new_energy - current_energy
        return math.exp(delta / temperature)

    def generate_neighbor(self, current_state):
        """Generate neighbor solution using swap operation"""
        neighbor = current_state.copy()

        # Randomly choose neighbor generation method
        method = random.choice(['swap', 'reverse', 'insert'])

        if method == 'swap':
            # Swap two random positions
            i, j = random.sample(range(self.solution_length), 2)
            neighbor.solution[i], neighbor.solution[j] = neighbor.solution[j], neighbor.solution[i]

        elif method == 'reverse':
            # Reverse a random subsequence
            i, j = sorted(random.sample(range(self.solution_length), 2))
            neighbor.solution[i:j + 1] = reversed(neighbor.solution[i:j + 1])

        else:  # insert
            # Remove from one position and insert at another
            i, j = random.sample(range(self.solution_length), 2)
            value = neighbor.solution.pop(i)
            neighbor.solution.insert(j, value)

        neighbor.calculate_fitness()
        return neighbor

    def execute(self):
        """Execute the simulated annealing algorithm"""
        temperature = self.initial_temperature

        # Main loop
        while temperature > self.final_temperature:
            for _ in range(self.solution_length):  # Inner loop iterations
                # Generate neighbor solution
                neighbor_state = self.generate_neighbor(self.current_state)

                # Calculate acceptance probability
                ap = self.acceptance_probability(
                    self.current_state.fitness_value,
                    neighbor_state.fitness_value,
                    temperature
                )

                # Accept or reject neighbor solution
                if random.random() < ap:
                    self.current_state = neighbor_state
                    self.update_best_solution()

            # Cool down
            temperature *= self.alpha

        return [self.best_solution], self.best_fitness


class SA_TSP(Algorithm):
    def __init__(self, agent_length, dataset_name):
        super().__init__(agent_length, dataset_name)

    def run(self):
        algorithm = SimulatedAnnealing(
            self.AGENT_LENGTH,
            self.tsp_fitness_function,
            2,  # max iterations
            1,  # not used in SA but kept for interface consistency
            [i for i in range(1, self.AGENT_LENGTH + 1)],
            False  # minimization problem for TSP
        )
        return algorithm.execute()
