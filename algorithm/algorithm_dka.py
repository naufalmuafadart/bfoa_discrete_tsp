from algorithm.algorithm import Algorithm
import random
import math
import copy

class Komodo:
    def __init__(self, chromosome_length, fitness_function, list_poi, is_maximizing=True):
        self.chromosome_length = chromosome_length
        self.fitness_function = fitness_function
        self.is_maximizing = is_maximizing

        # Current position (permutation)
        self.position = None
        self.fitness_value = None

        self.generate_random_position(list_poi)

    def generate_random_position(self, list_poi):
        self.position = list_poi.copy()
        random.shuffle(self.position)
        self.calculate_fitness()

    def calculate_fitness(self):
        self.fitness_value = self.fitness_function(self.position)


class DiscreteKomodoAlgorithm:
    def __init__(self, chromosome_length, fitness, max_iter, population_size, list_poi, is_maximizing=True):
        self.chromosome_length = chromosome_length
        self.fitness_function = fitness
        self.max_iter = max_iter
        self.population_size = population_size
        self.list_poi = list_poi
        self.is_maximizing = is_maximizing

        # DKA parameters
        self.p = 0.5  # portion of big male komodo
        self.smep = 5  # small male exploration probability (0,10)

        # Calculate population segments
        self.n_big_male = int((self.p * self.population_size) - 1)
        self.n_female = 1
        self.n_small_male = self.population_size - (self.n_big_male + self.n_female)

        # Population and tracking
        self.population = []
        self.best_solution = None
        self.best_fitness = float('-inf') if is_maximizing else float('inf')

        self.initialize_population()

    def initialize_population(self):
        """Initialize random population of komodos"""
        for _ in range(self.population_size):
            komodo = Komodo(
                self.chromosome_length,
                self.fitness_function,
                self.list_poi,
                self.is_maximizing
            )
            self.population.append(komodo)

            # Update best solution if necessary
            if self.is_better(komodo.fitness_value, self.best_fitness):
                self.best_fitness = komodo.fitness_value
                self.best_solution = komodo.position[:]

    def is_better(self, new_fitness, current_fitness):
        """Helper function to compare fitness values based on optimization direction"""
        return (self.is_maximizing and new_fitness > current_fitness) or \
            (not self.is_maximizing and new_fitness < current_fitness)

    def swap_operator(self, position):
        """Simple swap operator for mutation"""
        new_pos = position[:]
        i, j = random.sample(range(len(new_pos)), 2)
        new_pos[i], new_pos[j] = new_pos[j], new_pos[i]
        return new_pos

    def edge_construction(self, current_pos, target_pos):
        """Edge construction operator"""
        new_pos = current_pos[:]

        # Randomly select and copy a segment from target
        start = random.randint(0, len(current_pos) - 2)
        end = random.randint(start + 1, len(current_pos) - 1)
        segment = target_pos[start:end + 1]

        # Remove segment elements from current position
        remaining = [x for x in new_pos if x not in segment]

        # Insert segment at random position
        insert_pos = random.randint(0, len(remaining))
        new_pos = remaining[:insert_pos] + segment + remaining[insert_pos:]

        return new_pos

    def update_komodo(self, komodo, a):
        """Update komodo's position"""
        new_position = None
        new_fitness = None

        if random.random() < a:  # Exploitation (edge construction)
            if random.random() < 0.5:  # Move towards best solution
                new_position = self.edge_construction(komodo.position, self.best_solution)
            else:  # Move towards random komodo
                random_komodo = random.choice(self.population)
                new_position = self.edge_construction(komodo.position, random_komodo.position)
        else:  # Exploration (swap operator)
            new_position = self.swap_operator(komodo.position)

        new_fitness = self.fitness_function(new_position)

        # Create new komodo with updated position
        new_komodo = Komodo(self.chromosome_length, self.fitness_function, self.list_poi, self.is_maximizing)
        new_komodo.position = new_position
        new_komodo.fitness_value = new_fitness

        return new_komodo

    def execute(self):
        """Execute the DKA algorithm"""
        for iteration in range(self.max_iter):
            # Control parameter for exploration/exploitation
            a = 1 - (iteration / self.max_iter)

            # Update population
            new_population = []

            # Update big males
            for i in range(self.n_big_male):
                new_komodo = self.update_komodo(self.population[i], a)
                new_population.append(new_komodo)

                # Update best solution if necessary
                if self.is_better(new_komodo.fitness_value, self.best_fitness):
                    self.best_fitness = new_komodo.fitness_value
                    self.best_solution = new_komodo.position[:]

            # Update female
            female = self.update_komodo(self.population[self.n_big_male], a)
            new_population.append(female)

            # Update small males
            for i in range(self.n_big_male + 1, self.population_size):
                if random.randint(0, 10) <= self.smep:
                    new_komodo = self.update_komodo(self.population[i], 0)  # More exploration
                else:
                    new_komodo = self.update_komodo(self.population[i], 1)  # More exploitation
                new_population.append(new_komodo)

            # Update population
            self.population = new_population

        return [self.best_solution], self.best_fitness


class DKA_TSP(Algorithm):
    def __init__(self, agent_length, dataset_name):
        super().__init__(agent_length, dataset_name)

    def run(self):
        algorithm = DiscreteKomodoAlgorithm(
            self.AGENT_LENGTH,
            self.tsp_fitness_function,
            25,  # max iterations
            10,  # population size
            [i for i in range(1, self.AGENT_LENGTH + 1)],
            False  # minimization problem for TSP
        )
        return algorithm.execute()
