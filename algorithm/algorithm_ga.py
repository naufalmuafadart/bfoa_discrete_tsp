from algorithm.algorithm import Algorithm
import random
import numpy as np


class Individual:
    def __init__(self, chromosome_length, fitness_function, list_poi, is_maximizing=True):
        self.chromosome_length = chromosome_length
        self.fitness_function = fitness_function
        self.is_maximizing = is_maximizing
        self.fitness_value = None
        self.chromosome = None

        self.generate_random_chromosome(list_poi)
        self.calculate_fitness()

    def generate_random_chromosome(self, list_poi):
        # Create a copy of the list to avoid modifying the original
        shuffled_list = list_poi.copy()
        # Shuffle the list in place
        random.shuffle(shuffled_list)
        self.chromosome = shuffled_list
        self.fitness_value = self.fitness_function(self.chromosome)

    def calculate_fitness(self):
        self.fitness_value = self.fitness_function(self.chromosome)

    def set_chromosome(self, chromosome):
        self.chromosome = chromosome
        self.calculate_fitness()


class GeneticAlgorithm:
    def __init__(self, chromosome_length, fitness, max_iter, population_size, list_poi, is_maximizing=True):
        self.chromosome_length = chromosome_length
        self.fitness_function = fitness
        self.max_iter = max_iter
        self.population_size = population_size
        self.list_poi = list_poi
        self.is_maximizing = is_maximizing

        # GA parameters
        self.crossover_rate = 0.8
        self.mutation_rate = 0.2
        self.elitism_size = 2

        # Population and tracking
        self.population = []
        self.best_solution = None
        self.best_fitness = float('-inf') if is_maximizing else float('inf')

        self.initialize_population()

    def initialize_population(self):
        """Initialize random population"""
        for _ in range(self.population_size):
            individual = Individual(
                self.chromosome_length,
                self.fitness_function,
                self.list_poi,
                self.is_maximizing
            )
            self.population.append(individual)

    def tournament_selection(self, tournament_size=3):
        """Select parent using tournament selection"""
        tournament = random.sample(self.population, tournament_size)
        if self.is_maximizing:
            return max(tournament, key=lambda x: x.fitness_value)
        return min(tournament, key=lambda x: x.fitness_value)

    def order_crossover(self, parent1, parent2):
        """Perform order crossover (OX) for permutation problems"""
        size = len(parent1.chromosome)
        # Choose crossover points
        point1, point2 = sorted(random.sample(range(size), 2))

        # Create offspring
        offspring = [None] * size
        # Copy segment from parent1
        segment = parent1.chromosome[point1:point2]
        offspring[point1:point2] = segment

        # Fill remaining positions from parent2
        remaining = [x for x in parent2.chromosome if x not in segment]
        j = 0
        for i in range(size):
            if offspring[i] is None:
                offspring[i] = remaining[j]
                j += 1

        child = Individual(self.chromosome_length, self.fitness_function, self.list_poi, self.is_maximizing)
        child.set_chromosome(offspring)
        return child

    def swap_mutation(self, individual):
        """Perform swap mutation"""
        if random.random() < self.mutation_rate:
            pos1, pos2 = random.sample(range(len(individual.chromosome)), 2)
            individual.chromosome[pos1], individual.chromosome[pos2] = \
                individual.chromosome[pos2], individual.chromosome[pos1]
            individual.calculate_fitness()

    def get_elite(self):
        """Get elite individuals from population"""
        sorted_pop = sorted(
            self.population,
            key=lambda x: x.fitness_value,
            reverse=self.is_maximizing
        )
        return sorted_pop[:self.elitism_size]

    def update_best_solution(self):
        """Update best solution found so far"""
        current_best = max(self.population, key=lambda x: x.fitness_value) if self.is_maximizing \
            else min(self.population, key=lambda x: x.fitness_value)

        if self.is_maximizing and current_best.fitness_value > self.best_fitness:
            self.best_solution = current_best.chromosome[:]
            self.best_fitness = current_best.fitness_value
        elif not self.is_maximizing and current_best.fitness_value < self.best_fitness:
            self.best_solution = current_best.chromosome[:]
            self.best_fitness = current_best.fitness_value

    def execute(self):
        """Execute the genetic algorithm"""
        for generation in range(self.max_iter):
            # Get elite individuals
            elite = self.get_elite()

            # Create new population
            new_population = []

            # Add elite individuals to new population
            new_population.extend(elite)

            # Create rest of the population through crossover and mutation
            while len(new_population) < self.population_size:
                if random.random() < self.crossover_rate:
                    parent1 = self.tournament_selection()
                    parent2 = self.tournament_selection()
                    offspring = self.order_crossover(parent1, parent2)
                    self.swap_mutation(offspring)
                    new_population.append(offspring)
                else:
                    # Copy individual from current population
                    new_population.append(random.choice(self.population))

            # Update population
            self.population = new_population

            # Update best solution
            self.update_best_solution()

        return [self.best_solution], self.best_fitness


class GA_TSP(Algorithm):
    def __init__(self, agent_length, dataset_name):
        super().__init__(agent_length, dataset_name)

    def run(self):
        algorithm = GeneticAlgorithm(
            self.AGENT_LENGTH,
            self.tsp_fitness_function,
            100,  # max iterations
            50,  # population size
            [i for i in range(1, self.AGENT_LENGTH + 1)],
            False  # minimization problem for TSP
        )
        return algorithm.execute()