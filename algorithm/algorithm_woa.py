from algorithm.algorithm import Algorithm
import random
import math


class Whale:
    def __init__(self, chromosome_length, fitness_function, list_poi, is_maximizing=True):
        self.chromosome_length = chromosome_length
        self.fitness_function = fitness_function
        self.is_maximizing = is_maximizing
        
        # Current position (permutation)
        self.position = None
        self.fitness_value = None
        
        self.generate_random_position(list_poi)
        
    def generate_random_position(self, list_poi):
        # Create a random permutation
        self.position = list_poi.copy()
        random.shuffle(self.position)
        self.calculate_fitness()
        
    def calculate_fitness(self):
        self.fitness_value = self.fitness_function(self.position)


class WhaleOptimizationAlgorithm:
    def __init__(self, chromosome_length, fitness, max_iter, population_size, list_poi, is_maximizing=True):
        self.chromosome_length = chromosome_length
        self.fitness_function = fitness
        self.max_iter = max_iter
        self.population_size = population_size
        self.list_poi = list_poi
        self.is_maximizing = is_maximizing
        
        # WOA parameters
        self.a_decrease_factor = 2.0  # Linearly decreased from 2 to 0
        self.b = 1.0  # Spiral shape constant
        
        # Population and tracking
        self.population = []
        self.best_solution = None
        self.best_fitness = float('-inf') if is_maximizing else float('inf')
        
        self.initialize_population()
        
    def initialize_population(self):
        """Initialize random population of whales"""
        for _ in range(self.population_size):
            whale = Whale(
                self.chromosome_length,
                self.fitness_function,
                self.list_poi,
                self.is_maximizing
            )
            self.population.append(whale)
            
            # Update best solution if necessary
            if (self.is_maximizing and whale.fitness_value > self.best_fitness) or \
               (not self.is_maximizing and whale.fitness_value < self.best_fitness):
                self.best_fitness = whale.fitness_value
                self.best_solution = whale.position[:]
    
    def apply_bubble_net_attack(self, current_pos, best_pos, a):
        """Simulate bubble-net attacking method (exploitation)"""
        new_pos = current_pos[:]
        
        # Spiral updating position
        if random.random() < 0.5:
            # Create spiral movement using swap sequences
            l = random.uniform(-1, 1)
            distance = self.get_distance(current_pos, best_pos)
            num_swaps = int(abs(distance * math.exp(self.b * l) * math.cos(2 * math.pi * l)))
            
            # Perform swaps to move towards best position
            for _ in range(min(num_swaps, len(current_pos))):
                i, j = random.sample(range(len(current_pos)), 2)
                new_pos[i], new_pos[j] = new_pos[j], new_pos[i]
        
        # Shrinking encircling mechanism
        else:
            r = random.random()
            A = 2 * a * r - a  # A decreases linearly from a to -a
            C = 2 * r  # Random weight between 0 and 2
            
            if abs(A) < 1:  # Exploitation
                # Move towards best solution using swaps
                num_swaps = int(abs(A) * len(current_pos))
                target = best_pos
            else:  # Exploration
                # Move towards random whale
                random_whale = random.choice(self.population)
                target = random_whale.position
                num_swaps = int(C * len(current_pos))
            
            # Perform swaps to move towards target
            current = new_pos[:]
            for i in range(len(current)):
                if current[i] != target[i] and num_swaps > 0:
                    j = current.index(target[i])
                    new_pos[i], new_pos[j] = new_pos[j], new_pos[i]
                    num_swaps -= 1
        
        return new_pos
    
    def get_distance(self, pos1, pos2):
        """Calculate normalized distance between two permutations"""
        distance = 0
        for i in range(len(pos1)):
            if pos1[i] != pos2[i]:
                distance += 1
        return distance / len(pos1)
    
    def update_whale(self, whale, a):
        """Update whale's position"""
        new_position = self.apply_bubble_net_attack(whale.position, self.best_solution, a)
        
        # Create new whale with updated position
        new_whale = Whale(self.chromosome_length, self.fitness_function, self.list_poi, self.is_maximizing)
        new_whale.position = new_position
        new_whale.calculate_fitness()
        
        return new_whale
    
    def execute(self):
        """Execute the WOA algorithm"""
        for iteration in range(self.max_iter):
            # Calculate a (linearly decreased from 2 to 0)
            a = self.a_decrease_factor * (1 - iteration / self.max_iter)
            
            # Update each whale's position
            new_population = []
            for whale in self.population:
                new_whale = self.update_whale(whale, a)
                new_population.append(new_whale)
                
                # Update best solution if necessary
                if (self.is_maximizing and new_whale.fitness_value > self.best_fitness) or \
                   (not self.is_maximizing and new_whale.fitness_value < self.best_fitness):
                    self.best_fitness = new_whale.fitness_value
                    self.best_solution = new_whale.position[:]
            
            # Update population
            self.population = new_population
        
        return [self.best_solution], self.best_fitness


class WOA_TSP(Algorithm):
    def __init__(self, agent_length, dataset_name):
        super().__init__(agent_length, dataset_name)
    
    def run(self):
        algorithm = WhaleOptimizationAlgorithm(
            self.AGENT_LENGTH,
            self.tsp_fitness_function,
            2,  # max iterations
            50,  # population size
            [i for i in range(1, self.AGENT_LENGTH + 1)],
            False  # minimization problem for TSP
        )
        return algorithm.execute()