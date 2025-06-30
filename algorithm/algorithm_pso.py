from algorithm.algorithm import Algorithm
import random
import numpy as np


class Particle:
    def __init__(self, chromosome_length, fitness_function, list_poi, is_maximizing=True):
        self.chromosome_length = chromosome_length
        self.fitness_function = fitness_function
        self.is_maximizing = is_maximizing
        
        # Current position (permutation)
        self.position = None
        # Best position found by this particle
        self.pbest_position = None
        # Current and best fitness values
        self.fitness_value = float('-inf') if is_maximizing else float('inf')
        self.pbest_value = float('-inf') if is_maximizing else float('inf')
        # Velocity (list of swap operations)
        self.velocity = []
        
        self.generate_random_position(list_poi)
        
    def generate_random_position(self, list_poi):
        # Create a random permutation
        self.position = list_poi.copy()
        random.shuffle(self.position)
        self.calculate_fitness()
        
        # Initialize pbest
        self.pbest_position = self.position[:]
        self.pbest_value = self.fitness_value
        
    def calculate_fitness(self):
        self.fitness_value = self.fitness_function(self.position)
        
    def update_pbest(self):
        if (self.is_maximizing and self.fitness_value > self.pbest_value) or \
           (not self.is_maximizing and self.fitness_value < self.pbest_value):
            self.pbest_value = self.fitness_value
            self.pbest_position = self.position[:]


class ParticleSwarmOptimization:
    def __init__(self, chromosome_length, fitness, max_iter, swarm_size, list_poi, is_maximizing=True):
        self.chromosome_length = chromosome_length
        self.fitness_function = fitness
        self.max_iter = max_iter
        self.swarm_size = swarm_size
        self.list_poi = list_poi
        self.is_maximizing = is_maximizing
        
        # PSO parameters
        self.w = 0.9  # inertia weight
        self.c1 = 2.0  # cognitive weight
        self.c2 = 2.0  # social weight
        
        # Swarm and tracking
        self.swarm = []
        self.gbest_position = None
        self.gbest_value = float('-inf') if is_maximizing else float('inf')
        
        self.initialize_swarm()
        
    def initialize_swarm(self):
        """Initialize random swarm"""
        for _ in range(self.swarm_size):
            particle = Particle(
                self.chromosome_length,
                self.fitness_function,
                self.list_poi,
                self.is_maximizing
            )
            self.swarm.append(particle)
            
            # Update global best if necessary
            if (self.is_maximizing and particle.fitness_value > self.gbest_value) or \
               (not self.is_maximizing and particle.fitness_value < self.gbest_value):
                self.gbest_value = particle.fitness_value
                self.gbest_position = particle.position[:]
    
    def apply_velocity(self, position, velocity):
        """Apply swap operations (velocity) to position"""
        new_position = position[:]
        for i, j in velocity:
            new_position[i], new_position[j] = new_position[j], new_position[i]
        return new_position
    
    def calculate_velocity(self, particle):
        """Calculate new velocity based on PSO formula"""
        new_velocity = []
        
        # Inertia component (keep some previous velocity)
        if random.random() < self.w and particle.velocity:
            num_ops = random.randint(1, len(particle.velocity))
            new_velocity.extend(random.sample(particle.velocity, num_ops))
        
        # Cognitive component (move towards personal best)
        if random.random() < self.c1:
            # Generate swaps to move towards pbest
            current = particle.position[:]
            target = particle.pbest_position[:]
            for i in range(len(current)):
                if current[i] != target[i]:
                    j = current.index(target[i])
                    new_velocity.append((i, j))
                    current[i], current[j] = current[j], current[i]
        
        # Social component (move towards global best)
        if random.random() < self.c2:
            # Generate swaps to move towards gbest
            current = particle.position[:]
            target = self.gbest_position[:]
            for i in range(len(current)):
                if current[i] != target[i]:
                    j = current.index(target[i])
                    new_velocity.append((i, j))
                    current[i], current[j] = current[j], current[i]
        
        return new_velocity
    
    def execute(self):
        """Execute the PSO algorithm"""
        for iteration in range(self.max_iter):
            for particle in self.swarm:
                # Calculate new velocity
                particle.velocity = self.calculate_velocity(particle)
                
                # Update position
                new_position = self.apply_velocity(particle.position, particle.velocity)
                particle.position = new_position
                particle.calculate_fitness()
                
                # Update personal best
                particle.update_pbest()
                
                # Update global best
                if (self.is_maximizing and particle.fitness_value > self.gbest_value) or \
                   (not self.is_maximizing and particle.fitness_value < self.gbest_value):
                    self.gbest_value = particle.fitness_value
                    self.gbest_position = particle.position[:]
        
        return [self.gbest_position], self.gbest_value


class PSO_TSP(Algorithm):
    def __init__(self, agent_length, dataset_name):
        super().__init__(agent_length, dataset_name)
    
    def run(self):
        algorithm = ParticleSwarmOptimization(
            self.AGENT_LENGTH,
            self.tsp_fitness_function,
            100,  # max iterations
            50,  # swarm size
            [i for i in range(1, self.AGENT_LENGTH + 1)],
            False  # minimization problem for TSP
        )
        return algorithm.execute()
