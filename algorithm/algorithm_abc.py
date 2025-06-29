from algorithm.algorithm import Algorithm
import random

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

class Bee:
    def __init__(self, agent_length, fitness_function, list_poi, is_maximizing=True):
        self.agent_length = agent_length
        self.fitness_function = fitness_function
        self.is_maximizing = is_maximizing
        self.fitness_value = None
        self.vector = None
        self.trial = 0  # Counter for abandoned solutions
        
        self.generate_random_vector(list_poi)
        self.calculate_fitness()
        self.swap_sequence = []  # Store the sequence of swaps

    def generate_random_vector(self, list_poi):
        # Create a copy of the list to avoid modifying the original
        shuffled_list = list_poi.copy()
        # Shuffle the list in place
        random.shuffle(shuffled_list)
        self.vector = shuffled_list
        self.calculate_fitness()

    def calculate_fitness(self):
        self.fitness_value = self.fitness_function(self.vector)

    def set_vector(self, vector):
        self.vector = vector
        self.calculate_fitness()

    def employed_bee_phase(self, list_poi):
        """
        Employed bee phase using swap sequences
        """
        if len(self.vector) <= 1:
            return
            
        # Generate swap sequence
        sequence_length = random.randint(1, 3)  # Can try different lengths
        swap_sequence = []
        for _ in range(sequence_length):
            i, j = random.sample(range(len(self.vector)), 2)
            swap_sequence.append((i, j))
            
        # Apply swap sequence to generate new solution
        new_vector = apply_swap_sequence(self.vector, swap_sequence)
        new_fitness = self.fitness_function(new_vector)
        
        # Update if better solution found
        if (self.is_maximizing and new_fitness > self.fitness_value) or \
           (not self.is_maximizing and new_fitness < self.fitness_value):
            self.vector = new_vector
            self.fitness_value = new_fitness
            self.swap_sequence = swap_sequence  # Store successful swap sequence
            self.trial = 0
        else:
            self.trial += 1

    def onlooker_bee_phase(self, partner_bee):
        """
        Onlooker bee phase using swap sequences
        """
        if len(self.vector) <= 1:
            return
            
        # Combine swap sequences from current bee and partner
        combined_sequence = self.swap_sequence + partner_bee.swap_sequence
        if not combined_sequence:  # If no previous successful swaps
            # Generate new swap sequence
            i, j = random.sample(range(len(self.vector)), 2)
            combined_sequence = [(i, j)]
            
        # Apply combined swap sequence
        new_vector = apply_swap_sequence(self.vector, combined_sequence)
        new_fitness = self.fitness_function(new_vector)
        
        # Update if better solution found
        if (self.is_maximizing and new_fitness > self.fitness_value) or \
           (not self.is_maximizing and new_fitness < self.fitness_value):
            self.vector = new_vector
            self.fitness_value = new_fitness
            self.swap_sequence = combined_sequence
            self.trial = 0
        else:
            self.trial += 1

    def scout_bee_phase(self, list_poi):
        """
        Scout bee phase: Generate new random solution when current solution is abandoned
        """
        self.generate_random_vector(list_poi)
        self.trial = 0

class ABC:
    def __init__(self, agent_length, fitness, max_iter, colony_size, list_poi, is_maximizing=True):
        self.agent_length = agent_length
        self.fitness_function = fitness
        self.max_iter = max_iter
        self.list_poi = list_poi
        self.is_maximizing = is_maximizing
        self.colony_size = colony_size
        self.limit = colony_size * agent_length  # Limit for abandoning solutions
        
        # Initialize populations
        self.employed_bees = []
        self.onlooker_bees = []
        self.best_solution = None
        self.best_fitness = float('-inf') if is_maximizing else float('inf')

    def initialize_population(self):
        # Create employed bees
        for _ in range(self.colony_size):
            bee = Bee(
                self.agent_length,
                self.fitness_function,
                self.list_poi,
                self.is_maximizing
            )
            self.employed_bees.append(bee)
            
            # Update best solution if necessary
            if self.best_solution is None or \
               (self.is_maximizing and bee.fitness_value > self.best_fitness) or \
               (not self.is_maximizing and bee.fitness_value < self.best_fitness):
                self.best_solution = bee.vector.copy()
                self.best_fitness = bee.fitness_value

        # Create onlooker bees (initially same as employed bees)
        self.onlooker_bees = [Bee(
            self.agent_length,
            self.fitness_function,
            self.list_poi,
            self.is_maximizing
        ) for _ in range(self.colony_size)]

    def calculate_probabilities(self):
        """
        Calculate probability for onlooker bees to choose food sources
        """
        total_fitness = sum(bee.fitness_value for bee in self.employed_bees)
        return [bee.fitness_value/total_fitness for bee in self.employed_bees]

    def update_best_solution(self, bee):
        if (self.is_maximizing and bee.fitness_value > self.best_fitness) or \
           (not self.is_maximizing and bee.fitness_value < self.best_fitness):
            self.best_solution = bee.vector.copy()
            self.best_fitness = bee.fitness_value

    def execute(self):
        self.initialize_population()
        
        for iteration in range(self.max_iter):
            # Employed Bee Phase
            for bee in self.employed_bees:
                bee.employed_bee_phase(self.list_poi)
                self.update_best_solution(bee)
            
            # Onlooker Bee Phase
            probabilities = self.calculate_probabilities()
            for bee, prob in zip(self.onlooker_bees, probabilities):
                if random.random() < prob:
                    # Choose a random employed bee as partner
                    partner = random.choice(self.employed_bees)
                    bee.onlooker_bee_phase(partner)
                    self.update_best_solution(bee)
            
            # Scout Bee Phase
            for bee in self.employed_bees + self.onlooker_bees:
                if bee.trial >= self.limit:
                    bee.scout_bee_phase(self.list_poi)
                    self.update_best_solution(bee)
        
        return [self.best_solution], self.best_fitness

class ABC_TSP(Algorithm):
    def __init__(self, agent_length, dataset_name):
        super().__init__(agent_length, dataset_name)

    def run(self):
        algorithm = ABC(
            self.AGENT_LENGTH,
            self.tsp_fitness_function,
            100,  # max iterations
            10,   # colony size
            [i for i in range(1, self.AGENT_LENGTH + 1)],
            False  # is_maximizing
        )
        return algorithm.execute()
