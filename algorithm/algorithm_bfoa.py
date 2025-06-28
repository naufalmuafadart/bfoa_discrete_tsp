from enum import Enum
from algorithm.algorithm import Algorithm
import random


class Agent:
    def __init__(self, agent_length, fitness_function, list_poi, is_squad_1=True, is_maximizing=True):
        self.agent_length = agent_length
        self.fitness_function = fitness_function
        self.is_squad_1 = is_squad_1
        self.is_maximizing = is_maximizing
        self.fitness_value = None
        self.vector = None

        self.generate_random_vector(list_poi)
        self.calculate_fitness()

    def generate_random_vector(self, list_poi):
        # Create a copy of the list to avoid modifying the original
        shuffled_list = list_poi.copy()
        # Shuffle the list in place
        random.shuffle(shuffled_list)
        self.vector = shuffled_list
        self.fitness_value = self.fitness_function(self.vector)

    def calculate_fitness(self):
        self.fitness_value = self.fitness_function(self.vector)

    def set_vector(self, vector):
        self.vector = vector
        self.fitness_value = self.fitness_function(self.vector)

    """
    Perform airplane movement using scramble mutation
    """
    def airplane_movement(self):
        n = len(self.vector)
        i, j = sorted(random.sample(range(n), 2))  # Select 2 random distinct positions
        subsequence = self.vector[i:j + 1]
        random.shuffle(subsequence)
        self.vector[i:j + 1] = subsequence
        self.calculate_fitness()

    """
    Perform builder movement using insert mutation
    """
    def builder_movement(self):
        n = len(self.vector)
        # Pick two random positions
        pos1, pos2 = random.sample(range(n), 2)
        city = self.vector.pop(pos1)
        self.vector.insert(pos2, city)
        self.calculate_fitness()

    """
    Perform Partially Mapped Crossover (PMX) between the current vector and enemy commander vector

    Args:
        enemy_commander (list): Enemy chromosome
    """
    def commander_movement(self, enemy_commander):
        # Choose two random crossover points
        point1 = random.randint(0, self.agent_length - 2)
        point2 = random.randint(point1 + 1, self.agent_length - 1)

        # Initialize offspring as copies of parents
        offspring1 = [None] * self.agent_length
        offspring2 = [None] * self.agent_length

        # Copy the mapping section
        for i in range(point1, point2 + 1):
            offspring1[i] = self.vector[i]
            offspring2[i] = enemy_commander[i]

        # Create mapping between elements in the mapping section
        mapping1 = {enemy_commander[i]: self.vector[i] for i in range(point1, point2 + 1)}

        # Fill in remaining positions for offspring1
        for i in range(self.agent_length):
            if i < point1 or i > point2:
                # Current element from enemy_commander
                current = enemy_commander[i]
                # While the element exists in mapping, keep mapping
                while current in mapping1:
                    current = mapping1[current]
                offspring1[i] = current
        self.calculate_fitness()

    """
    Perform flanking movement using cyclic moves mutation
    
    This method selects a segment of the tour and rotates it by a random amount
    """
    def cavalry_movement(self):
        n = len(self.vector)
        i, j = sorted(random.sample(range(n), 2))
        # Get the segment to rotate
        segment = self.vector[i:j + 1]
        # Choose a random rotation amount (between 1 and length-1 to ensure actual change)
        rotation = random.randint(1, len(segment) - 1)
        # Rotate the segment and place it back
        self.vector[i:j + 1] = segment[rotation:] + segment[:rotation]
        self.calculate_fitness()

    """
    Perform special force movement using targeted insertion

    Args:
        enemy_commander (list): Enemy chromosome
    """
    def special_force_movement(self):
        if len(self.vector) <= 2:
            return

            # 1. Randomly select a city to move
        n = len(self.vector)
        remove_idx = random.randint(0, n - 1)
        city = self.vector[remove_idx]

        # 2. Remove the city from current position
        current_tour = self.vector[:remove_idx] + self.vector[remove_idx + 1:]

        # 3. Find the best insertion position
        best_position = 0
        best_fitness = float('inf') if not self.is_maximizing else float('-inf')

        # Try each possible insertion position
        for i in range(len(current_tour) + 1):
            # Create a candidate tour
            candidate_tour = current_tour[:i] + [city] + current_tour[i:]
            # Calculate fitness
            candidate_fitness = self.fitness_function(candidate_tour)

            # Update best position if this position is better
            if (not self.is_maximizing and candidate_fitness < best_fitness) or \
                    (self.is_maximizing and candidate_fitness > best_fitness):
                best_fitness = candidate_fitness
                best_position = i

        # 4. Insert the city in the best position found
        self.vector = current_tour[:best_position] + [city] + current_tour[best_position:]
        self.calculate_fitness()


class SquadMode(Enum):
    ATTACKING = 1
    DEFENDING = 2

class Squad:
    def __init__(self, mode):
        self.mode = mode
        self.air_forces = []
        self.commander = None
        self.left_cavalry = None
        self.right_cavalry = None
        self.special_force = None
        self.builder = None

    def sort_air_forces(self, is_maximizing):
        self.air_forces.sort(key=lambda agent: agent.fitness_value, reverse=not is_maximizing)

    def assign_squad(self, is_maximizing):
        if self.commander  is None: # if squad is still empty
            self.sort_air_forces(is_maximizing)
            self.commander = self.air_forces[0]
            self.left_cavalry = self.air_forces[1]
            self.right_cavalry = self.air_forces[2]
            self.special_force = self.air_forces[3]
            self.builder = self.air_forces[4]
            return
        # if squad is not empty
        self.air_forces[0] = self.commander
        self.air_forces[1] = self.left_cavalry
        self.air_forces[2] = self.right_cavalry
        self.air_forces[3] = self.special_force
        self.air_forces[4] = self.builder
        self.sort_air_forces(is_maximizing)
        self.commander = self.air_forces[0]
        self.left_cavalry = self.air_forces[1]
        self.right_cavalry = self.air_forces[2]
        self.special_force = self.air_forces[3]
        self.builder = self.air_forces[4]

class BFOA:
    def __init__(self, agent_length, fitness, max_iter, N, list_poi, is_maximizing=True):
        self.agent_length = agent_length
        self.fitness_function = fitness
        self.max_iter = max_iter
        self.list_poi = list_poi
        self.is_maximizing = is_maximizing
        self.N = N
        self.n_plane = 8
        self.population = []
        self.phase_1_max_iteration = 2
        self.phase_2_max_iteration = 2
        self.squad1 = Squad(SquadMode.ATTACKING)
        self.squad2 = Squad(SquadMode.DEFENDING)
        self.best_troops = None
        self.best_troops_fitness = 0

    # Function to initialize population (airplane) in phase 1
    def phase_1_initialization(self):
        for i in range(self.n_plane):
            self.squad1.air_forces.append(Agent(
                self.agent_length,
                self.fitness_function,
                self.list_poi,
                True,
                self.is_maximizing))
            self.squad2.air_forces.append(Agent(
                self.agent_length,
                self.fitness_function,
                self.list_poi,
                False,
                self.is_maximizing))

    def is_squad_1_better(self):
        return (self.squad1.commander.fitness_value > self.squad2.commander.fitness_value and self.is_maximizing) or\
            (self.squad1.commander.fitness_value < self.squad2.commander.fitness_value and not self.is_maximizing)

    def is_fitness_value_a_better_than_b(self, a, b):
        if self.is_maximizing:
            return a > b
        return a <= b

    def update_best_troops(self):
        if self.best_troops is None:
            self.best_troops = self.squad1.commander.vector
            self.best_troops_fitness = self.squad1.commander.fitness_value
        if self.is_squad_1_better() and self.is_fitness_value_a_better_than_b(self.squad1.commander.fitness_value,
                                                                              self.best_troops_fitness):
            self.best_troops = self.squad1.commander.vector
            self.best_troops_fitness = self.squad1.commander.fitness_value
        elif not self.is_squad_1_better() and self.is_fitness_value_a_better_than_b(
                self.squad2.commander.fitness_value, self.best_troops_fitness):
            self.best_troops = self.squad2.commander.vector
            self.best_troops_fitness = self.squad2.commander.fitness_value

    def execute(self):
        # print('start phase 1')
        self.phase_1_initialization()
        for i in range(self.phase_1_max_iteration): # Airplane movement
            for j in range(self.n_plane):
                self.squad1.air_forces[j].airplane_movement()
                self.squad2.air_forces[j].airplane_movement()

        # assign air force to squad
        self.squad1.assign_squad(self.is_maximizing)
        self.squad2.assign_squad(self.is_maximizing)

        # determine attacking and defending squad
        if self.is_squad_1_better():
            self.squad1.mode = SquadMode.DEFENDING
            self.squad2.mode = SquadMode.ATTACKING
        else:
            self.squad1.mode = SquadMode.ATTACKING
            self.squad2.mode = SquadMode.DEFENDING
        # print('end phase 1')

        # print('start phase 2')
        for i in range(self.phase_2_max_iteration): # builder movement
            if self.is_squad_1_better():
                self.squad1.commander.builder_movement()
                self.squad1.left_cavalry.builder_movement()
                self.squad1.right_cavalry.builder_movement()
                self.squad1.builder.builder_movement()

                self.squad2.commander.commander_movement(self.squad1.commander.vector)
                self.squad2.left_cavalry.cavalry_movement()
                self.squad2.right_cavalry.cavalry_movement()
                self.squad2.builder.builder_movement()
            else:
                self.squad1.commander.commander_movement(self.squad2.commander.vector)
                self.squad1.left_cavalry.cavalry_movement()
                self.squad1.right_cavalry.cavalry_movement()
                self.squad1.builder.builder_movement()

                self.squad2.commander.builder_movement()
                self.squad2.left_cavalry.builder_movement()
                self.squad2.right_cavalry.builder_movement()
                self.squad2.builder.builder_movement()

            # Move special force
            self.squad1.special_force.special_force_movement()
            self.squad2.special_force.special_force_movement()

            # assign squad based on fitness
            self.squad1.assign_squad(self.is_maximizing)
            self.squad2.assign_squad(self.is_maximizing)

            # update best troops
            self.update_best_troops()

            # determine attacking and defending squad
            if self.is_squad_1_better():
                self.squad1.mode = SquadMode.DEFENDING
                self.squad2.mode = SquadMode.ATTACKING
            else:
                self.squad1.mode = SquadMode.ATTACKING
                self.squad2.mode = SquadMode.DEFENDING
        # print('phase 2 done')
        # print('best troops: ', self.best_troops)
        # print('best troops fitness: ', self.best_troops_fitness)
        return [self.best_troops], self.best_troops_fitness

class BfOA_TSP(Algorithm):
    def __init__(self, agent_length, dataset_name):
        super().__init__(agent_length, dataset_name)

    def run(self):
        algorithm = BFOA(
            self.AGENT_LENGTH,
            self.tsp_fitness_function,
            100,
            10,
            [i for i in range(1, self.AGENT_LENGTH + 1)],
            False
        )
        return algorithm.execute()