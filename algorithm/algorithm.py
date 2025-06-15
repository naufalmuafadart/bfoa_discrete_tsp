import numpy as np

class Algorithm:
    def __init__(self, agent_length, dataset_name):
        self.AGENT_LENGTH = agent_length
        self.DATASET_NAME = dataset_name
        self.distance_matrix = None
        # get dataset
        self.prepare_dataset()

    def prepare_dataset(self):
        self.distance_matrix = np.load(f'./dataset/distance_matrix/{self.DATASET_NAME}.npy')
        self.AGENT_LENGTH = self.distance_matrix.shape[0]

    def get_travel_time(self, a, b):
        a = a -1
        b = b -1
        rows, columns = self.distance_matrix.shape
        if a > rows - 1 or b > columns - 1 or a < 0 or b < 0:
            raise IndexError('a and b must in range of distance matrix')
        return self.distance_matrix[a][b]

    def tsp_fitness_function(self, solution):
        if solution is None or solution == [] or not isinstance(solution, list):
            raise ValueError('Solution not valid')
        if len(set(solution)) != len(solution):
            raise ValueError('Solution has duplicate value')
        if len(solution) != self.AGENT_LENGTH:
            raise ValueError('Solution length is not correct')
        duration = 0
        for index, poi in enumerate(solution):
            if index == len(solution) - 1:
                duration += self.get_travel_time(solution[index], solution[0])
            else:
                duration += self.get_travel_time(solution[index], solution[index+1])
        return duration
