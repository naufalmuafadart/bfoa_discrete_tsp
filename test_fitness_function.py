import unittest
from algorithm.algorithm import Algorithm

class TestFitnessFunction(unittest.TestCase):
    def setUp(self):
        self.algorithm = Algorithm(
            [1, 2, 3],
            123,
            1,
            1,
            1,
            1,
            1
        )

    def test_it_should_throw_an_exception_when_the_list_of_points_is_empty_or_none(self):
        with self.assertRaises(ValueError):
            self.algorithm.tsp_fitness_function([])

        with self.assertRaises(ValueError):
            self.algorithm.tsp_fitness_function(None)

    def test_it_should_throw_an_exception_when_the_list_of_points_is_not_a_list(self):
        with self.assertRaises(ValueError):
            self.algorithm.tsp_fitness_function(True)

        with self.assertRaises(ValueError):
            self.algorithm.tsp_fitness_function(0)

        with self.assertRaises(ValueError):
            self.algorithm.tsp_fitness_function('abc')

    def test_it_should_throw_an_exception_when_the_list_of_points_contains_duplicate_points(self):
        with self.assertRaises(ValueError):
            self.algorithm.tsp_fitness_function([1, 2, 1])

    def test_it_should_throw_an_exception_when_the_length_of_solution_is_not_same_with_agent_length(self):
        with self.assertRaises(ValueError):
            self.algorithm.tsp_fitness_function([1, 2, 3, 4])

    def it_should_throw_an_exception_when_the_list_of_points_contains_points_outside_of_the_range(self):
        with self.assertRaises(ValueError):
            self.algorithm.tsp_fitness_function([1, 2, 1000])

    def test_it_should_return_the_correct_fitness_value(self):
        self.assertEqual(self.algorithm.tsp_fitness_function([1, 2, 3]), 1837)

        algorithm = Algorithm(
            [1],
            123,
            1,
            1,
            1,
            1,
            1
        )
        self.assertEqual(algorithm.tsp_fitness_function([1]), 792)

if __name__ == '__main__':
    unittest.main()
