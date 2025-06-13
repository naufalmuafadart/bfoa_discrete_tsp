import unittest
from algorithm.algorithm import Algorithm


class TestTravelTime(unittest.TestCase):
    def setUp(self):
        # Assuming the function is part of a class, create an instance
        # Replace YourClassName with the actual class name
        self.travel_calculator = Algorithm(
            [1, 2, 3],
            123,
            1,
            1,
            1,
            1,
            1
        )

    def test_get_travel_time_valid_points(self):
        # Test with valid points
        self.assertEqual(self.travel_calculator.get_travel_time(1, 2), 134)
        self.assertEqual(self.travel_calculator.get_travel_time(1, 3), 605)
        self.assertEqual(self.travel_calculator.get_travel_time(1, 4), 595)

    def test_get_travel_time_invalid_points(self):
        # Test with non-existent points
        with self.assertRaises(IndexError):
            self.travel_calculator.get_travel_time(678, 5)

if __name__ == '__main__':
    unittest.main()
