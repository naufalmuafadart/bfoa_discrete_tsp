import unittest
from algorithm.algorithm import Algorithm


class TestTravelTime(unittest.TestCase):
    def setUp(self):
        # Assuming the function is part of a class, create an instance
        # Replace YourClassName with the actual class name
        self.travel_calculator = Algorithm()

    def test_get_travel_time_valid_points(self):
        # Test with valid points
        self.assertEqual(self.travel_calculator.get_travel_time(1, 2), 451)
        self.assertEqual(self.travel_calculator.get_travel_time(13, 13), 0)

    def test_get_travel_time_invalid_points(self):
        # Test with non-existent points
        with self.assertRaises(IndexError):
            self.travel_calculator.get_travel_time(678, 5)

if __name__ == '__main__':
    unittest.main()
