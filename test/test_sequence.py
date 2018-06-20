import unittest

from nose.tools import assert_equal

from task.algorithmic_task import sequence, is_prime


class TestSequence(unittest.TestCase):
    values = ([2, 3, 9, 2, 5, 1, 3, 7, 10],
              [2, 1, 3, 4, 3, 10, 6, 6, 1, 7, 10, 10, 10])

    expected_value_sequence = [2, 9, 2, 5, 7, 10]
    expected_value_is_prime = True

    def test_sequence_success(self):
        result = sequence(self.values[0], self.values[1])
        assert_equal(self.expected_value_sequence, result)

    def test_sequence_not_fail(self):
        result = sequence([], [])
        assert_equal([], result)

    def test_is_prime(self):
        result = is_prime(2)
        assert_equal(self.expected_value_is_prime, result)


if __name__ == '__main__':
    unittest.main()
