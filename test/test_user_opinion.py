import unittest

from nose.tools import assert_equal
from counting.session_counting import user_opinion, formatting_time


class TestUser_opinion(unittest.TestCase):
    values = (['2017-03-10 08:13:11', '2017-03-10 19:01:27',
               '2017-03-11 07:35:55', '2017-03-11 16:15:11', '2017-03-12 08:01:41', '2017-03-12 17:19:08'],
              ['2017-03-10 18:58:11', '2017-03-10 19:01:27', '2017-03-11 07:35:55',
               '2017-03-11 16:15:11', '2017-03-12 08:01:41', '2017-03-12 17:19:08'],
              ['2017-03-08 17:11:13', '2017-03-11 17:22:47', '2017-03-11 19:23:51', '2017-03-11 22:03:12',
               '2017-03-12 08:32:04', '2017-03-12 13:19:08', '2017-03-12 17:19:08']
              )

    expected_value_opinion = True, False
    expected_value_formatting_date = 0, 648.2666666666667

    def test_user_opinion_success(self):
        result = user_opinion(self.values[0])
        assert_equal(self.expected_value_opinion[0], result)

    def test_user_opinion_fail_not_enough_number_of_sessions(self):
        result = user_opinion(self.values[1])
        assert_equal(self.expected_value_opinion[1], result)

    def test_user_opinion_fail_not_enough_number_of_days(self):
        result = user_opinion(self.values[2])
        assert_equal(self.expected_value_opinion[1], result)

    def test_formatting_time(self):
        result = formatting_time(self.values[0][0], self.values[0][1])
        assert_equal(self.expected_value_formatting_date, result)


if __name__ == '__main__':
    unittest.main()
