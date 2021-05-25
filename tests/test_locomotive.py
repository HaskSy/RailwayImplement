import unittest

from model.model import *


class MyTestCase(unittest.TestCase):

    def test_locomotive_init_raises(self):
        with self.assertRaises(ValueError) as context:
            Locomotive(-1)
        self.assertEqual('locomotive_id cannot be negative', str(context.exception))


if __name__ == '__main__':
    unittest.main()
