import unittest

from model.model import *


class MyTestCase(unittest.TestCase):

    def test_carriage_init_raises(self):
        with self.assertRaises(ValueError) as context:
            Carriage(-1, Cargo(11, 1, 1, CargoType.CONTAINER))
        self.assertEqual('carriage_id cannot be negative', str(context.exception))

        with self.assertRaises(AssertionError) as context:
            Carriage(1, 1)
        self.assertEqual(f'cargo object is not Cargo type, current type: {type(1)}', str(context.exception))


if __name__ == '__main__':
    unittest.main()
