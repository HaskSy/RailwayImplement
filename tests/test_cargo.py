import unittest

from model.model import *


class MyTestCase(unittest.TestCase):

    def test_cargo_init_raises(self):
        with self.assertRaises(ValueError) as context:
            Cargo(-1, 1, 1, CargoType.CONTAINER)
        self.assertEqual('cargo_id cannot be negative', str(context.exception))

        with self.assertRaises(ValueError) as context:
            Cargo(1, -1, 1, CargoType.CONTAINER)
        self.assertEqual('mass cannot be negative', str(context.exception))

        with self.assertRaises(ValueError) as context:
            Cargo(1, 1, -1, CargoType.CONTAINER)
        self.assertEqual('ID cannot be negative', str(context.exception))


if __name__ == '__main__':
    unittest.main()