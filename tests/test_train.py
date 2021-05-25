import unittest

from model.model import *


class MyTestCase(unittest.TestCase):

    def test_train_init_raises(self):
        with self.assertRaises(AssertionError) as context:
            Train(1, 1)
        self.assertEqual(f'locomotive object is not Locomotive type, current type: {type(1)}', str(context.exception))

        with self.assertRaises(ValueError) as context:
            Train(Locomotive(1), -1)
        self.assertEqual('destination cannot be negative', str(context.exception))
        self.assertEqual(Train(Locomotive(1), 1).name, "1_1")

    def test_am_i_legal(self):
        pass

if __name__ == '__main__':
    unittest.main()
