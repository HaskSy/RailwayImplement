import unittest

from model.model import *


class MyTestCase(unittest.TestCase):

    def test_world_init_raises(self):
        with self.assertRaises(AssertionError) as context:
            World("Hello", 1)
        self.assertEqual(f'init_graph object is not Graph type, current type: {type(1)}', str(context.exception))


if __name__ == '__main__':
    unittest.main()
