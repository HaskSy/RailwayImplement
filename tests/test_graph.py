import unittest

from model.model import *
import numpy as np


class MyTestCase(unittest.TestCase):

    def test_graph_init_raises(self):
        with self.assertRaises(ValueError) as context:
            Graph(n=-1, edges=[])
        self.assertEqual('Number of vertices cannot be negative', str(context.exception))

        with self.assertRaises(TypeError) as context:
            Graph(n=0, edges=[])
        self.assertEqual('Give adjacency_matrix or n + edges pair for graph initialisation', str(context.exception))

        with self.assertRaises(TypeError) as context:
            Graph(n=4)
        self.assertEqual('Give adjacency_matrix or n + edges pair for graph initialisation', str(context.exception))

    def test_graph_init(self):
        a = Graph(n=5, edges=[(1, 3), (3, 4), (4, 2), (2, 1), (2, 3)])
        adj = [[0, 0, 0, 0, 0],
               [0, 0, 1, 1, 0],
               [0, 1, 0, 1, 1],
               [0, 1, 1, 0, 1],
               [0, 0, 1, 1, 0]]

        self.assertEqual(a.adjacency_matrix._get_data(), adj)

        a = Graph(n=5, edges=[(1, 3), (3, 4), (4, 2), (2, 4), (2, 1), (2, 3)], directed=True)

        adj = [[0, 0, 0, 0, 0],
               [0, 0, 0, 1, 0],
               [0, 1, 0, 1, 1],
               [0, 0, 0, 0, 1],
               [0, 0, 1, 0, 0]]

        self.assertEqual(a.adjacency_matrix._get_data(), adj)
        self.assertEqual(a.get_edges_list(), [(1, 3), (3, 4), (4, 2), (2, 4), (2, 1), (2, 3)])
        self.assertEqual(a.get_vertices_list().indices, [0, 1, 2, 3, 4])

        adj = [[0, 1, 1, 1],
               [1, 0, 1, 0],
               [1, 1, 0, 0],
               [1, 0, 0, 0]]

        a = Graph(adjacency_matrix=adj)

        self.assertEqual(a.adjacency_matrix._get_data(), adj)
        self.assertEqual(a.get_edges_list(), [(0, 1), (0, 2), (0, 3), (1, 0), (1, 2), (2, 0), (2, 1), (3, 0)])
        self.assertEqual(a.get_vertices_list().indices, [0, 1, 2, 3])

        adj = [[0, 1, 0, 0, 0],
               [0, 0, 1, 1, 0],
               [0, 0, 0, 0, 1],
               [1, 0, 1, 0, 1],
               [0, 0, 1, 0, 0]]

        a = Graph(adjacency_matrix=adj)

        self.assertEqual(a.adjacency_matrix._get_data(), adj)
        self.assertEqual(a.get_edges_list(), [(0, 1), (1, 2), (1, 3), (2, 4), (3, 0), (3, 2), (3, 4), (4, 2)])
        self.assertEqual(a.get_vertices_list().indices, [0, 1, 2, 3, 4])

    def test_vertices_naming(self):
        cities = ['Paris', 'San Marino', 'Beijing', 'Tokyo', 'Kyoto']

        adj = [[0, 1, 0, 1, 0],
               [0, 0, 1, 0, 0],
               [0, 1, 0, 0, 1],
               [1, 0, 1, 0, 0],
               [1, 0, 0, 1, 0]]

        a = Graph(adjacency_matrix=adj, cities=cities)

        self.assertEqual(a.adjacency_matrix._get_data(), adj)
        self.assertEqual(a.get_edges_list(), [(0, 1), (0, 3), (1, 2), (2, 1), (2, 4), (3, 0), (3, 2), (4, 0), (4, 3)])
        self.assertEqual(a.get_vertices_list()['name'], cities)

    def test_p_matrix_generation(self):
        adj = [[0, 1, 0, 1, 0],
               [0, 0, 1, 0, 0],
               [0, 1, 0, 0, 1],
               [1, 0, 1, 0, 0],
               [1, 0, 0, 1, 0]]

        a = Graph(adjacency_matrix=adj)

        p = [[0, 0, 1, 0, 2],
             [4, 1, 1, 4, 2],
             [4, 2, 2, 4, 2],
             [3, 0, 3, 3, 2],
             [4, 0, 3, 4, 4]]

        self.assertTrue(np.array_equal(a.floyd_warshall(), p))

        a = Graph(n=7, edges=[(0, 2), (0, 4), (1, 4), (2, 3), (2, 4), (3, 6), (4, 5), (5, 6)], directed=True)

        p = [[0, -np.inf, 0, 2, 0, 4, 3],
             [-np.inf, 1, -np.inf, -np.inf, 1, 4, 5],
             [-np.inf, -np.inf, 2, 2, 2, 4, 3],
             [-np.inf, -np.inf, -np.inf, 3, -np.inf, -np.inf, 3],
             [-np.inf, -np.inf, -np.inf, -np.inf, 4, 4, 5],
             [-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 5, 5],
             [-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 6]]

        self.assertTrue(np.array_equal(a.floyd_warshall(), p))

    def test_path_restoration(self):
        pass



if __name__ == '__main__':
    unittest.main()
