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
        adj1 = [[0, 1, 0, 1, 1, 1, 0],
                [1, 0, 1, 1, 0, 0, 0],
                [0, 1, 0, 1, 0, 0, 0],
                [1, 1, 1, 0, 1, 0, 0],
                [1, 0, 0, 1, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 1, 0]]

        adj2 = [[0, 0, 0, 0, 1, 0, 1],
                [1, 0, 0, 0, 0, 0, 1],
                [0, 1, 0, 0, 0, 0, 1],
                [0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 1, 1],
                [0, 0, 0, 1, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0]]

        adj3 = [[0, 1, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 0]]

        a1 = Graph(adjacency_matrix=adj1)
        a2 = Graph(adjacency_matrix=adj2)
        a3 = Graph(adjacency_matrix=adj3)

        p1 = a1.floyd_warshall()
        p2 = a2.floyd_warshall()
        p3 = a3.floyd_warshall()

        paths1 = [[0], [0, 1], [0, 1, 2], [0, 3], [0, 4], [0, 5], [0, 5, 6], [1, 0], [1], [1, 2], [1, 3], [1, 0, 4], [1, 0, 5],
         [1, 0, 5, 6], [2, 1, 0], [2, 1], [2], [2, 3], [2, 3, 4], [2, 1, 0, 5], [2, 1, 0, 5, 6], [3, 0], [3, 1], [3, 2],
         [3], [3, 4], [3, 0, 5], [3, 0, 5, 6], [4, 0], [4, 0, 1], [4, 3, 2], [4, 3], [4], [4, 0, 5], [4, 0, 5, 6],
         [5, 0], [5, 0, 1], [5, 0, 1, 2], [5, 0, 3], [5, 0, 4], [5], [5, 6], [6, 5, 0], [6, 5, 0, 1], [6, 5, 0, 1, 2],
         [6, 5, 0, 3], [6, 5, 0, 4], [6, 5], [6]]

        for i in range(len(adj1)):
            for j in range(len(adj1)):
                self.assertEqual(paths1[i*len(adj1) + j], a1.restore_path_fw(p1, i, j))

        paths2 = [[0], [0, 4, 5, 3, 2, 1], [0, 4, 5, 3, 2], [0, 4, 5, 3], [0, 4], [0, 4, 5], [0, 6], [1, 0], [1],
                  [1, 0, 4, 5, 3, 2], [1, 0, 4, 5, 3], [1, 0, 4], [1, 0, 4, 5], [1, 6], [2, 1, 0], [2, 1], [2],
                  [2, 1, 0, 4, 5, 3], [2, 1, 0, 4], [2, 1, 0, 4, 5], [2, 6], [3, 2, 1, 0], [3, 2, 1], [3, 2], [3],
                  [3, 2, 1, 0, 4], [3, 2, 1, 0, 4, 5], [3, 6], [4, 5, 3, 2, 1, 0], [4, 5, 3, 2, 1], [4, 5, 3, 2],
                  [4, 5, 3], [4], [4, 5], [4, 6], [5, 3, 2, 1, 0], [5, 3, 2, 1], [5, 3, 2], [5, 3], [5, 3, 2, 1, 0, 4],
                  [5], [5, 6], [6, '-', 0], [6, '-', 1], [6, '-', 2], [6, '-', 3], [6, '-', 4], [6, '-', 5], [6]]

        for i in range(len(adj2)):
            for j in range(len(adj2)):
                self.assertEqual(paths2[i*len(adj2) + j], a2.restore_path_fw(p2, i, j))

        paths3 = [[0], [0, 1], [0, 2], [0, 1, 3], [0, 2, 4], [0, 1, 3, 5], [0, 2, 4, 6], [0, 1, 3, 5, 7],
                  [1, 3, 5, 7, 0], [1], [1, 3, 5, 7, 0, 2], [1, 3], [1, 3, 5, 7, 0, 2, 4], [1, 3, 5],
                  [1, 3, 5, 7, 0, 2, 4, 6], [1, 3, 5, 7], [2, 3, 5, 7, 0], [2, 3, 5, 7, 0, 1], [2], [2, 3], [2, 4],
                  [2, 3, 5], [2, 4, 6], [2, 3, 5, 7], [3, 5, 7, 0], [3, 5, 7, 0, 1], [3, 5, 7, 0, 2], [3],
                  [3, 5, 7, 0, 2, 4], [3, 5], [3, 5, 7, 0, 2, 4, 6], [3, 5, 7], [4, 5, 7, 0], [4, 5, 7, 0, 1],
                  [4, 5, 7, 0, 2], [4, 5, 7, 0, 1, 3], [4], [4, 5], [4, 6], [4, 5, 7], [5, 7, 0], [5, 7, 0, 1],
                  [5, 7, 0, 2], [5, 7, 0, 1, 3], [5, 7, 0, 2, 4], [5], [5, 7, 0, 2, 4, 6], [5, 7], [6, 7, 0],
                  [6, 7, 0, 1], [6, 7, 0, 2], [6, 7, 0, 1, 3], [6, 7, 0, 2, 4], [6, 7, 0, 1, 3, 5], [6], [6, 7], [7, 0],
                  [7, 0, 1], [7, 0, 2], [7, 0, 1, 3], [7, 0, 2, 4], [7, 0, 1, 3, 5], [7, 0, 2, 4, 6], [7]]

        for i in range(len(adj3)):
            for j in range(len(adj3)):
                self.assertEqual(paths3[i*len(adj3) + j], a3.restore_path_fw(p3, i, j))


if __name__ == '__main__':
    unittest.main()
