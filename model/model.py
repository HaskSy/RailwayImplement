from __future__ import annotations

import igraph as ig
import numpy as np
from enum import Enum
from typing import (Final,
                    List,
                    Dict)
from const import constants

CONST = constants.Constants(CARRIAGE_WEIGHT=1,
                            MAX_LOCOMOTIVE_CARRYING=100,
                            TRAIN_LENGTH_MAX=50)


class CargoType(Enum):
    PEOPLE = 0
    LIQUID = 1
    CONTAINER = 2


class Cargo:

    def __init__(self, cargo_id: int, mass: int, destination: int, cargo_type: CargoType):
        self.cargo_id: int = cargo_id
        self.mass: int = mass
        self.destination: int = destination
        self.cargo_type: CargoType = cargo_type


class Carriage:

    __carriage_weight: Final = CONST.CARRIAGE_WEIGHT

    def __init__(self, carriage_id: int, cargo: Cargo = None):
        self.carriage_id: int = carriage_id
        self.cargo: Cargo = cargo
        self.current_mass = cargo.mass + self.__carriage_weight


class Locomotive:

    max_locomotive_carrying: Final = CONST.MAX_LOCOMOTIVE_CARRYING

    def __init__(self, locomotive_id: int):
        self.locomotive_id: int = locomotive_id


class Train:

    train_length_max: Final = CONST.TRAIN_LENGTH_MAX

    def __init__(self, locomotive: Locomotive, destination: int, carriages: List[Carriage] = None):
        if carriages is None:
            carriages = []

        self.name: str = str(locomotive.locomotive_id) + "_" + str(destination)
        self.locomotive: Locomotive = locomotive
        self.destination: int = destination
        self.carriages: List[Carriage] = carriages
        self.departure: int

    def am_i_legal(self) -> bool:
        # Function which check if this train actually able to move (might be renamed)
        if len(self.carriages) > self.train_length_max:
            return False

        # sum(map(lambda x: x.carriage_weight + x.cargo.mass, self.carriages))
        sum_mass = 0
        for carriage in self.carriages:
            sum_mass += carriage.current_mass
        if self.locomotive.max_locomotive_carrying > sum_mass:
            return False
        return True


class Station:

    def __init__(self, station_id: int) -> None:
        self.station_id: int = station_id
        self.export_cargos: List[Cargo] = []
        self.import_cargos: List[Cargo] = []
        self.carriages: List[Carriage] = []
        self.locomotives: List[Locomotive] = []
        self.trains_in: List[Train] = []
        self.trains_out: List[Train] = []
        self.destination_dict: Dict[int, int] = dict()

    def add_export_cargo(self, cargo: Cargo) -> None:
        self.export_cargos.append(cargo)

    def delete_export_cargo(self, cargo_id: int) -> Cargo:
        index = 0
        for cargo in self.export_cargos:
            if cargo.cargo_id == cargo_id:
                return self.export_cargos.pop(index)
            index += 1
        raise ValueError("element is not in list")

    def add_import_cargo(self, cargo: Cargo) -> None:
        self.import_cargos.append(cargo)

    def delete_import_cargo(self, cargo_id: int) -> Cargo:
        index = 0
        for cargo in self.import_cargos:
            if cargo.cargo_id == cargo_id:
                return self.import_cargos.pop(index)
            index += 1
        raise ValueError("element is not in list")

    def add_carriage(self, carriage: Carriage) -> None:
        self.carriages.append(carriage)

    def delete_carriage(self, carriage_id: int) -> Carriage:
        index = 0
        for carriage in self.carriages:
            if carriage.carriage_id == carriage_id:
                return self.carriages.pop(index)
            index += 1
        raise ValueError("element is not in list")

    def add_locomotive(self, locomotive: Locomotive) -> None:
        self.locomotives.append(locomotive)

    def delete_locomotive(self, locomotive_id: int) -> Locomotive:
        index = 0
        for locomotive in self.locomotives:
            if locomotive.locomotive_id == locomotive_id:
                return self.locomotives.pop(index)
            index += 1
        raise ValueError("element is not in list")

    def add_train_in(self, train_in: Train) -> None:
        self.trains_in.append(train_in)

    def delete_train_in(self, train_name: str) -> Train:
        index = 0
        for train in self.trains_in:
            if train.name == train_name:
                return self.trains_in.pop(index)
            index += 1
        raise ValueError("element is not in list")

    def add_train_out(self, train_out: Train) -> None:
        self.trains_out.append(train_out)

    def delete_train_out(self, train_name: str) -> Train:
        index = 0
        for train in self.trains_out:
            if train.name == train_name:
                return self.trains_out.pop(index)
            index += 1
        raise ValueError("element is not in list")

    def decompose_trains(self) -> None:
        for train in self.trains_in:
            locomotive = train.locomotive
            carriages = train.carriages
            self.add_locomotive(locomotive)
            for i in range(len(carriages)):
                if carriages[i].cargo.destination == self.station_id:
                    self.add_import_cargo(carriages[i].cargo)
                    carriages[i].cargo = None
                self.add_carriage(carriages[i])

    def compose_trains(self) -> None:
        pass

    def log(self) -> None:
        pass

    # Maybe there is to be some additional functions if you need


class Graph:
    """
        Как создавать объект графа?
            graph = Graph(4, edges=[[0, 1], [2, 3], [2, 1], [0, 3], [0, 2]],
                          cities=["Moscow", "New York", "Tokyo", "Rome"], directed=True)

        Задать веса на рёбрах?
            graph.set_distances_between_cities([100, 2000, 3, 5, 1])

        Список соседних вершин?
            graph.get_neighboring_vertices(vertex)

        Нарисовать граф?
            graph.draw_graph()

        Добавить:
        1. Всякие штучки при рисовании графа
        2. Документация
    """

    def __init__(self, n: int, edges: list, cities=None, adjacency_matrix=None, directed=False, *args, **kwargs):
        """
        Args:
            n: count of vertices
            edges: list of edges
            cities: list of strings with cities names
            adjacency_matrix: adjacency_matrix (list of lists)
            directed: is directed? default = FALSE
            *args: smt
            **kwargs: smt
        """
        self.graph = ig.Graph(n=n, edges=edges, directed=directed)
        self.graph_with_matrix = ig.Graph.Adjacency(adjacency_matrix) if adjacency_matrix is not None else [[]]
        if cities is not None:
            self.graph.vs["name"] = cities
        self.vertices = self.graph.vs
        self.edges = self.graph.get_edgelist()

    def __str__(self):
        return self.graph.__str__()

    def add_vertices(self, vertices):
        """
        This function can add new vertex
        Args:
            vertices:
        Returns: None
        """
        self.graph.add_vertices(vertices)

    def del_vertices(self, vertices):
        """
        This function can delete vertex
        Args:
            vertices: This are the vertices that we'll delete
        Returns: None
        """
        self.graph.delete_vertices(vertices)

    def join_graphs(self, other):
        """
        Args:
            other: graph that we'll join
        Returns: None
        """
        self.graph.union(other)

    def set_distances_between_cities(self, distances):
        """
        Args:
            distances: list of distances between cities
        Returns: None
        """
        if len(distances) != self.graph.ecount():
            print("You haven't indicated all distances")
        else:
            self.graph.es["distance"] = distances

    def get_neighboring_vertices(self, vertex) -> list:
        """
        Args:
            vertex: the vertex at witch we'll see the neighbors
        Returns: list of neighbors
        """
        return self.graph.neighbors(vertex)

    def get_graph(self) -> Graph:
        """
        Returns: igraph.Graph object
        """
        return self.graph

    def get_vertices_list(self):
        """
        Returns: list of vertices
        """
        return self.vertices

    def get_edges_list(self):
        """
        Returns: list of edges
        """
        return self.edges

    def get_vertices_count(self):
        """
        Returns: vertices count
        """
        return self.graph.vcount()

    def get_edges_count(self):
        """
        Returns: edges count
        """
        return self.graph.ecount()

    def draw_graph(self, layout="kk", vertex_color=None):
        """
        Args:
            layout: ...
            vertex_color: ...
        Returns: None. Nevertheless this function draws graph!
        """
        layout = self.graph.layout(layout)
        visual_style = {"layout": layout, "vertex_label": self.graph.vs["name"], "vertex_size": 20,
                        "edge_label": self.graph.es["distance"]}
        ig.plot(self.graph, **visual_style)

    def floyd_warshall(self):
        adj_graph = self.graph.get_adjacency()
        print(adj_graph)
        path_matrix = np.zeros(adj_graph.shape)
        n = adj_graph.shape[0]
        for i in range(0, n):
            for j in range(0, n):
                path_matrix[i, j] = i
                if i != j and adj_graph[i, j] == 0:
                    path_matrix[i, j] = -30000
                    adj_graph[i, j] = 30000  # set zeros to any large number which is bigger then the longest way

        for k in range(0, n):
            for i in range(0, n):
                for j in range(0, n):
                    if adj_graph[i, j] > adj_graph[i, k] + adj_graph[k, j]:
                        adj_graph[i, j] = adj_graph[i, k] + adj_graph[k, j]
                        path_matrix[i, j] = path_matrix[k, j]

        return path_matrix

    def restore_path_fw(self, path_matrix, i, j, a=None):
        if a is None:
            a = []
        i, j = int(i), int(j)
        if i == j:
            a.append(i)
        elif path_matrix[i, j] == -30000:
            a.append(i)
            a.append("-")
            a.append(j)
        else:
            a = self.restore_path_fw(path_matrix, i, path_matrix[i, j], a)
            a.append(j)
        return a


class World:

    # II priority task, cannot be done without graph implementation

    def __init__(self, init_graph: Graph = None):
        self.date = 0
        self.stations = {}
        self.graph: Graph = init_graph
        self.station_index = 0

    def __fill_stations_dict(self):
        for vertex in self.graph.get_vertices_list().indices:
            self.stations[vertex] = Station(vertex)

    def __set_station_dest_dict(self, station: Station, p_matrix) -> None:
        for vertex in self.graph.get_vertices_list().indices:
            path = self.graph.restore_path_fw(p_matrix, station.station_id, vertex)
            if "-" in path:
                station.destination_dict[vertex] = -1
            else:
                station.destination_dict[vertex] = path[1]

    def fill_dest_dicts(self) -> None:
        path_matrix = self.graph.floyd_warshall()
        for station in self.stations:
            self.__set_station_dest_dict(station, path_matrix)

    def tick(self) -> None:
        for station in self.stations:
            for train in station.trains_out:
                self.stations[train.destination].add_train_in(station.delete_train_out(train.name))
