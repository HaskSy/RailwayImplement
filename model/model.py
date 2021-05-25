from __future__ import annotations

import time
import os
import igraph as ig
import numpy as np
from enum import Enum
from typing import (Final,
                    List,
                    Dict)
from const import constants

CONST = constants.Constants(CARRIAGE_WEIGHT=1,
                            MAX_LOCOMOTIVE_CARRYING=10000,
                            TRAIN_LENGTH_MAX=50)


class CargoType(Enum):
    PEOPLE = 0
    LIQUID = 1
    CONTAINER = 2


class Cargo:

    def __init__(self, cargo_id: int, mass: int, destination: int, cargo_type: CargoType):
        if cargo_id < 0:
            raise ValueError('cargo_id cannot be negative')
        if mass < 0:
            raise ValueError('mass cannot be negative')
        if destination < 0:
            raise ValueError('ID cannot be negative')

        self.cargo_id: int = cargo_id
        self.mass: int = mass
        self.destination: int = destination
        self.cargo_type: CargoType = cargo_type


class Carriage:
    __carriage_weight: Final = CONST.CARRIAGE_WEIGHT

    def __init__(self, carriage_id: int, cargo: Cargo):
      
        if carriage_id < 0:
            raise ValueError('carriage_id cannot be negative')
        assert type(cargo) == Cargo, \
            f'cargo object is not Cargo type, current type: {type(cargo)}'

        self.carriage_id: int = carriage_id
        self.cargo: Cargo = cargo
        self.current_mass = self.__carriage_weight + self.cargo.mass


class Locomotive:
    max_locomotive_carrying: Final = CONST.MAX_LOCOMOTIVE_CARRYING

    def __init__(self, locomotive_id: int):
        if locomotive_id < 0:
            raise ValueError('locomotive_id cannot be negative')
        self.locomotive_id: int = locomotive_id


class Train:
    train_length_max: Final = CONST.TRAIN_LENGTH_MAX

    def __init__(self, locomotive: Locomotive, destination: int, carriages: List[Carriage] = None):

        assert type(locomotive) == Locomotive, \
            f'locomotive object is not Locomotive type, current type: {type(locomotive)}'
        if destination < 0:
            raise ValueError('destination cannot be negative')

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
      
        if station_id < 0:
            raise ValueError("station_id cannot be negative")

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

    def compose_trains(self, destination: int) -> None:
        # собираем вагоны в словарь по направлениям
        # по направлениям: видимо смежная вершина нашей станции
        # подаем на вход словарю наш дестинатион и получаем смежную вершину
        # лучше эффективно - жадно
        # считаем возможные максимальные длины каждых направлений, а потом выбираем лучшие
        # заполняем все локомотивы пока не кончатся вагоны или сами локомотивы
        # не знаю, что делать с удалением, потом можно впихнуть если попросят
        carriages_dict = {}  # словарик для вагонов по направлению
        for carriage in self.carriages:
            dict_key = self.destination_dict[carriage.cargo.destination]
            if dict_key in carriages_dict.keys():
                carriages_dict[dict_key].append(carriage)
            else:
                carriages_dict[dict_key] = [carriage]
        # реализуем жадное составление поездов
        # отсортируем вагоны по возрастанию массы груза...
        for value in carriages_dict.values():
            value.sort(key=lambda x: x.cargo.mass)
            # value.sort(key=lambda x: x.cargo.mass, reverse=True)

        while self.locomotives != [] and (self.carriages != [] or carriages_dict != {}):
            locomotive = self.locomotives.pop()  # берем один локомотив
            # ищем направление с максимальным допустимым количеством вагонов
            dest_max = list(carriages_dict.keys())[0]
            max_len = 0
            for key in carriages_dict.keys():
                mas = carriages_dict[key]
                n = len(mas)
                # посчитаем возможную длину для данного направления
                # напишем счетчики
                current_carrying = 0  # for max_locomotive_carrying
                current_length = 0  # train_length_max
                for i in range(n):
                    massa = mas[i].current_mass
                    if massa + current_carrying > locomotive.max_locomotive_carrying or\
                            current_length + 1 > CONST.TRAIN_LENGTH_MAX:
                        break

                    current_length += 1
                    current_carrying += massa
                # а теперь сравним
                if current_length == CONST.TRAIN_LENGTH_MAX:
                    dest_max = key
                    max_len = current_length
                    break
                if current_length > max_len:
                    dest_max = key
                    max_len = current_length
            # получили направление с нужной длиной вагонов. осталось заправить этими вагонами локомотив
            train = Train(locomotive, dest_max)  # создаем новый поезд
            new_carriages = carriages_dict[dest_max][:max_len]  # новый список вагонов
            train.carriages = new_carriages
            self.trains_out.append(train)  # чух-чух
            # удалим использованные вагоны
            for delt in new_carriages:
                self.carriages.remove(delt)
                carriages_dict[dest_max].remove(delt)



    def work_with_carriages(self) -> None:
        self.decompose_trains()
        for carriage in self.carriages:
            if carriage.cargo == None and self.export_cargos != []: #not carriage.cargo
                carriage.cargo = self.export_cargos.pop()
        self.compose_trains()



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

    def __init__(self, n: int = 0, edges: list = None, cities=None, adjacency_matrix=None, directed=False, *args,
                 **kwargs):
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
        self.graph = None
        self.adjacency_matrix = [[]]

        if n < 0:
            raise ValueError('Number of vertices cannot be negative')

        if n > 0 and edges is not None:
            self.graph = ig.Graph(n=n, edges=edges, directed=directed)
            self.adjacency_matrix = self.graph.get_adjacency()

        elif adjacency_matrix is not None:
            self.graph = ig.Graph.Adjacency(matrix=adjacency_matrix)
            self.adjacency_matrix = self.graph.get_adjacency()

        else:
            raise TypeError('Give adjacency_matrix or n + edges pair for graph initialisation')

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
        adj_graph = self.adjacency_matrix
        path_matrix = np.zeros(adj_graph.shape, dtype=float)
        n = adj_graph.shape[0]
        for i in range(0, n):
            for j in range(0, n):
                path_matrix[i, j] = i
                if i != j and adj_graph[i, j] == 0:
                    path_matrix[i, j] = -np.inf
                    adj_graph[i, j] = np.inf

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
        elif np.isinf(path_matrix[i][j]):
            a.append(i)
            a.append("-")
            a.append(j)
        else:
            a = self.restore_path_fw(path_matrix, i, path_matrix[i][j], a)
            a.append(j)
        return a


class World:

    def __init__(self, name: str, init_graph: Graph):

        assert type(init_graph) == Graph, \
            f'init_graph object is not Graph type, current type: {type(init_graph)}'

        self.date = 0
        self.stations = {}
        self.graph: Graph = init_graph
        self.station_index = 0
        self.name = name
        if not os.path.isfile(self.name + '_stats.txt'):
            with open(name + '_stats.txt', 'x'):
                pass


    def __fill_stations_dict(self):
        for vertex in self.graph.get_vertices_list().indices:
            self.stations[vertex] = Station(vertex)

    def __set_station_dest_dict(self, station: Station, p_matrix) -> None:
        for vertex in self.graph.get_vertices_list().indices:
            path = self.graph.restore_path_fw(p_matrix, self.stations[station].station_id, vertex)
            if "-" in path or vertex == station:
                self.stations[station].destination_dict[vertex] = -1
            else:
                self.stations[station].destination_dict[vertex] = path[1]

    def fill_dest_dicts(self) -> None:
        path_matrix = self.graph.floyd_warshall()
        for station in self.stations:
            self.__set_station_dest_dict(station, path_matrix)

    def collect_stats(self) -> None:
        with open(self.name + '_stats.txt', 'a') as stats:
            stats = open(self.name + '_stats.txt', 'a')
            stats.write('\n Date: ' + str(self.date) + '\n')
            for station in self.stations:
                stats.write("Station: " + str(self.stations[station].station_id) + '\n' + '\t' + "Cargos: " + '\n')
                for cargo in self.stations[station].import_cargos:
                    stats.write('\t' + str(cargo.cargo_id) + " [ " + str(cargo.cargo_type) + " ] " + '\n')
                stats.write("Trains:" + '\n')
                for train in self.stations[station].trains_out:
                    stats.write(
                        '\t' + train.name + " : " + station + " -> " + str(train.dest) +
                        " - Locomotive: " + str(train.locomotive.locomotive_id) + '\n')
                    for carriage in train.carriages:
                        stats.write(
                            '\t' + '\t' + str(carriage.carriage_id) + " [ " + carriage.cargo.cargo_type +
                            " ] " + " » " + str(carriage.cargo.destination) + '\n')

    def tick(self) -> None:
        self.collect_stats()
        for station in self.stations:
            for train in self.stations[station].trains_out:
                self.stations[train.destination].add_train_in(self.stations[station].delete_train_out(train.name))
        self.date += 1

        
if __name__ == "__main__":
    name = str(input('Введите имя:'))
    n = int(input('Введите количество станций'))

    stations = [Station(i) for i in range(n)]

    for station in stations:

        station.export_cargos = [Cargo(cargo_id=i*station.station_id,
                                       mass=np.random.randint(1, 51),
                                       destination=np.random.choice([i for i in range(n) if i != station.station_id]),
                                       cargo_type=np.random.choice(list(CargoType))) for i in range(10)]
        station.import_cargos = [Cargo(cargo_id=i*station.station_id,
                                       mass=np.random.randint(1, 51),
                                       destination=station.station_id,
                                       cargo_type=np.random.choice(list(CargoType))) for i in range(10, 20)]

        station.carriages = [Carriage(carriage_id= 10*i*station.station_id,
                                      cargo=Cargo(cargo_id=100 * i * station.station_id,
                                                  mass=np.random.randint(1, 51),
                                                  destination=np.random.randint(0, n),
                                                  cargo_type=np.random.choice(list(CargoType)))) for i in range(20, 25)]
        station.locomotives = [Locomotive(locomotive_id=i*station.station_id) for i in range(7)]
        station.trains_in = [Train(locomotive=Locomotive(locomotive_id=i * 10 * station.station_id),
                                   destination=np.random.randint(0, n),
                                   carriages=[Carriage(carriage_id=1000 * j * i * station.station_id,
                                                       cargo=Cargo(cargo_id=10000 * i * j,
                                                                   mass=np.random.randint(1, 50),
                                                                   destination=np.random.randint(0, 3),
                                                                   cargo_type=np.random.choice(list(CargoType))))
                                              for j in range(15)])
                             for i in range(10)]
        station.trains_out = []

    adj_mat = [[0, 1, 1, 0, 0, 0, 0, 0],
               [0, 0, 0, 1, 0, 0, 0, 0],
               [0, 0, 0, 1, 1, 0, 0, 0],
               [0, 0, 0, 0, 0, 1, 0, 0],
               [0, 0, 0, 0, 0, 1, 1, 0],
               [0, 0, 0, 0, 0, 0, 0, 1],
               [0, 0, 0, 0, 0, 0, 0, 1],
               [1, 0, 0, 0, 0, 0, 0, 0]]

    graph = Graph(adjacency_matrix=adj_mat)
    world = World(name, graph)
    for station in stations:
        world.stations[station.station_id] = station

    world.fill_dest_dicts()

    t = 10
    while True:
        for i in range(t):
            # time.sleep(1000)
            world.tick()
        b = str(input('Хотите продолжить? (да/нет)'))
        if b != 'да':
            break
