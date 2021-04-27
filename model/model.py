from __future__ import annotations

from enum import Enum
from typing import (Final,
                    List)
from const import constants

CONST = constants.Constants(CARRIAGE_WEIGHT=1,
                            MAX_LOCOMOTIVE_CARRYING=100,
                            TRAIN_LENGTH_MAX=50)


class CargoType(Enum):
    PEOPLE = 0
    LIQUID = 1
    CONTAINER = 2


class Graph:
    """
        Реализовать:
        Граф в виде матрицы смежности. Можно сделать еще список смежности.
        1. Объединение графов
        2. Добавление вершины; Удаление вершины;
        3. Функция, которая показывает соседние вершины графа;
        4. Документация.

        5. Добавить проверки при добавлнии и удалении вершин
    """
    def __init__(self, V: set, E: set, *args, **kwargs):
        """
        Вершины (пока что) - цифры
        Ребра - кортежи из двух цифр
        Args:
            V: set of vertices
            U: set of edges
            *args: smt
            **kwargs: smt
        """
        self.V = V
        self.E = E
        self.len_V = len(V)
        self.len_E = len(E)
        self.adjacency_matrix = [[0 for vertex in V] for vertex in V]
        self.__pull_adjacency_matrix(self.E)

    def __str__(self):
        return '\n'.join(str(list) for list in self.adjacency_matrix)

    def __pull_adjacency_matrix(self, E):
        """
        This function pulls our adjacency_matrix with 1 if edge does exist
        Args:
            E: set of edges
        Returns: None
        """
        for edge in E:
            self.adjacency_matrix[edge[0] - 1][edge[1] - 1] = 1
            self.adjacency_matrix[edge[1] - 1][edge[0] - 1] = 1

    def add_vertex(self, vertex, new_edges):
        """
        This function can add new vertex
        Args:
            new_edges: New edges which correspond to the vertex
            vertex: Our new vertex
        Returns: None
        """
        self.adjacency_matrix.append([0 for _ in range(self.len_V)])
        self.__pull_adjacency_matrix(new_edges)

    def del_vertex(self, vertex):
        """
        This function can delete vertex
        Args:
            vertex: This is the vertex that we'll delete
        Returns: None
        """
        self.adjacency_matrix.pop(vertex)

    def join_graphs(self):
        pass

    def get_neighboring_vertices(self):
        pass

    def get_graph(self):
        return self.adjacency_matrix


class World:

    # II priority task, cannot be done without graph implementation

    def __init__(self):
        self.data = 0
        self.stations = []
        self.graph = []


class Cargo:

    def __init__(self, name: int, mass: int, destination: int, cargo_type: CargoType):
        self.name = name
        self.mass = mass
        self.destination = destination
        self.cargo_type = cargo_type


class Carriage:

    carriage_weight: Final = CONST.CARRIAGE_WEIGHT

    def __init__(self, name: int, cargo: Cargo = None):
        self.name = name
        self.cargo = cargo


class Locomotive:

    max_locomotive_carrying: Final = CONST.MAX_LOCOMOTIVE_CARRYING

    def __init__(self, name: int):
        self.name = name


class Train:

    train_length_max: Final = CONST.TRAIN_LENGTH_MAX

    def __init__(self, locomotive: Locomotive, destination: int, carriages: List[Carriage] = None):
        if carriages is None:
            carriages = []

        self.locomotive = locomotive
        self.destination = destination
        self.carriages = carriages

    def am_i_legal(self) -> bool:
        # Function which check if this train actually able to move (might be renamed)
        if len(self.carriages) > self.train_length_max:
            return False

        # sum(map(lambda x: x.carriage_weight + x.cargo.mass, self.carriages))
        sum_mass = 0
        for carriage in self.carriages:
            sum_mass += carriage.carriage_weight + carriage.cargo.mass
        if self.locomotive.max_locomotive_carrying > sum_mass:
            return False
        return True


class Station:

    def __init__(self, name: int) -> None:
        self.name = name
        self.export_cargos = []
        self.import_cargos = []
        self.carriages = []
        self.locomotives = []
        self.trains_in = []
        self.trains_out = []

    def add_export_cargo(self, cargo: Cargo) -> None:
        self.export_cargos.append(cargo)

    def delete_export_cargo(self, name: int) -> Cargo:
        index = 0
        for cargo in self.export_cargos:
            if cargo.name == name:
                return self.export_cargos.pop(index)
            index += 1
        raise ValueError("element is not in list")

    def add_import_cargo(self, cargo: Cargo) -> None:
        self.import_cargos.append(cargo)

    def delete_import_cargo(self, name: int) -> Cargo:
        index = 0
        for cargo in self.import_cargos:
            if cargo.name == name:
                return self.import_cargos.pop(index)
            index += 1
        raise ValueError("element is not in list")

    def add_carriage(self, carriage: Carriage) -> None:
        self.carriages.append(carriage)

    def delete_carriage(self, name: int) -> Carriage:
        index = 0
        for carriage in self.carriages:
            if carriage.name == name:
                return self.carriages.pop(index)
            index += 1
        raise ValueError("element is not in list")

    def add_locomotive(self, locomotive: Locomotive) -> None:
        self.locomotives.append(locomotive)

    def delete_locomotive(self, name: int) -> Locomotive:
        index = 0
        for locomotive in self.locomotives:
            if locomotive.name == name:
                return self.locomotives.pop(index)
            index += 1
        raise ValueError("element is not in list")

    def add_train_in(self, train_in: Train) -> None:
        self.trains_in.append(train_in)

    def delete_train_in(self, name: int) -> Train:
        index = 0
        for train in self.trains_in:
            if train.name == name:
                return self.trains_in.pop(index)
            index += 1
        raise ValueError("element is not in list")

    def add_train_out(self, train_out: Train) -> None:
        self.trains_out.append(train_out)

    def delete_train_out(self, name: int) -> Train:
        index = 0
        for train in self.trains_out:
            if train.name == name:
                return self.trains_out.pop(index)
            index += 1
        raise ValueError("element is not in list")

    def decompose_trains(self) -> None:
        for train in self.trains_in:
            locomotive = train.locomotive
            carriages = train.carriages
            self.add_locomotive(locomotive)
            for i in range(len(carriages)):
                if carriages[i].cargo.destination == self.name:
                    self.add_import_cargo(carriages[i].cargo)
                    carriages[i].cargo = None
                self.add_carriage(carriages[i])

    def compose_trains(self) -> None:
        pass

    def log(self) -> None:
        pass

    # Maybe there is to be some additional functions if you need


if __name__ == "__main__":
    graph = Graph({1, 2, 3}, {(1, 2), (1, 3), (3, 3)})
    print(graph)
