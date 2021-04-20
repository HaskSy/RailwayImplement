from __future__ import annotations
from enum import Enum
from typing import (Final,
                    List)
from const import constants

CONST = constants.Constants(CARRIAGE_WEIGHT=1,
                            LOCOMOTIVE_CARRYING=100,
                            TRAIN_LENGTH_MAX=50)


class CargoType(Enum):
    PEOPLE = 0
    LIQUID = 1
    CONTAINER = 2


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

    locomotive_carrying: Final = CONST.LOCOMOTIVE_CARRYING

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
        if self.locomotive.locomotive_carrying > sum_mass:
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
