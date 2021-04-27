from __future__ import annotations
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
    pass


class World:

    # II priority task, cannot be done without graph implementation

    def __init__(self):
        self.date = 0
        self.stations: List[Station] = []
        self.graph: Graph

    def _fill_dest_dicts(self) -> None:
        """
        Raise __set_station_dest_dict for each station
        """
        pass

    def __set_station_dest_dict(self, station: Station) -> None:
        pass

