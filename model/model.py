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


class World:

    # II priority task, cannot be done without graph implementation

    def __init__(self):
        self.date = 0
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



    def log(self) -> None:
        pass

    # Maybe there is to be some additional functions if you need
