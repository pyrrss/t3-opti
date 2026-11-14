from __future__ import annotations
from dataclasses import dataclass, field
import math
import json
import os
import time
import platform
import statistics
import pandas as pd
import csv

# --- Motores de Optimización ---
import gurobipy as gp
from gurobipy import GRB

import cplex
from docplex.mp.model import Model as CplexModel


@dataclass(frozen=True)
class Client:
    index: int
    x: int
    y: int
    window: tuple[float, float] | None

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "x": self.x,
            "y": self.y,
            "window": list(self.window) if self.window is not None else None
        }

    @classmethod
    def from_dict(cls, d: dict) -> Client:
        win = tuple(d["window"]) if d.get("window") is not None else None
        return cls(index=d["index"], x=d["x"], y=d["y"], window=win)


@dataclass(frozen=True)
class Order:
    client: int  # i
    day: int     # t
    q_it: int
    st_it: float

    def to_dict(self) -> dict:
        return {
            "client": self.client,
            "day": self.day,
            "q_it": self.q_it,
            "st_it": self.st_it
        }

    @classmethod
    def from_dict(cls, d: dict) -> Order:
        return cls(client=d["client"], day=d["day"], q_it=d["q_it"], st_it=d["st_it"])


@dataclass
class Instance:
    name: str
    source: str
    days: list[int]
    routes: list[int]
    routes_capacity: int
    deposit_xy: tuple[float, float]
    deposit_opening: float
    deposit_closing: float
    clients: list[Client]
    orders: list[Order]
    description: str = ""
    params: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        indexes = [c.index for c in self.clients]
        if indexes != list(range(1, len(indexes) + 1)):
            raise ValueError(f"client indexes must be 1..n, got {indexes}")

        self._orders_dict: dict[tuple[int, int], Order] = {}
        self._orders_by_day: dict[int, list[Order]] = {t: [] for t in self.days}

        for order in self.orders:
            if not 1 <= order.client <= self.n:
                raise ValueError(f"order for a client that doesn't exist: {order}")
            if order.day not in self.days:
                raise ValueError(f"order on a day that doesn't exist: {order}")
            if order.q_it <= 0 or order.st_it < 0:
                raise ValueError(f"invalid q_it/st_it: {order}")
            if (order.client, order.day) in self._orders_dict:
                raise ValueError(f"duplicate order: {order}")

            self._orders_dict[(order.client, order.day)] = order
            self._orders_by_day[order.day].append(order)

        # Validación corregida: verifica que el cliente tenga pedidos al menos 1 de los días
        clients_without_orders = set(indexes) - {o.client for o in self.orders}
        if clients_without_orders:
            raise ValueError(f"clients without orders in the entire horizon: {sorted(clients_without_orders)}")

        for client in self.clients:
            if client.window is not None:
                a, b = client.window
                if not self.deposit_opening <= a <= b <= self.deposit_closing:
                    raise ValueError(f"window outside depot hours: {client}")

    @property
    def n(self) -> int:
        return len(self.clients)

    @property
    def end(self) -> int:
        return self.n + 1

    def xy(self, nodo: int) -> tuple[float, float]:
        if nodo == 0 or nodo == self.end:
            return self.deposit_xy
        c = self.clients[nodo - 1]
        return float(c.x), float(c.y)

    def dist(self, i: int, j: int) -> float:
        (xi, yi), (xj, yj) = self.xy(i), self.xy(j)
        return math.hypot(xi - xj, yi - yj)

    def orders_day(self, t: int) -> list[Order]:
        return self._orders_by_day.get(t, [])
        
    def active_clients(self, t: int) -> list[int]:
        """
        ESTO ES NUEVO: Devuelve lista de clientes activos en día t.
        """
        return [o.client for o in self.orders_day(t)]

    def order(self, i: int, t: int) -> Order | None:
        return self._orders_dict.get((i, t))

    def window(self, i: int) -> tuple[float, float] | None:
        return self.clients[i - 1].window

    def save_json(self, filepath: str) -> None:
        data = {
            "name": self.name,
            "source": self.source,
            "days": self.days,
            "routes": self.routes,
            "routes_capacity": self.routes_capacity,
            "deposit_xy": list(self.deposit_xy),
            "deposit_opening": self.deposit_opening,
            "deposit_closing": self.deposit_closing,
            "description": self.description,
            "params": self.params,
            "clients": [c.to_dict() for c in self.clients],
            "orders": [o.to_dict() for o in self.orders]
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load_json(cls, filepath: str) -> Instance:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        clients = [Client.from_dict(c) for c in data["clients"]]
        orders = [Order.from_dict(o) for o in data["orders"]]
        return cls(
            name=data["name"],
            source=data["source"],
            days=data["days"],
            routes=data["routes"],
            routes_capacity=data["routes_capacity"],
            deposit_xy=tuple(data["deposit_xy"]),
            deposit_opening=data["deposit_opening"],
            deposit_closing=data["deposit_closing"],
            clients=clients,
            orders=orders,
            description=data.get("description", ""),
            params=data.get("params", {})
        )


Edge = tuple[int, int, int]  # (i, j, t): arco de i a j el día t

@dataclass
class ModelData:
    R: list[int]
    D: list[int]
    NC: list[int]                                   # Todos los clientes (histórico)
    active_clients: dict[int, list[int]]            # NUEVO: Clientes activos por día t -> [i, ...]
    end: int
    O: dict[tuple[int, int], Order]                 # orders (i, t)
    edges: list[Edge]
    pruned: int
    incoming: dict[tuple[int, int], list[int]]      # (j, t) -> [i, ...]
    outgoing: dict[tuple[int, int], list[int]]      # (i, t) -> [j, ...]
    client_edges: list[Edge]                        # i -> j  (clients)
    depot_departures: list[Edge]                    # 0 -> j
    depot_returns: list[Edge]                       # i -> end
    with_tw: list[tuple[int, int]]                  # orders with a time window
    M: float


def prepare_model_data(instance: Instance, prune_edges: bool = True) -> ModelData:
    end = instance.end
    O = {(o.client, o.day): o for o in instance.orders}
    
    # Construcción de clientes activos por día (Para corregir el BUG de la TC2)
    active_clients_by_day = {t: instance.active_clients(t) for t in instance.days}

    # IMPORTANTE: build_edges_set DEBE considerar solo los clientes activos
    edges, pruned = build_edges_set(instance, prune_edges)

    incoming: dict[tuple[int, int], list[int]] = {}
    outgoing: dict[tuple[int, int], list[int]] = {}
    client_edges, depot_departures, depot_returns = [], [], []
    
    for i, j, t in edges:
        outgoing.setdefault((i, t), []).append(j)
        incoming.setdefault((j, t), []).append(i)
        if i != 0 and j != end:
            client_edges.append((i, j, t))
        elif i == 0 and j != end:
            depot_departures.append((0, j, t))
        elif j == end and i != 0:
            depot_returns.append((i, end, t))

    max_st = max((o.st_it for o in instance.orders), default=0.0)
    max_tt = max((instance.dist(i, j) for i, j, _ in edges), default=0.0)
    M = instance.deposit_closing + max_st + max_tt

    return ModelData(
        R=instance.routes, 
        D=instance.days, 
        NC=list(range(1, instance.n + 1)), 
        active_clients=active_clients_by_day,  # <--- Agregado
        end=end,
        O=O, 
        edges=edges, 
        pruned=pruned, 
        incoming=incoming, 
        outgoing=outgoing,
        client_edges=client_edges, 
        depot_departures=depot_departures,
        depot_returns=depot_returns,
        with_tw=[(i, t) for (i, t) in O if instance.window(i) is not None],
        M=M,
    )


def earliest_start(instance: Instance, i: int) -> float:
    arrival = instance.deposit_opening + instance.dist(0, i)
    window = instance.window(i)
    return max(arrival, window[0]) if window else arrival


def latest_start(instance: Instance, j: int) -> float:
    window = instance.window(j)
    return window[1] if window else instance.deposit_closing


def build_edges_set(instance: Instance, prune: bool = True) -> tuple[list[Edge], int]:
    end = instance.end
    edges: list[Edge] = []
    pruned = 0
    for t in instance.days:
        orders_t = {o.client: o for o in instance.orders_day(t)}

        # Arco ficticio
        edges.append((0, end, t))
        for j, order_j in orders_t.items():
            if (earliest_start(instance, j) > latest_start(instance, j)
                or earliest_start(instance, j) + order_j.st_it + instance.dist(j, end) > instance.deposit_closing):
                raise ValueError(f"El cliente {j} no puede ser atendido el día {t} dentro de su ventana/horario de depósito.")
            edges.append((0, j, t))
            edges.append((j, end, t))

        for i, order_i in orders_t.items():
            for j, order_j in orders_t.items():
                if i == j:
                    continue
                if prune:
                    arrival_j = earliest_start(instance, i) + order_i.st_it + instance.dist(i, j)
                    window_j = instance.window(j)
                    if window_j is not None:
                        arrival_j = max(arrival_j, window_j[0])
                    if (arrival_j > latest_start(instance, j)
                        or arrival_j + order_j.st_it + instance.dist(j, end) > instance.deposit_closing):
                        pruned += 1
                        continue
                edges.append((i, j, t))
    return edges, pruned


# -- FLUJO PRINCIPAL --
if __name__ == "__main__":
    print("Nada aun")
