
from __future__ import annotations
from dataclasses import dataclass, field
import math
import json
import os
import time
import platform
import statistics



# ============================================================================
# Definición de la Estructura de Datos e Instancias
# ============================================================================
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

# 1. Definir directorio objetivo en Google Colab
DIR_INSTANCES = "./"
os.makedirs(DIR_INSTANCES, exist_ok=True)

PATH_C101 = os.path.join(DIR_INSTANCES, "Solomon_C101_15c_3d_multi.json")
PATH_R101 = os.path.join(DIR_INSTANCES, "Solomon_R101_15c_3d_multi.json")

# 2. Contenido JSON de las instancias modificadas
json_c101_str = """{
  "name": "Solomon_C101_15c_3d_mod",
  "source": "CVRPLIB - Solomon C101 adaptado",
  "days": [1, 2, 3],
  "routes": [1, 2, 3],
  "routes_capacity": 115,
  "deposit_xy": [40.0, 50.0],
  "deposit_opening": 0.0,
  "deposit_closing": 1236.0,
  "description": "C101 (clusterizada): ventanas ensanchadas +-120, servicio 50, Q=115 (capacidad ajustada).",
  "params": {},
  "clients": [
    {"index": 1, "x": 45, "y": 68, "window": [792.0, 1087.0]},
    {"index": 2, "x": 45, "y": 70, "window": [705.0, 990.0]},
    {"index": 3, "x": 42, "y": 66, "window": [0.0, 266.0]},
    {"index": 4, "x": 42, "y": 68, "window": [607.0, 902.0]},
    {"index": 5, "x": 42, "y": 65, "window": [0.0, 187.0]},
    {"index": 6, "x": 40, "y": 69, "window": [501.0, 822.0]},
    {"index": 7, "x": 40, "y": 66, "window": [50.0, 345.0]},
    {"index": 8, "x": 38, "y": 68, "window": [135.0, 444.0]},
    {"index": 9, "x": 38, "y": 70, "window": [414.0, 725.0]},
    {"index": 10, "x": 35, "y": 66, "window": [237.0, 530.0]},
    {"index": 11, "x": 35, "y": 69, "window": [328.0, 625.0]},
    {"index": 12, "x": 25, "y": 85, "window": [532.0, 841.0]},
    {"index": 13, "x": 22, "y": 85, "window": [0.0, 212.0]},
    {"index": 14, "x": 22, "y": 80, "window": [447.0, 760.0]},
    {"index": 15, "x": 20, "y": 85, "window": [264.0, 549.0]}
  ],
  "orders": [
    {"client": 1, "day": 1, "q_it": 10, "st_it": 50},
    {"client": 2, "day": 1, "q_it": 20, "st_it": 50},
    {"client": 3, "day": 1, "q_it": 10, "st_it": 50},
    {"client": 4, "day": 1, "q_it": 30, "st_it": 50},
    {"client": 5, "day": 1, "q_it": 10, "st_it": 50},
    {"client": 6, "day": 1, "q_it": 20, "st_it": 50},
    {"client": 7, "day": 1, "q_it": 20, "st_it": 50},
    {"client": 8, "day": 1, "q_it": 10, "st_it": 50},
    {"client": 9, "day": 1, "q_it": 10, "st_it": 50},
    {"client": 10, "day": 1, "q_it": 10, "st_it": 50},
    {"client": 11, "day": 1, "q_it": 20, "st_it": 50},
    {"client": 12, "day": 1, "q_it": 20, "st_it": 50},
    {"client": 13, "day": 1, "q_it": 10, "st_it": 50},
    {"client": 14, "day": 1, "q_it": 10, "st_it": 50},
    {"client": 15, "day": 1, "q_it": 40, "st_it": 50},
    {"client": 1, "day": 2, "q_it": 15, "st_it": 50},
    {"client": 2, "day": 2, "q_it": 10, "st_it": 50},
    {"client": 3, "day": 2, "q_it": 20, "st_it": 50},
    {"client": 4, "day": 2, "q_it": 10, "st_it": 50},
    {"client": 5, "day": 2, "q_it": 30, "st_it": 50},
    {"client": 6, "day": 2, "q_it": 10, "st_it": 50},
    {"client": 7, "day": 2, "q_it": 10, "st_it": 50},
    {"client": 8, "day": 2, "q_it": 20, "st_it": 50},
    {"client": 9, "day": 2, "q_it": 20, "st_it": 50},
    {"client": 10, "day": 2, "q_it": 30, "st_it": 50},
    {"client": 11, "day": 2, "q_it": 10, "st_it": 50},
    {"client": 12, "day": 2, "q_it": 10, "st_it": 50},
    {"client": 13, "day": 2, "q_it": 20, "st_it": 50},
    {"client": 14, "day": 2, "q_it": 20, "st_it": 50},
    {"client": 15, "day": 2, "q_it": 10, "st_it": 50},
    {"client": 1, "day": 3, "q_it": 20, "st_it": 50},
    {"client": 2, "day": 3, "q_it": 30, "st_it": 50},
    {"client": 3, "day": 3, "q_it": 10, "st_it": 50},
    {"client": 4, "day": 3, "q_it": 20, "st_it": 50},
    {"client": 5, "day": 3, "q_it": 10, "st_it": 50},
    {"client": 6, "day": 3, "q_it": 20, "st_it": 50},
    {"client": 7, "day": 3, "q_it": 10, "st_it": 50},
    {"client": 8, "day": 3, "q_it": 30, "st_it": 50},
    {"client": 9, "day": 3, "q_it": 10, "st_it": 50},
    {"client": 10, "day": 3, "q_it": 10, "st_it": 50},
    {"client": 11, "day": 3, "q_it": 30, "st_it": 50},
    {"client": 12, "day": 3, "q_it": 20, "st_it": 50},
    {"client": 13, "day": 3, "q_it": 10, "st_it": 50},
    {"client": 14, "day": 3, "q_it": 30, "st_it": 50},
    {"client": 15, "day": 3, "q_it": 20, "st_it": 50}
  ]
}"""

json_r101_str = """{
  "name": "Solomon_R101_15c_3d_mod",
  "source": "CVRPLIB - Solomon R101 adaptado",
  "days": [1, 2, 3],
  "routes": [1, 2, 3],
  "routes_capacity": 108,
  "deposit_xy": [35.0, 35.0],
  "deposit_opening": 0.0,
  "deposit_closing": 960.0,
  "description": "R101 (dispersa): ventanas +-20, servicio 15, Q=108 (muy ajustada), clientes 5,10,15 sin ventana.",
  "params": {},
  "clients": [
    {"index": 1, "x": 41, "y": 49, "window": [141.0, 191.0]},
    {"index": 2, "x": 35, "y": 17, "window": [30.0, 80.0]},
    {"index": 3, "x": 55, "y": 45, "window": [96.0, 146.0]},
    {"index": 4, "x": 55, "y": 20, "window": [129.0, 179.0]},
    {"index": 5, "x": 15, "y": 30, "window": null},
    {"index": 6, "x": 25, "y": 30, "window": [79.0, 129.0]},
    {"index": 7, "x": 20, "y": 50, "window": [61.0, 111.0]},
    {"index": 8, "x": 10, "y": 43, "window": [75.0, 125.0]},
    {"index": 9, "x": 55, "y": 60, "window": [77.0, 127.0]},
    {"index": 10, "x": 30, "y": 60, "window": null},
    {"index": 11, "x": 20, "y": 65, "window": [47.0, 97.0]},
    {"index": 12, "x": 50, "y": 35, "window": [43.0, 93.0]},
    {"index": 13, "x": 30, "y": 25, "window": [139.0, 189.0]},
    {"index": 14, "x": 15, "y": 10, "window": [103.0, 153.0]},
    {"index": 15, "x": 30, "y": 5, "window": null}
  ],
  "orders": [
    {"client": 1, "day": 1, "q_it": 18, "st_it": 15},
    {"client": 2, "day": 1, "q_it": 26, "st_it": 15},
    {"client": 3, "day": 1, "q_it": 11, "st_it": 15},
    {"client": 4, "day": 1, "q_it": 30, "st_it": 15},
    {"client": 5, "day": 1, "q_it": 21, "st_it": 15},
    {"client": 6, "day": 1, "q_it": 19, "st_it": 15},
    {"client": 7, "day": 1, "q_it": 15, "st_it": 15},
    {"client": 8, "day": 1, "q_it": 16, "st_it": 15},
    {"client": 9, "day": 1, "q_it": 29, "st_it": 15},
    {"client": 10, "day": 1, "q_it": 23, "st_it": 15},
    {"client": 11, "day": 1, "q_it": 20, "st_it": 15},
    {"client": 12, "day": 1, "q_it": 25, "st_it": 15},
    {"client": 13, "day": 1, "q_it": 14, "st_it": 15},
    {"client": 14, "day": 1, "q_it": 12, "st_it": 15},
    {"client": 15, "day": 1, "q_it": 27, "st_it": 15},
    {"client": 1, "day": 2, "q_it": 22, "st_it": 15},
    {"client": 2, "day": 2, "q_it": 14, "st_it": 15},
    {"client": 3, "day": 2, "q_it": 25, "st_it": 15},
    {"client": 4, "day": 2, "q_it": 18, "st_it": 15},
    {"client": 5, "day": 2, "q_it": 12, "st_it": 15},
    {"client": 6, "day": 2, "q_it": 28, "st_it": 15},
    {"client": 7, "day": 2, "q_it": 24, "st_it": 15},
    {"client": 8, "day": 2, "q_it": 19, "st_it": 15},
    {"client": 9, "day": 2, "q_it": 15, "st_it": 15},
    {"client": 10, "day": 2, "q_it": 17, "st_it": 15},
    {"client": 11, "day": 2, "q_it": 26, "st_it": 15},
    {"client": 12, "day": 2, "q_it": 13, "st_it": 15},
    {"client": 13, "day": 2, "q_it": 21, "st_it": 15},
    {"client": 14, "day": 2, "q_it": 30, "st_it": 15},
    {"client": 15, "day": 2, "q_it": 16, "st_it": 15},
    {"client": 1, "day": 3, "q_it": 15, "st_it": 15},
    {"client": 2, "day": 3, "q_it": 29, "st_it": 15},
    {"client": 3, "day": 3, "q_it": 19, "st_it": 15},
    {"client": 4, "day": 3, "q_it": 22, "st_it": 15},
    {"client": 5, "day": 3, "q_it": 27, "st_it": 15},
    {"client": 6, "day": 3, "q_it": 14, "st_it": 15},
    {"client": 7, "day": 3, "q_it": 18, "st_it": 15},
    {"client": 8, "day": 3, "q_it": 25, "st_it": 15},
    {"client": 9, "day": 3, "q_it": 20, "st_it": 15},
    {"client": 10, "day": 3, "q_it": 28, "st_it": 15},
    {"client": 11, "day": 3, "q_it": 12, "st_it": 15},
    {"client": 12, "day": 3, "q_it": 21, "st_it": 15},
    {"client": 13, "day": 3, "q_it": 26, "st_it": 15},
    {"client": 14, "day": 3, "q_it": 17, "st_it": 15},
    {"client": 15, "day": 3, "q_it": 24, "st_it": 15}
  ]
}"""

# 3. Convertir a multi-día y escribir los JSON
# Patrón de pedidos: cliente -> días en los que pide (frecuencia alta / media / baja)
PATRON_A = {1:[1,2,3], 2:[1,3], 3:[1,2], 4:[1,2,3], 5:[2,3], 6:[1,3], 7:[1], 8:[2,3],
            9:[1,2,3], 10:[2], 11:[1,3], 12:[1,2], 13:[3], 14:[1,2,3], 15:[2,3]}
PATRON_B = {1:[1,2,3], 2:[1,2], 3:[2,3], 4:[1,2,3], 5:[1,3], 6:[1,2], 7:[3], 8:[1,2,3],
            9:[1,3], 10:[2], 11:[2,3], 12:[1], 13:[1,2,3], 14:[2,3], 15:[1,2]}
Q_A, Q_B = 90, 80            # capacidad recalibrada (antes 115 / 108)
AMPLIAR_VENTANA_B = 20       # min extra a cada lado en B (±20 -> ±40)

def hacer_multidia(json_str, patron, Q, ampliar=0.0):
    d = json.loads(json_str)
    d["orders"] = [o for o in d["orders"] if o["day"] in patron[o["client"]]]
    d["routes_capacity"] = Q
    for c in d["clients"]:
        if c["window"] is not None and ampliar:
            a, b = c["window"]
            c["window"] = [max(d["deposit_opening"], a - ampliar), min(d["deposit_closing"], b + ampliar)]
    d["name"] = d["name"].replace("_mod", "_multi")
    d["description"] += f" | Multi-día: {len(d['orders'])}/45 pedidos, Q={Q}."
    return json.dumps(d, indent=2)

with open(PATH_C101, "w", encoding="utf-8") as f:
    f.write(hacer_multidia(json_c101_str, PATRON_A, Q_A))
with open(PATH_R101, "w", encoding="utf-8") as f:
    f.write(hacer_multidia(json_r101_str, PATRON_B, Q_B, AMPLIAR_VENTANA_B))

instancia_A_mod = Instance.load_json(PATH_C101)
instancia_B_mod = Instance.load_json(PATH_R101)

for inst in (instancia_A_mod, instancia_B_mod):
    por_dia = {t: len(inst.orders_day(t)) for t in inst.days}
    dem_dia = {t: sum(o.q_it for o in inst.orders_day(t)) for t in inst.days}
    print(f"{inst.name}: {len(inst.orders)}/{inst.n*len(inst.days)} pedidos | "
          f"clientes/día={por_dia} | demanda/día={dem_dia} | Q={inst.routes_capacity}")

print("\n" + "="*80)
print("INSTANCIAS MULTI-DÍA CARGADAS")
print("="*80)
