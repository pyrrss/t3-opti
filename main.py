
# ============================================================================
# IMPORTS
# ============================================================================

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


# ============================================================================
# Construcción e Implementación del Modelos en Gurobi y CPLEX
# ============================================================================
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

@dataclass
class ConVRPModel:
    instance: Instance
    model: gp.Model
    data: ModelData
    X: gp.tupledict      # X[r, i, j, t]
    Z: gp.tupledict      # Z[r, i]
    W: gp.tupledict      # W[i, t]
    dW: gp.tupledict     # dW[r]
    rW: gp.tupledict     # rW[r, t]

def build_gurobi_model(instance: Instance, 
                       data: ModelData, 
                       name: str | None = None, 
                       env: gp.Env | None = None) -> ConVRPModel:
    """Construye usando Gurobi"""
    m = gp.Model(name or f"conVRP_GRB_{instance.name}", env=env)

    dist = instance.dist

    # --- VARIABLES ---
    X = m.addVars([(r, i, j, t) for r in data.R for (i, j, t) in data.edges], vtype=GRB.BINARY, name="X")
    Z = m.addVars([(r, i) for r in data.R for i in data.NC], vtype=GRB.BINARY, name="Z")
    W = m.addVars(list(data.O), lb=0.0, name="W")
    dW = m.addVars(data.R, lb=0.0, name="dW")
    rW = m.addVars([(r, t) for r in data.R for t in data.D], lb=0.0, name="rW")

    # --- (1) FUNCIÓN OBJETIVO ---
    m.setObjective(gp.quicksum(dist(i, j) * X[r, i, j, t] 
                               for r in data.R for (i, j, t) in data.edges), GRB.MINIMIZE)

    # --- RESTRICCIONES ---
    # (2) Asignación consistente
    m.addConstrs((gp.quicksum(Z[r, i] for r in data.R) == 1 for i in data.NC), name="c2_assignment")

    # (3)-(4) Conservación de flujo y grado (Solo iterando sobre data.O evita el error de la TC2)
    m.addConstrs(
        (gp.quicksum(X[r, i, j, t] for i in data.incoming.get((j, t), [])) == Z[r, j]
         for (j, t) in data.O for r in data.R),
        name="c3_in")

    m.addConstrs(
        (gp.quicksum(X[r, i, j, t] for j in data.outgoing.get((i, t), [])) == Z[r, i]
         for (i, t) in data.O for r in data.R),
        name="c4_out")

    # (5) Salida del depósito para cada vehículo y día
    m.addConstrs(
        (gp.quicksum(X[r, 0, j, t] for j in data.outgoing.get((0, t), [])) == 1
         for t in data.D for r in data.R),
        name="c5_leave_depot")

    # (7) Precedencia temporal (Big-M corregido)
    m.addConstrs(
        (W[i, t] + data.O[i, t].st_it + dist(i, j) <= W[j, t] + data.M * (1 - X[r, i, j, t])
         for r in data.R for (i, j, t) in data.client_edges),
        name="c7_time")

    # (8) Ventanas de tiempo (clientes con ventana estricta)
    m.addConstrs((W[i, t] >= instance.window(i)[0] for (i, t) in data.with_tw), name="c8_tw_open")
    m.addConstrs((W[i, t] <= instance.window(i)[1] for (i, t) in data.with_tw), name="c8_tw_close")

    # (9) Tiempo de salida del depósito
    m.addConstrs(
        (dW[r] + dist(0, j) <= W[j, t] + data.M * (1 - X[r, 0, j, t])
         for r in data.R for (i, j, t) in data.depot_departures),
        name="c9_departure")

    # (10) Tiempo de retorno al depósito
    m.addConstrs(
        (W[i, t] + data.O[i, t].st_it + dist(i, instance.end) <= rW[r, t] + data.M * (1 - X[r, i, instance.end, t])
         for r in data.R for (i, j, t) in data.depot_returns),
        name="c10_return")

    # (11)-(12) Respetar horario global del depósito
    m.addConstrs((dW[r] >= instance.deposit_opening for r in data.R), name="c11_opening")
    m.addConstrs((rW[r, t] <= instance.deposit_closing for r in data.R for t in data.D), name="c12_closing")

    # (14) Capacidad máxima de la ruta (evaluando solo los pedidos del día t)
    m.addConstrs(
        (gp.quicksum(o.q_it * Z[r, o.client] for o in instance.orders_day(t)) <= instance.routes_capacity
         for r in data.R for t in data.D),
        name="c14_capacity")

    m.update()
    return ConVRPModel(instance, m, data, X, Z, W, dW, rW)

@dataclass
class ConVRPModelCplex:
    instance: Instance
    model: CplexModel
    data: ModelData
    X: dict              # X[r, i, j, t]
    Z: dict              # Z[r, i]
    W: dict              # W[i, t]
    dW: dict             # dW[r]
    rW: dict             # rW[r, t]

def build_cplex_model(instance: Instance,
                      data: ModelData,
                      name: str | None = None,
                      relaxed: bool = False) -> ConVRPModelCplex:
    """Construye usando CPLEX (docplex)"""
    mdl = CplexModel(name=name or f"conVRP_{instance.name}_cplex")
    
    dist = instance.dist

    # --- VARIABLES ---
    x_keys = [(r, i, j, t) for r in data.R for (i, j, t) in data.edges]
    z_keys = [(r, i) for r in data.R for i in data.NC]
    
    # Soporte nativo para Relajación Continua requerida en la TC3
    if relaxed:
        X = mdl.continuous_var_dict(x_keys, lb=0, ub=1, name="X")
        Z = mdl.continuous_var_dict(z_keys, lb=0, ub=1, name="Z")
    else:
        X = mdl.binary_var_dict(x_keys, name="X")
        Z = mdl.binary_var_dict(z_keys, name="Z")
        
    W = mdl.continuous_var_dict(list(data.O), lb=0, name="W")
    dW = mdl.continuous_var_dict(data.R, lb=0, name="dW")
    rW = mdl.continuous_var_dict([(r, t) for r in data.R for t in data.D], lb=0, name="rW")

    # --- (1) FUNCIÓN OBJETIVO ---
    mdl.minimize(mdl.sum(dist(i, j) * X[r, i, j, t] for r in data.R for (i, j, t) in data.edges))
    
    # --- RESTRICCIONES ---
    # (2) Asignación consistente
    mdl.add_constraints((mdl.sum(Z[r, i] for r in data.R) == 1 for i in data.NC))
    
    # (3)-(4) Conservación de flujo y grado
    mdl.add_constraints((mdl.sum(X[r, i, j, t] for i in data.incoming.get((j, t), [])) == Z[r, j]
                         for (j, t) in data.O for r in data.R))

    mdl.add_constraints((mdl.sum(X[r, i, j, t] for j in data.outgoing.get((i, t), [])) == Z[r, i]
                         for (i, t) in data.O for r in data.R))

    # (5) Salida del depósito
    mdl.add_constraints((mdl.sum(X[r, 0, j, t] for j in data.outgoing.get((0, t), [])) == 1
                         for t in data.D for r in data.R))
                         
    # (7) Precedencia temporal (Big-M corregido)
    mdl.add_constraints((W[i, t] + data.O[i, t].st_it + dist(i, j) <= W[j, t] + data.M * (1 - X[r, i, j, t])
                         for r in data.R for (i, j, t) in data.client_edges))
                         
    # (8) Ventanas de tiempo
    mdl.add_constraints((W[i, t] >= instance.window(i)[0] for (i, t) in data.with_tw))
    mdl.add_constraints((W[i, t] <= instance.window(i)[1] for (i, t) in data.with_tw))
    
    # (9) Tiempo de salida del depósito
    mdl.add_constraints((dW[r] + dist(0, j) <= W[j, t] + data.M * (1 - X[r, 0, j, t])
                         for r in data.R for (_, j, t) in data.depot_departures))
                         
    # (10) Tiempo de retorno al depósito
    mdl.add_constraints((W[i, t] + data.O[i, t].st_it + dist(i, data.end) <= rW[r, t] + data.M * (1 - X[r, i, data.end, t])
                         for r in data.R for (i, _, t) in data.depot_returns))
                         
    # (11)-(12) Respetar horario global del depósito
    mdl.add_constraints((dW[r] >= instance.deposit_opening for r in data.R))
    mdl.add_constraints((rW[r, t] <= instance.deposit_closing for r in data.R for t in data.D))
    
    # (14) Capacidad máxima
    mdl.add_constraints((mdl.sum(o.q_it * Z[r, o.client] for o in instance.orders_day(t)) <= instance.routes_capacity
                         for r in data.R for t in data.D))

    return ConVRPModelCplex(instance, mdl, data, X, Z, W, dW, rW)

# ============================================================================
# MOTOR DE EXPERIMENTOS Y BENCHMARKING
# ============================================================================

CONFIG = dict(THREADS=1, TIME_LIMIT=600.0, OUTPUT_DIR="resultados_t3")
os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)
CSV_SOLVERS = os.path.join(CONFIG["OUTPUT_DIR"], "resumen_solvers.csv")
GRB_STATUS = {2: "OPTIMAL", 3: "INFEASIBLE", 9: "TIME_LIMIT", 11: "INTERRUPTED"}

def _guardar_fila(fila: dict) -> None:
    existe = os.path.exists(CSV_SOLVERS)
    with open(CSV_SOLVERS, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fila.keys()))
        if not existe: w.writeheader()
        w.writerow(fila)

def _fila_base(instancia, solver, relajado, log):
    return dict(instancia=instancia.name, solver=solver, modo="LP" if relajado else "MIP",
                estado=None, objetivo=None, cota=None, gap_pct=None, nodos=None,
                tiempo_s=None, n_vars=None, n_constr=None, error=None, log=log)

def resolver_gurobi(instancia: Instance, data: ModelData, relajado=False):
    modo = "LP" if relajado else "MIP"
    log = os.path.join(CONFIG["OUTPUT_DIR"], f"{instancia.name}__gurobi__{modo}.log")
    if os.path.exists(log): os.remove(log)
    fila = _fila_base(instancia, "gurobi", relajado, log)
    
    # Se usa la nueva función constructora pasando la data precalculada
    convrp = build_gurobi_model(instancia, data, name=f"{instancia.name}_{modo}")
    m = convrp.model.relax() if relajado else convrp.model
    
    m.Params.Threads = CONFIG["THREADS"]
    m.Params.TimeLimit = CONFIG["TIME_LIMIT"]
    m.Params.LogFile = log
    m.optimize()
    
    fila.update(estado=GRB_STATUS.get(m.Status, str(m.Status)), tiempo_s=round(m.Runtime, 2),
                n_vars=m.NumVars, n_constr=m.NumConstrs)
    if m.SolCount > 0:
        fila["objetivo"] = round(m.ObjVal, 4)
        if not relajado:
            fila.update(cota=round(m.ObjBound, 4), gap_pct=round(m.MIPGap * 100, 4), nodos=int(m.NodeCount))
        else:
            fila.update(cota=round(m.ObjVal, 4), gap_pct=0.0, nodos=0)
            
    _guardar_fila(fila)
    return fila

def resolver_cplex(instancia: Instance, data: ModelData, relajado=False):
    modo = "LP" if relajado else "MIP"
    log = os.path.join(CONFIG["OUTPUT_DIR"], f"{instancia.name}__cplex__{modo}.log")
    fila = _fila_base(instancia, "cplex", relajado, log)
    
    # Se usa la nueva función constructora pasando la data precalculada
    c = build_cplex_model(instancia, data, name=f"{instancia.name}_{modo}_cplex", relaxed=relajado)
    mdl = c.model
    
    fila.update(n_vars=mdl.number_of_variables, n_constr=mdl.number_of_constraints)
    mdl.context.cplex_parameters.threads = CONFIG["THREADS"]
    mdl.context.cplex_parameters.timelimit = CONFIG["TIME_LIMIT"]
    
    try:
        with open(log, "w", encoding="utf-8") as f:
            sol = mdl.solve(log_output=f)
    except Exception as e:
        fila["error"] = str(e)[:200]
        _guardar_fila(fila)
        print(f"  !! CPLEX falló en {instancia.name} [{modo}]: {fila['error']}")
        return fila
        
    sd = mdl.solve_details
    fila.update(estado=sd.status, tiempo_s=round(sd.time, 2))
    
    if sol is not None:
        fila["objetivo"] = round(sol.objective_value, 4)
        if not relajado:
            fila.update(cota=round(sd.best_bound, 4), gap_pct=round(sd.mip_relative_gap * 100, 4),
                        nodos=int(sd.nb_nodes_processed))
        else:
            fila.update(cota=round(sol.objective_value, 4), gap_pct=0.0, nodos=0)
            
    _guardar_fila(fila)
    return fila

RESOLVEDORES = {"gurobi": resolver_gurobi, "cplex": resolver_cplex}

def chequear_entorno(instancias):
    info = dict(python=platform.python_version(), so=platform.platform(), cpu=platform.processor(),
                n_cpu=os.cpu_count(), gurobi=str(gp.gurobi.version()), cplex=cplex.__version__, config=CONFIG)
    with open(os.path.join(CONFIG["OUTPUT_DIR"], "entorno.json"), "w") as f:
        json.dump(info, f, indent=2)
    print("[*] Entorno:")
    print(json.dumps(info, indent=2))
    
    for inst in instancias:
        data = prepare_model_data(inst)
        mdl = build_cplex_model(inst, data).model
        n_v, n_c = mdl.number_of_variables, mdl.number_of_constraints
        print(f"[*] {inst.name}: {n_v} vars, {n_c} restricciones")
        if n_v > 1000 or n_c > 1000:
            print("  ⚠ Excede el límite de CPLEX community (1000/1000): se necesita CPLEX completo/académico.")

def correr_comparacion(instancias, solvers=("gurobi", "cplex")):
    for inst in instancias:
        print(f"\n{'='*60}\nEvaluando instancia: {inst.name}\n{'='*60}")
        # PRECALCULAMOS DATA UNA SOLA VEZ PARA AMBOS SOLVERS
        data = prepare_model_data(inst, prune_edges=True)
        
        for relajado in (True, False):
            for s in solvers:
                fila = RESOLVEDORES[s](inst, data, relajado)
                print(f"✓ {s.upper():<7} | {fila['modo']} | {fila['estado']:<10} | obj={fila['objetivo']} | "
                      f"gap={fila['gap_pct']}% | {fila['tiempo_s']}s | nodos={fila['nodos']}")
    return pd.read_csv(CSV_SOLVERS)

def tabla_validacion_cruzada(df=None, tol_rel=1e-4):
    df = pd.read_csv(CSV_SOLVERS) if df is None else df
    df = df.drop_duplicates(subset=["instancia", "solver", "modo"], keep="last")
    p = df.pivot(index=["instancia", "modo"], columns="solver",
                 values=["objetivo", "tiempo_s", "gap_pct", "nodos"])
    p.columns = [f"{a}_{b}" for a, b in p.columns]
    p = p.reset_index()
    
    if {"objetivo_gurobi", "objetivo_cplex"} <= set(p.columns):
        # Operador de resta arreglado
        p["dif_obj_rel"] = (p["objetivo_gurobi"] - p["objetivo_cplex"]).abs() / p["objetivo_gurobi"].abs()
        p["coincide"] = p["dif_obj_rel"] <= tol_rel
    return p


# -- FLUJO PRINCIPAL --
if __name__ == "__main__":
    print("Nada aun")
