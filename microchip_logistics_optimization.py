"""
Script: ffd_logistics.py
-----------------------------------
Este script implementa e executa uma heurística gulosa (First-Fit Decreasing / Greedy)
para o problema de alocação logística baseado no artigo de Supply Chain Logistics Problem.

Objetivo:
---------
O código busca minimizar o custo total de transporte e armazenamento, atribuindo pedidos
a armazéns e transportadoras disponíveis, levando em consideração:
- pesos dos pedidos,
- produtos compatíveis com cada armazém,
- portos de origem,
- níveis de serviço e capacidade diária dos armazéns.

Modo de operação:
-----------------
O programa aceita datasets externos em CSV localizados no diretório ./datasets.
Se os arquivos CSV não forem encontrados, ele gera automaticamente dados sintéticos equivalentes,
preservando a estrutura original para simulação e testes.

Arquivos aceitos:
-----------------
- datasets/orders.csv
- datasets/warehouses.csv
- datasets/freight_rates.csv
- datasets/vmi_customers.csv

Saídas geradas:
---------------
- output/assignments_result.csv → lista de atribuições (pedido x armazém x custo)
- saída no terminal com tempo total e custo final

Dependências:
-------------
Instale as bibliotecas necessárias com:
    pip install pandas numpy pulp tabulate

Desenvolvido para estudos de otimização logística e análise de heurísticas.
"""

import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import pulp
from tabulate import tabulate

# -----------------------------
# Configuração
# -----------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATASET_DIR = "datasets"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

NUM_ORDERS = 9216
NUM_WAREHOUSES = 15
NUM_PORTS = 11
NUM_COURIERS = 9
SERVICE_LEVELS = ["DTD", "DTP", "CRF"]

VOLUME_DISCOUNTS = [(0, 1.0), (10, 0.95), (50, 0.90), (100, 0.85)]

# -----------------------------
# Estruturas de dados
# -----------------------------
@dataclass
class Order:
    order_id: int
    weight: float
    product_type: int
    origin_port: int
    customer_id: int
    service_level: str

@dataclass
class Warehouse:
    wid: int
    capacity_daily: int
    storage_cost_per_unit: float
    products_allowed: set
    ports_linked: set

@dataclass
class CourierRate:
    courier_id: int
    service_level: str
    mode: str
    weight_brackets: List[Tuple[float, float, float]]

# -----------------------------
# Funções utilitárias
# -----------------------------
def pick_weight():
    return float(np.random.lognormal(mean=1.2, sigma=0.8))

def apply_volume_discount(base_cost: float, total_in_warehouse: int) -> float:
    for thr, factor in reversed(VOLUME_DISCOUNTS):
        if total_in_warehouse >= thr:
            return base_cost * factor
    return base_cost

def get_adjusted_cost(base_cost, used_flag, residual_rate):
    usage_reward = 0 if used_flag else 1
    if residual_rate == 0:
        residual_rate = 0.0001  # Evitar divisão por zero
    return base_cost + usage_reward / residual_rate

# -----------------------------
# Carregamento de CSVs externos
# -----------------------------
def load_orders_csv(path: str) -> Dict[int, Order]:
    df = pd.read_csv(path)
    orders = {}
    for _, row in df.iterrows():
        orders[int(row['order_id'])] = Order(
            order_id=int(row['order_id']),
            weight=float(row['weight']),
            product_type=int(row['product_type']),
            origin_port=int(row['origin_port']),
            customer_id=int(row['customer_id']),
            service_level=str(row['service_level']).strip()
        )
    return orders

def load_warehouses_csv(path: str) -> Dict[int, Warehouse]:
    df = pd.read_csv(path)
    warehouses = {}
    for _, row in df.iterrows():
        products = set(eval(str(row['products_allowed'])))
        ports = set(eval(str(row['ports_linked'])))
        warehouses[int(row['wid'])] = Warehouse(
            wid=int(row['wid']),
            capacity_daily=int(row['capacity_daily']),
            storage_cost_per_unit=float(row['storage_cost_per_unit']),
            products_allowed=products,
            ports_linked=ports
        )
    return warehouses

def load_freight_csv(path: str) -> List[CourierRate]:
    df = pd.read_csv(path)
    freight = []
    grouped = df.groupby(['courier_id', 'service_level', 'mode'])
    for (cid, sl, mode), group in grouped:
        brackets = [(float(r.min_weight), float(r.max_weight), float(r.unit_cost)) for _, r in group.iterrows()]
        freight.append(CourierRate(courier_id=int(cid), service_level=str(sl), mode=str(mode), weight_brackets=brackets))
    return freight

def load_vmi_csv(path: str) -> Dict[int, int]:
    df = pd.read_csv(path)
    return {int(r.customer_id): int(r.warehouse_id) for _, r in df.iterrows()}

# -----------------------------
# Geração sintética (fallback)
# -----------------------------
def generate_synthetic_data():
    print("→ Gerando dados sintéticos (datasets reais não encontrados)...")
    warehouses = {}
    for w in range(NUM_WAREHOUSES):
        warehouses[w] = Warehouse(
            wid=w,
            capacity_daily=random.randint(400, 900),
            storage_cost_per_unit=round(random.uniform(0.5, 2.0), 2),
            products_allowed=set(random.sample(range(10), k=random.randint(4, 10))),
            ports_linked=set(random.sample(range(NUM_PORTS), k=random.randint(3, 7)))
        )

    orders = {}
    for o in range(NUM_ORDERS):
        orders[o] = Order(
            order_id=o,
            weight=round(pick_weight(), 3),
            product_type=random.randint(0, 9),
            origin_port=random.randint(0, NUM_PORTS - 1),
            customer_id=random.randint(0, 2000),
            service_level=random.choices(SERVICE_LEVELS, weights=[0.6, 0.3, 0.1])[0]
        )

    freight_rates = []
    for c in range(NUM_COURIERS):
        for s in SERVICE_LEVELS:
            mode = random.choice(["air", "road"])
            base = random.uniform(3.0, 12.0) if mode == 'air' else random.uniform(0.5, 3.0)
            brackets = []
            for mn, mx in [(0, 2), (2, 5), (5, 10), (10, 9999)]:
                unit = round(base * random.uniform(0.9, 1.1), 3)
                brackets.append((mn, mx, unit))
            freight_rates.append(CourierRate(courier_id=c, service_level=s, mode=mode, weight_brackets=brackets))

    vmi_customers = {random.randint(0, 2000): random.randint(0, NUM_WAREHOUSES - 1) for _ in range(50)}
    return orders, warehouses, freight_rates, vmi_customers

# -----------------------------
# Cálculo de custos
# -----------------------------
def get_unit_transport_cost(order: Order, warehouse: Warehouse, courier: CourierRate) -> float:
    w = order.weight
    unit = next((uc for (mn, mx, uc) in courier.weight_brackets if mn <= w <= mx), courier.weight_brackets[-1][2])
    base = unit * w
    if order.service_level == 'DTD':
        base *= 1.25
    elif order.service_level == 'CRF':
        base *= 1.05
    if courier.mode == 'air':
        base = max(base, 20.0)
    return round(base, 3)

# -----------------------------
# Heurística greedy fiel ao artigo
# -----------------------------
def greedy_assign(orders, warehouses, freight_rates, vmi_customers, strategy="cost-first"):
    start = time.time()
    total_cost = 0
    assigned = []
    warehouse_usage = {wid: 0 for wid in warehouses.keys()}
    warehouses_capacity = {wid: wh.capacity_daily for wid, wh in warehouses.items()}
    warehouses_used_flag = {wid: False for wid in warehouses.keys()}

    orders_list = list(orders.values())
    if strategy == "cost-first":
        def min_cost_for_order(ord_obj):
            costs = []
            for w, wh in warehouses.items():
                if ord_obj.product_type not in wh.products_allowed or ord_obj.origin_port not in wh.ports_linked:
                    continue
                for fr in freight_rates:
                    if fr.service_level != ord_obj.service_level:
                        continue
                    cost = get_unit_transport_cost(ord_obj, wh, fr) + wh.storage_cost_per_unit
                    costs.append(cost)
            return min(costs) if costs else float('inf')
        orders_list.sort(key=min_cost_for_order)
    elif strategy == "weight-first":
        orders_list.sort(key=lambda o: -o.weight)
    else:
        raise ValueError("Unknown strategy, choose 'cost-first' or 'weight-first'")

    for ord_obj in orders_list:
        feasible = []
        for w, wh in warehouses.items():
            if ord_obj.product_type not in wh.products_allowed or ord_obj.origin_port not in wh.ports_linked:
                continue
            if warehouses_capacity[w] < ord_obj.weight:
                continue
            if ord_obj.customer_id in vmi_customers and vmi_customers[ord_obj.customer_id] != w:
                continue
            for fr in freight_rates:
                if fr.service_level != ord_obj.service_level:
                    continue
                base_cost = get_unit_transport_cost(ord_obj, wh, fr) + wh.storage_cost_per_unit
                residual_rate = (warehouses_capacity[w] - warehouse_usage[w]) / wh.capacity_daily
                adj_cost = get_adjusted_cost(base_cost, warehouses_used_flag[w], residual_rate)
                feasible.append((adj_cost, base_cost, w, fr.courier_id, fr.service_level))

        if feasible:
            feasible.sort(key=lambda x: x[0])
            adj_cost, base_cost, w, courier_id, service = feasible[0]
            warehouse_usage[w] += ord_obj.weight
            warehouses_capacity[w] -= ord_obj.weight
            warehouses_used_flag[w] = True
            total_cost += base_cost
            assigned.append((ord_obj.order_id, w, courier_id, service, base_cost))
        else:
            assigned.append((ord_obj.order_id, None, None, None, 0))

    elapsed = time.time() - start
    return {'total_cost': round(total_cost, 2), 'elapsed': elapsed, 'assignments': assigned}

# -----------------------------
# Execução principal
# -----------------------------
def run():
    paths = {
        'orders': os.path.join(DATASET_DIR, 'orders.csv'),
        'warehouses': os.path.join(DATASET_DIR, 'warehouses.csv'),
        'freight': os.path.join(DATASET_DIR, 'freight_rates.csv'),
        'vmi': os.path.join(DATASET_DIR, 'vmi_customers.csv')
    }

    if all(os.path.exists(p) for p in paths.values()):
        print("→ Carregando datasets externos de ./datasets/")
        orders = load_orders_csv(paths['orders'])
        warehouses = load_warehouses_csv(paths['warehouses'])
        freight_rates = load_freight_csv(paths['freight'])
        vmi_customers = load_vmi_csv(paths['vmi'])
    else:
        orders, warehouses, freight_rates, vmi_customers = generate_synthetic_data()

    resultados = []

    # Execução da heurística Cost-first
    result_cost = greedy_assign(orders, warehouses, freight_rates, vmi_customers, strategy="cost-first")
    print(f"Heurística Cost-first executada em {result_cost['elapsed']:.2f}s, custo total = ${result_cost['total_cost']:,}")
    resultados.append({"strategy": "cost-first", "total_cost": result_cost["total_cost"], "elapsed": result_cost["elapsed"]})

    # Execução da heurística Weight-first
    result_weight = greedy_assign(orders, warehouses, freight_rates, vmi_customers, strategy="weight-first")
    print(f"Heurística Weight-first executada em {result_weight['elapsed']:.2f}s, custo total = ${result_weight['total_cost']:,}")
    resultados.append({"strategy": "weight-first", "total_cost": result_weight["total_cost"], "elapsed": result_weight["elapsed"]})

    # Salvar resultados da última execução em CSV
    df = pd.DataFrame(result_weight['assignments'], columns=['order_id', 'warehouse', 'courier', 'service', 'total_cost'])
    df.to_csv(os.path.join(OUTPUT_DIR, 'assignments_result.csv'), index=False)
    print(f"→ Resultados salvos em {OUTPUT_DIR}/assignments_result.csv")

    # Exibir resumo formatado
    mostrar_resultados(resultados)
def mostrar_resultados(resultados: List[Dict]):
    table = []
    for res in resultados:
        table.append([res['strategy'], f"${res['total_cost']:,}", f"{res['elapsed']:.2f}s"])
    print("\nResumo dos Resultados:")
    print(tabulate(table, headers=["Estratégia", "Custo Total", "Tempo de Execução"], tablefmt="grid"))

if __name__ == '__main__':
    run()
