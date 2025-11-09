"""
Script: microchip_logistics_optimization.py
-----------------------------------
Implementa a heurística **First-Fit Decreasing (FFD)** para otimização logística
baseada no problema Supply Chain Logistics Problem Dataset.

Objetivo:
---------
Minimizar o custo total de transporte e armazenamento,
atribuindo pedidos a armazéns e transportadoras disponíveis,
considerando:
- pesos dos pedidos,
- compatibilidade de produtos e portos,
- níveis de serviço (DTD, DTP, CRF),
- capacidade diária dos armazéns.

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
- output/execution_summary.txt → resumo com custo e tempo
- saída no terminal com resumo tabulado

Dependências:
-------------
    pip install pandas numpy pulp tabulate

Desenvolvido para estudos de otimização logística e análise de heurísticas.
"""

import os
import random
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
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
    capacity_daily: float
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

def get_unit_transport_cost(order: Order, courier: CourierRate) -> float:
    """Calcula o custo de transporte unitário baseado no peso e tipo de serviço."""
    w = order.weight
    for mn, mx, uc in courier.weight_brackets:
        if mn <= w <= mx:
            unit = uc
            break
    else:
        unit = courier.weight_brackets[-1][2]
    base = unit * w
    if order.service_level == 'DTD':
        base *= 1.25
    elif order.service_level == 'CRF':
        base *= 1.05
    if courier.mode == 'air':
        base = max(base, 20.0)
    return round(base, 3)

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
            capacity_daily=float(row['capacity_daily']),
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
            service_level=random.choice(SERVICE_LEVELS)
        )

    freight_rates = []
    for c in range(NUM_COURIERS):
        for s in SERVICE_LEVELS:
            mode = random.choice(["air", "road"])
            base = random.uniform(3.0, 12.0) if mode == 'air' else random.uniform(0.5, 3.0)
            brackets = [(0, 2, base * 0.9), (2, 5, base), (5, 10, base * 1.1), (10, 9999, base * 1.2)]
            freight_rates.append(CourierRate(courier_id=c, service_level=s, mode=mode, weight_brackets=brackets))

    vmi_customers = {random.randint(0, 2000): random.randint(0, NUM_WAREHOUSES - 1) for _ in range(50)}
    return orders, warehouses, freight_rates, vmi_customers

# -----------------------------
# Heurística First-Fit Decreasing (FFD)
# -----------------------------
def first_fit_decreasing(orders, warehouses, freight_rates, vmi_customers):
    start = time.time()
    assignments = []
    infeasible = []
    total_cost = 0.0

    remaining_capacity = {wid: wh.capacity_daily for wid, wh in warehouses.items()}

    # Ordenar pedidos em ordem decrescente de peso (FFD)
    sorted_orders = sorted(orders.values(), key=lambda x: -x.weight)

    for order in sorted_orders:
        feasible = False

        for wid, wh in warehouses.items():
            if order.product_type not in wh.products_allowed or order.origin_port not in wh.ports_linked:
                continue
            if remaining_capacity[wid] < order.weight:
                continue
            if order.customer_id in vmi_customers and vmi_customers[order.customer_id] != wid:
                continue

            # Selecionar custo mínimo de transporte disponível
            best_cost = float('inf')
            for fr in freight_rates:
                if fr.service_level == order.service_level:
                    cost = get_unit_transport_cost(order, fr)
                    if cost < best_cost:
                        best_cost = cost

            if best_cost == float('inf'):
                continue

            # Custo total = transporte + armazenagem
            total = best_cost + wh.storage_cost_per_unit
            total_cost += total
            remaining_capacity[wid] -= order.weight

            assignments.append({
                'order_id': order.order_id,
                'warehouse': wid,
                'courier_cost': best_cost,
                'storage_cost': wh.storage_cost_per_unit,
                'total_cost': total
            })

            feasible = True
            break  # First-Fit → parar no primeiro armazém viável

        if not feasible:
            infeasible.append(order.order_id)

    elapsed = time.time() - start
    return {
        'assignments': assignments,
        'infeasible': infeasible,
        'total_cost': round(total_cost, 2),
        'elapsed': elapsed
    }

# -----------------------------
# Exibição e salvamento de resultados
# -----------------------------
def mostrar_resultados(result):
    table = [
        ["Pedidos atribuídos", len(result['assignments'])],
        ["Pedidos inviáveis", len(result['infeasible'])],
        ["Custo total (USD)", f"${result['total_cost']:,}"],
        ["Tempo de execução", f"{result['elapsed']:.2f}s"]
    ]
    print("\nResumo dos Resultados (FFD):")
    print(tabulate(table, headers=["Métrica", "Valor"], tablefmt="grid"))

    with open(os.path.join(OUTPUT_DIR, "execution_summary.txt"), "w", encoding="utf-8") as f:
        for metric, value in table:
            f.write(f"{metric}: {value}\n")

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

    print(f"\nTotal de pedidos: {len(orders)} | Armazéns: {len(warehouses)}")

    result = first_fit_decreasing(orders, warehouses, freight_rates, vmi_customers)

    df = pd.DataFrame(result['assignments'])
    df.to_csv(os.path.join(OUTPUT_DIR, 'assignments_result.csv'), index=False)
    print(f"\n→ Resultados detalhados salvos em {OUTPUT_DIR}/assignments_result.csv")

    mostrar_resultados(result)

if __name__ == "__main__":
    run()
