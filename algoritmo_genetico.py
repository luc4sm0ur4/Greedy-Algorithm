import os
import csv
import random
import json

DATASET_DIR = "datasets"
os.makedirs(DATASET_DIR, exist_ok=True)

NUM_ORDERS = 9216
NUM_WAREHOUSES = 15
NUM_PORTS = 11
NUM_COURIERS = 9
SERVICE_LEVELS = ["DTD", "DTP", "CRF"]

# Parâmetros GA
POPULATION_SIZE = 50
GENERATIONS = 60
TOURNAMENT_SIZE = 3
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.1

class Order:
    def __init__(self, order_id, weight, product_type, origin_port, customer_id, service_level):
        self.order_id = order_id
        self.weight = weight
        self.product_type = product_type
        self.origin_port = origin_port
        self.customer_id = customer_id
        self.service_level = service_level

class Warehouse:
    def __init__(self, wid, capacity_daily, storage_cost_per_unit, products_allowed, ports_linked):
        self.wid = wid
        self.capacity_daily = capacity_daily
        self.storage_cost_per_unit = storage_cost_per_unit
        self.products_allowed = set(products_allowed)
        self.ports_linked = set(ports_linked)

class CourierRate:
    def __init__(self, courier_id, service_level, mode, weight_brackets):
        self.courier_id = courier_id
        self.service_level = service_level
        self.mode = mode
        self.weight_brackets = weight_brackets

def get_unit_transport_cost(order, warehouse, courier_rate):
    w = order.weight
    bracket = None
    for mn, mx, uc in courier_rate.weight_brackets:
        if mn <= w <= mx:
            bracket = uc
            break
    if bracket is None:
        bracket = courier_rate.weight_brackets[-1][2]
    base = bracket * w
    if order.service_level == 'DTD':
        base *= 1.25
    elif order.service_level == 'CRF':
        base *= 1.05
    if courier_rate.mode == 'air':
        base = max(base, 20.0)
    return round(base, 3)

def initialize_population(orders, warehouses, freight_rates):
    population = []
    for _ in range(POPULATION_SIZE):
        individual = {}
        for o in orders.values():
            feasible_warehouses = []
            for w in warehouses.values():
                if (o.product_type in w.products_allowed and o.origin_port in w.ports_linked):
                    feasible_warehouses.append(w.wid)
            individual[o.order_id] = random.choice(feasible_warehouses) if feasible_warehouses else None
        population.append(individual)
    return population

def evaluate_individual(individual, orders, warehouses, freight_rates):
    warehouses_capacity = {w.wid: w.capacity_daily for w in warehouses.values()}
    total_cost = 0.0
    for order_id, w_id in individual.items():
        if w_id is None:
            total_cost += 1e4
            continue
        order = orders[order_id]
        wh = warehouses[w_id]
        if warehouses_capacity[w_id] < order.weight:
            total_cost += 1e6
            continue
        costs = []
        for fr in freight_rates:
            if fr.service_level != order.service_level:
                continue
            # custo de transporte + armazenagem proporcional
            cost = get_unit_transport_cost(order, wh, fr) + wh.storage_cost_per_unit * order.weight
            costs.append(cost)
        if not costs:
            total_cost += 1e5
            continue
        min_cost = min(costs)
        total_cost += min_cost
        warehouses_capacity[w_id] -= order.weight
    return total_cost

def tournament_selection(population, fitnesses):
    selected = []
    for _ in range(2):
        contenders = random.sample(list(zip(population, fitnesses)), TOURNAMENT_SIZE)
        winner = min(contenders, key=lambda x: x[1])
        selected.append(winner[0])
    return selected[0], selected[1]

def crossover(parent1, parent2):
    if random.random() > CROSSOVER_RATE:
        return parent1.copy(), parent2.copy()
    point = random.randint(1, len(parent1) - 1)
    keys = list(parent1.keys())
    child1, child2 = {}, {}
    for i, key in enumerate(keys):
        child1[key] = parent1[key] if i < point else parent2[key]
        child2[key] = parent2[key] if i < point else parent1[key]
    return child1, child2

def mutation(individual, warehouses, orders):
    for order_id in individual.keys():
        if random.random() < MUTATION_RATE:
            o = orders[order_id]
            feasible_warehouses = []
            for w in warehouses.values():
                if o.product_type in w.products_allowed and o.origin_port in w.ports_linked:
                    feasible_warehouses.append(w.wid)
            if feasible_warehouses:
                individual[order_id] = random.choice(feasible_warehouses)
    return individual

def genetic_algorithm(orders, warehouses, freight_rates):
    population = initialize_population(orders, warehouses, freight_rates)
    best_solution, best_cost = None, float('inf')
    for gen in range(GENERATIONS):
        fitnesses = [evaluate_individual(ind, orders, warehouses, freight_rates) for ind in population]
        new_population = []
        while len(new_population) < POPULATION_SIZE:
            parent1, parent2 = tournament_selection(population, fitnesses)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutation(child1, warehouses, orders)
            child2 = mutation(child2, warehouses, orders)
            new_population.extend([child1, child2])
        population = new_population[:POPULATION_SIZE]
        gen_best_cost = min(fitnesses)
        if gen_best_cost < best_cost:
            best_cost = gen_best_cost
            best_solution = population[fitnesses.index(gen_best_cost)]
        print(f"Geração {gen+1}, Melhor custo: {best_cost:.2f}")
    return best_solution, best_cost

# Funções utilitárias para carregar dados dos CSVs

def load_orders_csv(path: str):
    import pandas as pd
    df = pd.read_csv(path)
    return {int(row['order_id']): Order(
        int(row['order_id']), float(row['weight']), int(row['product_type']),
        int(row['origin_port']), int(row['customer_id']), str(row['service_level']).strip()
    ) for _, row in df.iterrows()}

def load_warehouses_csv(path: str):
    import pandas as pd
    df = pd.read_csv(path)
    return {int(row['wid']): Warehouse(
        int(row['wid']), int(row['capacity_daily']), float(row['storage_cost_per_unit']),
        json.loads(row['products_allowed']), json.loads(row['ports_linked'])
    ) for _, row in df.iterrows()}

def load_freight_csv(path: str):
    import pandas as pd
    df = pd.read_csv(path)
    freight = []
    for (cid, sl, mode), group in df.groupby(['courier_id', 'service_level', 'mode']):
        brackets = [(float(r['min_weight']), float(r['max_weight']), float(r['unit_cost'])) for _, r in group.iterrows()]
        freight.append(CourierRate(int(cid), str(sl), str(mode), brackets))
    return freight

def generate_orders_csv():
    with open(os.path.join(DATASET_DIR, 'orders.csv'), mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['order_id', 'weight', 'product_type', 'origin_port', 'customer_id', 'service_level'])
        for oid in range(NUM_ORDERS):
            weight = round(random.expovariate(1/5.0), 3) + 0.1
            product_type = random.randint(0, 9)
            origin_port = random.randint(0, NUM_PORTS - 1)
            customer_id = random.randint(0, 2000)
            service_level = random.choices(SERVICE_LEVELS, weights=[0.6, 0.3, 0.1])[0]
            writer.writerow([oid, weight, product_type, origin_port, customer_id, service_level])
    print("orders.csv criado.")

def generate_warehouses_csv():
    with open(os.path.join(DATASET_DIR, 'warehouses.csv'), mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['wid', 'capacity_daily', 'storage_cost_per_unit', 'products_allowed', 'ports_linked'])
        for wid in range(NUM_WAREHOUSES):
            capacity_daily = random.randint(400, 900)
            storage_cost = round(random.uniform(0.5, 2.0), 2)
            products_allowed = sorted(random.sample(range(10), random.randint(4, 10)))
            ports_linked = sorted(random.sample(range(NUM_PORTS), random.randint(3, 7)))
            writer.writerow([wid, capacity_daily, storage_cost, json.dumps(products_allowed), json.dumps(ports_linked)])
    print("warehouses.csv criado.")

def generate_freight_rates_csv():
    with open(os.path.join(DATASET_DIR, 'freight_rates.csv'), mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['courier_id', 'service_level', 'mode', 'min_weight', 'max_weight', 'unit_cost'])
        for cid in range(NUM_COURIERS):
            for sl in SERVICE_LEVELS:
                mode = random.choice(['air', 'road'])
                base_cost = random.uniform(3.0, 12.0) if mode == 'air' else random.uniform(0.5, 3.0)
                ranges = [(0,2), (2,5), (5,10), (10,9999)]
                for mn, mx in ranges:
                    unit_cost = round(base_cost * random.uniform(0.9, 1.1), 3)
                    writer.writerow([cid, sl, mode, mn, mx, unit_cost])
    print("freight_rates.csv criado.")

def generate_vmi_customers_csv():
    with open(os.path.join(DATASET_DIR, 'vmi_customers.csv'), mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['customer_id', 'warehouse_id'])
        for _ in range(50):
            customer_id = random.randint(0, 2000)
            warehouse_id = random.randint(0, NUM_WAREHOUSES - 1)
            writer.writerow([customer_id, warehouse_id])
    print("vmi_customers.csv criado.")

def generate_all_csvs():
    generate_orders_csv()
    generate_warehouses_csv()
    generate_freight_rates_csv()
    generate_vmi_customers_csv()
    print("Todos os datasets foram criados em ./datasets.")

if __name__ == "__main__":
    generate_all_csvs()