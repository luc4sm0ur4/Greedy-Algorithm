"""
Script: ffd_logistics.py
-----------------------------------
Este script implementa a heurística **First-Fit Decreasing (FFD)** aplicada ao
problema de alocação logística, utilizando o dataset "Supply Chain Logistics Problem Dataset".

Objetivo:
---------
O algoritmo busca minimizar o custo total de transporte e armazenamento,
atribuindo pedidos a armazéns disponíveis conforme:
- peso dos pedidos,
- compatibilidade de produtos e portos,
- capacidade diária dos armazéns,
- e níveis de serviço logístico.

Funcionamento:
--------------
1. Lê automaticamente todas as planilhas do arquivo Excel localizado em ./datasets/
   (detectando nomes e colunas mesmo que variem de grafia).
2. Mantém identificadores alfa-numéricos originais (ex.: PORT09, WH01, PROD05).
3. Executa o algoritmo FFD com checagem de compatibilidade relaxada ou flexível.
4. Exibe diagnósticos com amostras de produtos e portos encontrados.
5. Gera relatórios automáticos com os resultados e tempos de execução.

Arquivos de entrada esperados:
------------------------------
- ./datasets/Supply_chain_logistics_problem.xlsx  (ou outro arquivo Excel compatível)

Saídas geradas:
---------------
- ./output/ffd_results.csv   → lista completa de pedidos atribuídos com custos
- ./output/ffd_summary.txt   → resumo da execução (custos, tempo, inviáveis)
- log de diagnóstico impresso no terminal

Dependências:
-------------
Instale as bibliotecas necessárias com:
    pip install pandas numpy pulp tabulate

Uso:
----
Execute no terminal:
    python ffd_logistics.py

Este código foi desenvolvido para fins acadêmicos e de pesquisa em heurísticas de
otimização logística e análise de desempenho em Supply Chain.
"""

import os
import time
import pandas as pd
from typing import Dict, List

OUTPUT_DIR = 'output'
DATASET_DIR = 'datasets'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

STRICT_COMPATIBILITY = False  # False = modo flexível (recomendado para Figshare)

# -----------------------------
# Funções auxiliares
# -----------------------------

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip().lower().replace(' ', '').replace('_', '') for c in df.columns]
    return df

def read_excel_all(path: str) -> Dict[str, pd.DataFrame]:
    xls = pd.ExcelFile(path)
    sheets = {}
    for name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=name)
        sheets[name.strip().lower().replace(' ', '')] = normalize_cols(df)
    return sheets

def find_col(possibilities: List[str], columns: List[str]) -> str:
    for cand in possibilities:
        for col in columns:
            if cand in col:
                return col
    raise KeyError(f"Nenhuma coluna correspondente encontrada para {possibilities}")

# -----------------------------
# Parsing das planilhas
# -----------------------------

def parse_orders(df: pd.DataFrame) -> Dict[str, dict]:
    cols = list(df.columns)
    id_col = find_col(['orderid', 'order', 'id'], cols)
    weight_col = find_col(['weight', 'peso'], cols)
    prod_col = find_col(['producttype', 'product'], cols)
    port_col = find_col(['originport', 'port', 'porto'], cols)
    cust_col = find_col(['customerid', 'customer', 'cliente'], cols)
    svc_col = find_col(['servicelevel', 'service', 'servico'], cols)

    print("→ Colunas detectadas em OrderList:")
    print(f"  ID: {id_col}, Weight: {weight_col}, Product: {prod_col}, Port: {port_col}, Customer: {cust_col}, Service: {svc_col}")

    orders = {}
    for _, r in df.iterrows():
        try:
            oid = str(r[id_col]).strip()
            orders[oid] = {
                'order_id': oid,
                'weight': float(r[weight_col]),
                'product_type': str(r[prod_col]).strip(),
                'origin_port': str(r[port_col]).strip(),
                'customer_id': str(r[cust_col]).strip(),
                'service_level': str(r[svc_col]).strip()
            }
        except Exception:
            continue
    return orders

def parse_warehouses(whcosts_df, whcaps_df, products_df, plantports_df) -> Dict[str, dict]:
    warehouses = {}

    def ensure_wh(pid):
        if pid not in warehouses:
            warehouses[pid] = {
                'wid': pid,
                'storage_cost_per_unit': 0.0,
                'capacity_daily': 0,
                'products_allowed': set(),
                'ports_linked': set()
            }

    for _, r in whcosts_df.iterrows():
        pid = str(r.iloc[0]).strip()
        ensure_wh(pid)
        try:
            warehouses[pid]['storage_cost_per_unit'] = float(r.iloc[1])
        except Exception:
            warehouses[pid]['storage_cost_per_unit'] = 0.0

    for _, r in whcaps_df.iterrows():
        pid = str(r.iloc[0]).strip()
        ensure_wh(pid)
        try:
            warehouses[pid]['capacity_daily'] = int(r.iloc[1])
        except Exception:
            warehouses[pid]['capacity_daily'] = 0

    for _, r in products_df.iterrows():
        pid = str(r.iloc[0]).strip()
        prod = str(r.iloc[1]).strip()
        ensure_wh(pid)
        warehouses[pid]['products_allowed'].add(prod)

    for _, r in plantports_df.iterrows():
        pid = str(r.iloc[0]).strip()
        port = str(r.iloc[1]).strip()
        ensure_wh(pid)
        warehouses[pid]['ports_linked'].add(port)

    print(f"→ Armazéns carregados: {len(warehouses)}")
    return warehouses

def parse_freight_rates(df: pd.DataFrame) -> List[dict]:
    freight = []
    skipped = 0
    for _, r in df.iterrows():
        try:
            courier_id = str(r.iloc[0]).strip()
            service_level = str(r.iloc[1]).strip()
            mode = str(r.iloc[2]).strip()
            min_weight = float(r.iloc[3])
            max_weight = float(r.iloc[4])
            unit_cost = float(r.iloc[5])
            freight.append({
                'courier_id': courier_id,
                'service_level': service_level,
                'mode': mode,
                'min_weight': min_weight,
                'max_weight': max_weight,
                'unit_cost': unit_cost
            })
        except (ValueError, TypeError):
            skipped += 1
            continue

    if skipped > 0:
        print(f"[Aviso] {skipped} linhas ignoradas em FreightRates (dados inválidos).")

    grouped = {}
    for f in freight:
        key = (f['courier_id'], f['service_level'], f['mode'])
        grouped.setdefault(key, []).append((f['min_weight'], f['max_weight'], f['unit_cost']))
    return [{'courier_id': k[0], 'service_level': k[1], 'mode': k[2], 'brackets': v} for k, v in grouped.items()]

def parse_vmi(df: pd.DataFrame) -> Dict[str, str]:
    vmi = {}
    for _, r in df.iterrows():
        try:
            vmi[str(r.iloc[0]).strip()] = str(r.iloc[1]).strip()
        except Exception:
            continue
    return vmi

# -----------------------------
# Cálculo de custo e FFD
# -----------------------------

def compute_transport_cost(weight: float, service: str, freight_list: List[dict]) -> float:
    best_cost = float('inf')
    for rec in freight_list:
        if rec['service_level'] != service:
            continue
        for mn, mx, unit in rec['brackets']:
            if mn <= weight <= mx:
                cost = unit * weight
                if rec['mode'] == 'air':
                    cost = max(cost, 20.0)
                best_cost = min(best_cost, cost)
                break
    return round(best_cost, 3) if best_cost != float('inf') else float('inf')

def first_fit_decreasing(orders, warehouses, freight, vmi):
    start = time.time()
    orders_sorted = sorted(orders.values(), key=lambda x: -x['weight'])
    remaining = {wid: warehouses[wid]['capacity_daily'] for wid in warehouses}

    assignments, infeasible = [], []
    for o in orders_sorted:
        cust = o['customer_id']
        target_whs = [vmi[cust]] if cust in vmi else list(warehouses.keys())
        assigned = False
        for w in target_whs:
            if w not in warehouses:
                continue
            wh = warehouses[w]

            # Compatibilidade flexível
            port_match = any(o['origin_port'].lower() in p.lower() or p.lower() in o['origin_port'].lower() for p in wh['ports_linked'])
            prod_match = any(o['product_type'].lower() in p.lower() or p.lower() in o['product_type'].lower() for p in wh['products_allowed'])

            if STRICT_COMPATIBILITY:
                if o['product_type'] not in wh['products_allowed'] or o['origin_port'] not in wh['ports_linked']:
                    continue
            else:
                if not port_match or not prod_match:
                    continue

            if remaining[w] <= 0:
                continue

            cost_trans = compute_transport_cost(o['weight'], o['service_level'], freight)
            if cost_trans == float('inf'):
                continue

            cost_store = wh['storage_cost_per_unit']
            assignments.append({
                'order_id': o['order_id'],
                'warehouse': w,
                'transport_cost': cost_trans,
                'storage_cost': cost_store,
                'total_cost': cost_trans + cost_store
            })
            remaining[w] -= 1
            assigned = True
            break
        if not assigned:
            infeasible.append(o['order_id'])

    elapsed = time.time() - start
    total_cost = sum(a['total_cost'] for a in assignments)
    return {'assignments': assignments, 'infeasible': infeasible, 'elapsed': elapsed, 'total_cost': total_cost}

# -----------------------------
# Execução principal
# -----------------------------

def main():
    excel_files = [f for f in os.listdir(DATASET_DIR) if f.lower().endswith(('.xlsx', '.xls'))]
    if not excel_files:
        raise FileNotFoundError('Nenhum arquivo Excel encontrado em ./datasets/')
    excel_path = os.path.join(DATASET_DIR, excel_files[0])

    print(f'Lendo dataset: {excel_path}')
    sheets = read_excel_all(excel_path)

    def get_sheet(name_part):
        for k, v in sheets.items():
            if name_part in k:
                return v
        raise KeyError(f"Planilha contendo '{name_part}' não encontrada.")

    orders = parse_orders(get_sheet('order'))
    warehouses = parse_warehouses(get_sheet('whcost'), get_sheet('whcap'), get_sheet('product'), get_sheet('plant'))
    freight = parse_freight_rates(get_sheet('freight'))
    vmi = parse_vmi(get_sheet('vmi'))

    print(f'Pedidos: {len(orders)}, Armazéns: {len(warehouses)}')

    # Diagnóstico de compatibilidade
    print("\n[DEBUG] Amostras de identificação para análise de compatibilidade:")
    order_ports = sorted({o['origin_port'] for o in orders.values()})
    order_prods = sorted({o['product_type'] for o in orders.values()})
    plant_ports = set()
    plant_prods = set()
    for w in warehouses.values():
        plant_ports.update(w['ports_linked'])
        plant_prods.update(w['products_allowed'])

    print(f"  → Portos em pedidos (amostra 10): {order_ports[:10]}")
    print(f"  → Portos em plantas (amostra 10): {list(plant_ports)[:10]}")
    print(f"  → Produtos em pedidos (amostra 10): {order_prods[:10]}")
    print(f"  → Produtos em plantas (amostra 10): {list(plant_prods)[:10]}\n")

    result = first_fit_decreasing(orders, warehouses, freight, vmi)

    df = pd.DataFrame(result['assignments'])
    df.to_csv(os.path.join(OUTPUT_DIR, 'ffd_results.csv'), index=False)

    with open(os.path.join(OUTPUT_DIR, 'ffd_summary.txt'), 'w', encoding='utf-8') as f:
        f.write(f"Total de pedidos: {len(orders)}\n")
        f.write(f"Pedidos atribuídos: {len(result['assignments'])}\n")
        f.write(f"Inviáveis: {len(result['infeasible'])}\n")
        f.write(f"Custo total: {result['total_cost']:.2f}\n")
        f.write(f"Tempo: {result['elapsed']:.3f}s\n")

    print(f"Execução concluída. Resultados salvos em {OUTPUT_DIR}/")

if __name__ == '__main__':
    main()
