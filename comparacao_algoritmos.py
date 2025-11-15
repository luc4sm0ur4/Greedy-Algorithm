import time
import pandas as pd
from algoritmo_genetico import genetic_algorithm, load_orders_csv, load_warehouses_csv, load_freight_csv
from microchip_logistics_optimization import greedy_assign  # Heurística gulosa adaptativa
from microchip_logistics_optimization_ffd import first_fit_decreasing  # Heurística FFD


def main():
    orders = load_orders_csv("datasets/orders.csv")
    warehouses = load_warehouses_csv("datasets/warehouses.csv")
    freight_rates = load_freight_csv("datasets/freight_rates.csv")
    vmi_customers = {}  # Opcional: implementar carregamento de vmi_customers em CSV se desejar

    resultados = []

    # Algoritmo Genético
    start = time.time()
    best_solution, best_cost = genetic_algorithm(orders, warehouses, freight_rates)
    elapsed = time.time() - start
    resultados.append({
        "Método": "Algoritmo Genético",
        "Custo Total (USD)": best_cost,
        "Tempo (s)": elapsed
    })
    print(f"[GA] Custo: ${best_cost:,.2f}, Tempo: {elapsed:.2f}s")

    # Heurística Gulosa Adaptativa - Cost-first
    start = time.time()
    res_cost = greedy_assign(orders, warehouses, freight_rates, vmi_customers, strategy="cost-first")
    elapsed = time.time() - start
    resultados.append({
        "Método": "Guloso Cost-first",
        "Custo Total (USD)": res_cost["total_cost"],
        "Tempo (s)": elapsed
    })
    print(f"[Guloso Cost-first] Custo: ${res_cost['total_cost']:,.2f}, Tempo: {elapsed:.2f}s")

    # Heurística Gulosa Adaptativa - Weight-first
    start = time.time()
    res_weight = greedy_assign(orders, warehouses, freight_rates, vmi_customers, strategy="weight-first")
    elapsed = time.time() - start
    resultados.append({
        "Método": "Guloso Weight-first",
        "Custo Total (USD)": res_weight["total_cost"],
        "Tempo (s)": elapsed
    })
    print(f"[Guloso Weight-first] Custo: ${res_weight['total_cost']:,.2f}, Tempo: {elapsed:.2f}s")

    # First-Fit Decreasing (FFD)
    start = time.time()
    result_ffd = first_fit_decreasing(orders, warehouses, freight_rates, vmi_customers)
    elapsed = time.time() - start
    resultados.append({
        "Método": "First-Fit Decreasing (FFD)",
        "Custo Total (USD)": result_ffd["total_cost"],
        "Tempo (s)": elapsed
    })
    print(f"[FFD] Custo: ${result_ffd['total_cost']:,.2f}, Tempo: {elapsed:.2f}s")

    # Exibir resumo dos resultados
    df = pd.DataFrame(resultados)
    print("\nComparação dos Algoritmos:")
    print(df.to_markdown(index=False))


if __name__ == "__main__":
    main()
