# 🏭 Supply Chain Logistics Optimization – Heurísticas FFD e Greedy

Este repositório contém dois scripts Python que implementam heurísticas de otimização logística
baseadas no problema **"Supply Chain Logistics Problem Dataset"**.  
Ambos foram desenvolvidos para estudos de heurísticas gulosa e *First-Fit Decreasing (FFD)* aplicadas à alocação de pedidos, armazéns e custos de transporte.

---

## 📘 Visão Geral

Os códigos simulam e/ou analisam o processo de distribuição logística entre clientes, portos, armazéns e transportadoras, buscando **minimizar o custo total** e **maximizar a eficiência operacional**.

Cada script pode operar com **dados reais** (fornecidos via arquivos externos) ou **dados sintéticos** (gerados automaticamente para testes e validações).

---

## 🧠 Descrição dos Códigos

### 🔹 `ffd_logistics_excel.py`

Implementa a heurística **First-Fit Decreasing (FFD)** com foco em datasets Excel (`.xlsx`).

**Principais recursos:**
- Lê automaticamente todas as planilhas do Excel (ordens, fretes, portos, produtos, etc.).
- Detecta nomes de colunas e planilhas mesmo com variações (case-insensitive e espaços).
- Mantém identificadores alfa-numéricos originais (`PORT09`, `WH01`, `PROD05`, etc.).
- Executa o FFD priorizando **ordens mais pesadas** e respeitando **capacidades e compatibilidades**.
- Gera relatórios automáticos com estatísticas e custos totais.

**Entradas esperadas:**
- `datasets/Supply_chain_logistics_problem.xlsx`

**Saídas geradas:**
- `output/ffd_results.csv` — lista de pedidos atribuídos com custos.
- `output/ffd_summary.txt` — resumo da execução (custos, tempo, inviáveis).
- Diagnósticos impressos no terminal com amostras de portos e produtos.

**Execução:**
```bash
python ffd_logistics_excel.py
