# 🏭 Supply Chain Logistics Optimization – Heurísticas FFD e Greedy

Este repositório contém dois scripts em **Python** que implementam heurísticas de otimização logística aplicadas ao problema **Supply Chain Logistics Problem Dataset**, amplamente utilizado em estudos de *Supply Chain Management* e *Operational Research*.

As heurísticas — **First-Fit Decreasing (FFD)** e **Greedy Adaptativa** — têm como objetivo **minimizar o custo logístico total**, **otimizar a alocação de pedidos**, e **analisar a eficiência de diferentes estratégias heurísticas** em cadeias de suprimentos complexas.

---

## 📘 Visão Geral

As soluções simulam o processo de **distribuição logística** entre **clientes, portos, armazéns e transportadoras**, considerando:

- Custos de transporte e armazenagem  
- Capacidades limitadas de armazéns  
- Níveis de serviço (*Service Levels*)  
- Compatibilidade entre portos e produtos  

Cada script pode operar de duas formas:

- 🔹 **Com dados reais**: importando arquivos `.xlsx` ou `.csv` localizados em `./datasets/`  
- 🔹 **Com dados sintéticos**: gerados automaticamente para testes, validação e experimentos controlados  

---

## 🧠 Heurísticas Implementadas

### 🔹 `ffd_logistics_excel.py`

Implementa a heurística **First-Fit Decreasing (FFD)**, tradicional em problemas de empacotamento e alocação logística.

**Como funciona:**

1. Lê automaticamente planilhas Excel (`.xlsx`) e normaliza nomes das colunas;  
2. Ordena os pedidos em ordem **decrescente de peso**;  
3. Atribui cada pedido ao primeiro armazém viável disponível (*First-Fit*);  
4. Considera restrições de capacidade, compatibilidade de produtos e portos, e níveis de serviço;  
5. Gera relatórios detalhados de custo e desempenho.

**Entradas esperadas:**

- `datasets/Supply_chain_logistics_problem.xlsx`

**Saídas geradas:**

- `output/ffd_results.csv`: pedidos atribuídos e custos detalhados  
- `output/ffd_summary.txt`: resumo com métricas e tempo de execução

**Execução:**

```bash
python ffd_logistics_excel.py
```


---

### 🔹 `microchip_logistics_optimization.py`

Este script implementa uma **heurística gulosa adaptativa (Greedy)** avançada para otimização logística, baseada no artigo científico que motiva o projeto.

**Objetivo:**

- Atribuir pedidos a armazéns e transportadoras com objetivo de minimizar o custo total, respeitando restrições de capacidade, compatibilidade e níveis de serviço.  
- Permitir execução com datasets reais ou sintéticos, para análise comparativa das heurísticas.

**Como funciona:**

- Lê arquivos `.csv` do diretório `./datasets/` (ou gera dados sintéticos);  
- Define classes estruturadas (`Order`, `Warehouse`, `CourierRate`) que modelam o problema;  
- Aplica duas estratégias heurísticas:  
  - **Cost-First**: prioriza pedidos pelo menor custo unitário;  
  - **Weight-First**: prioriza pedidos pelo maior peso para melhor aproveitamento de capacidade;  
- Ajusta dinamicamente os custos unitários para balancear utilização dos armazéns;  
- Atualiza capacidades restantes durante a alocação de pedidos;  
- Respeita restrições de VMI para ordens vinculadas.

**Entradas esperadas:**

- `datasets/orders.csv`  
- `datasets/warehouses.csv`  
- `datasets/freight_rates.csv`  
- `datasets/vmi_customers.csv`

**Saídas geradas:**

- `output/assignments_result.csv`: detalhamento das atribuições (pedido → armazém → transportadora);  
- Log de execução exibido no console.

**Execução:**

```bash
python microchip_logistics_optimization.py
````

---

## 📂 Estruturas Internas do Código `microchip_logistics_optimization.py`

| Classe       | Descrição                                                |
|--------------|----------------------------------------------------------|
| `Order`      | Pedido (peso, produto, origem, cliente, nível de serviço)|
| `Warehouse`  | Armazém (capacidade, custo, produtos e portos aceitos)   |
| `CourierRate`| Faixas de peso, custo unitário, modal de transporte      |

---

## ❓ Suporte e Colaboração

- Reporte problemas e sugestões via issues.  
- Pull requests são bem-vindos para aprimoramentos.  
- Para dúvidas ou colaborações, contate o mantenedor.

---

## 📚 Referência

Qu, J., & Xu, P. (2025). *Greedy Algorithm-Based Optimization for Cost-Efficient Supply Chain in Outbound Microchip Logistics*. Proceedings of CMNM 2025, Fuzhou, China.

---

Este documento padroniza e organiza informações essenciais para rodar, entender e estender as heurísticas implementadas no projeto.
