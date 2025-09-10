# Árvores de Decisão

## 2.1 Estrutura de uma árvore

Uma **árvore de decisão** é um modelo que organiza regras em formato de árvore binária, formada por nós e folhas:

* **Nó raiz (root):** onde todos os dados começam.
* **Nós internos (splits):** pontos de decisão baseados em uma condição (ex.: “metragem > 100?”).
* **Folhas (leaves):** cada folha contém a predição final para os exemplos que chegaram até ali.

👉 Pense numa árvore como uma sequência de **perguntas de “sim ou não”** que vai separando os dados até chegar numa resposta.

***

## 2.2 Como a árvore decide os splits

O algoritmo precisa escolher, a cada divisão, **qual feature** e **qual valor de corte (threshold)** usar.

A escolha é feita testando possíveis divisões e medindo qual delas gera grupos mais **homogêneos** em relação ao alvo (target).

* Em **regressão**, homogêneo significa: valores do target com **baixa variância**.
* Em **classificação**, homogêneo significa: maioria das observações pertencendo à **mesma classe**.
* Em **rankeamento**, homogêneo significa: divisões que ajudam a respeitar a **ordem relativa** entre itens.

***

## 2.3 Critérios de parada

Uma árvore pode crescer até separar perfeitamente todos os pontos (memorizar o treino), mas isso causa **overfitting**.

Por isso, usamos limites para o crescimento:

* **max\_depth:** profundidade máxima da árvore.
* **min\_child\_weight:** número mínimo de exemplos (ou peso) em cada nó.
* **gamma:** ganho mínimo necessário para justificar um novo split.

Esses limites são **hiperparâmetros** fundamentais para controlar a complexidade.

***

## 2.4 Exemplo prático – Regressão

Vamos prever o preço de casas a partir da metragem:

| Metragem (m²) | Preço (mil R$) |
| ------------- | -------------- |
| 50            | 200            |
| 60            | 240            |
| 70            | 250            |
| 120           | 500            |
| 130           | 550            |
| 150           | 600            |

***

### Passo 1 – Sem divisão (nó raiz)

Previsão = média global =

$$
\bar{y} = \frac{200+240+250+500+550+600}{6} = 390
$$

Erro (SSE):

$$
SSE = (200-390)^2 + (240-390)^2 + (250-390)^2 + (500-390)^2 + (550-390)^2 + (600-390)^2
$$

$$
SSE = 160.000
$$

📌 **Erro inicial = 160.000**

***

### Passo 2 – Split “metragem ≤ 100 vs > 100”

* Grupo 1 (≤100): \[200, 240, 250], média = 230\
  Erro = (200-230)² + (240-230)² + (250-230)² = 1.400
* Grupo 2 (>100): \[500, 550, 600], média = 550\
  Erro = (500-550)² + (550-550)² + (600-550)² = 5.000

$$
SSE_{split} = 1.400 + 5.000 = 6.400
$$

📌 **Erro caiu de 160.000 para 6.400 → redução de 96%**

***

### Passo 3 – Refinando o grupo >100

Dividindo “≤130” vs “>130”:

* Grupo 2a (120, 130): média = 525 → erro = 1.250
* Grupo 2b (150): média = 600 → erro = 0

Erro total = 1.400 (grupo ≤100) + 1.250 + 0 = **2.650**

📌 A árvore vai refinando os splits para reduzir progressivamente o erro.

***

## 2.5 Intuição em classificação

Quando o problema é classificação, a ideia é a mesma: criar grupos homogêneos.

Mas aqui homogeneidade significa **pureza de classes**.

Exemplo (spam vs não spam):

* Nó A: 50 e-mails → 48 spam, 2 não spam → quase puro → homogêneo.
* Nó B: 50 e-mails → 25 spam, 25 não spam → completamente misturado → heterogêneo.

Para medir essa pureza, usamos métricas como:

* **Índice de Gini** (quanto menor, mais puro).
* **Entropia** (quanto menor, mais previsível o nó).

📌 Não precisamos dos cálculos ainda — a ideia é que a árvore escolhe splits que tornam os nós mais “puros”.

***

## 2.6 Intuição em rankeamento

No caso de **rankeamento**, o objetivo da árvore não é separar classes ou prever valores absolutos, mas **aprender preferências**: dado um conjunto (query), quais itens devem ficar acima de outros?

* O modelo cria pares de comparação (doc A deve estar acima de doc B).
* A árvore é construída para reduzir as inversões (quando B fica acima de A por engano).
* A qualidade de um split é medida por métricas como **NDCG** ou versões diferenciáveis (LambdaRank).

📌 Aqui não entraremos em contas ainda — apenas guardamos a intuição: no ranking, a homogeneidade é “respeitar a ordem correta”.

***

## 2.7 Limitações de árvores simples

Apesar de fáceis de entender, árvores sozinhas têm problemas:

* **Alta variância:** pequenas mudanças nos dados podem gerar árvores muito diferentes.
* **Overfitting:** árvores profundas podem memorizar o treino.
* **Pouco poder preditivo isoladamente:** sozinhas raramente batem modelos mais sofisticados.

👉 É por isso que usamos **ensembles** (conjuntos de árvores). O XGBoost é um caso de **boosting de árvores**, que corrige as fraquezas de cada árvore individual.

***

## ✅ Resumo do Capítulo 2

* Árvores de decisão dividem dados em regras “sim/não” até chegar a uma predição.
* A qualidade dos splits é medida por homogeneidade: variância (regressão), pureza de classe (classificação), respeito à ordem (ranking).
* Splits reduzem o erro progressivamente, até que critérios de parada sejam atingidos.
* Árvores isoladas são fracas, mas quando combinadas formam modelos poderosos — e é aí que entra o **boosting**.
