# XGBoost para Rank:pairwise

## 10.1 Conceito de ranking

Diferente de regressão e classificação, aqui o foco não é prever valores absolutos ou classes, mas sim **ordens relativas**.

Exemplo:

* Em um buscador, dado o termo “restaurante japonês”, queremos que os mais relevantes apareçam no topo.
* Em trading, podemos querer ranquear ativos pelo retorno esperado, mesmo sem prever o valor exato.

👉 O modelo aprende **preferências**: que A deve estar acima de B, B acima de C, etc.

***

## 10.2 Formação de pares

O XGBoost implementa o **rank:pairwise**.

* Para cada **query** (conjunto de documentos/itens comparáveis), forma-se pares (i, j).
* Cada par indica que o item i deveria estar acima de j segundo o rótulo.
* O modelo então ajusta os **scores preditos** para que:

$$
\hat{y}_i > \hat{y}_j
$$

***

## 10.3 Função objetivo em pares

A função de perda é derivada da **log-loss binária**, aplicada a diferenças de scores:

$$
\ell(i, j) = \log \Big( 1 + e^{-(\hat{y}_i - \hat{y}_j)} \Big)
$$

Essa loss é pequena quando:

$$
\hat{y}_i \gg \hat{y}_j
$$

(ordem correta) e grande quando:

$$
\hat{y}_i \leq \hat{y}_j
$$

(ordem errada).

***

## 10.4 Como o XGBoost “olha em pares” (mas treina por itens)

*   Para cada **par (i, j)**, calcula-se um gradiente e hessiano em relação à diferença de scores:

    $$
    \Delta_{ij} = f_i - f_j
    $$
* O item “vencedor” (label maior) recebe gradiente negativo (quer aumentar seu score) e o perdedor recebe gradiente positivo (quer reduzir seu score).
* O hessiano é igual para ambos no par.
* No fim, cada **item** acumula ((g\_i, h\_i)) de todos os pares em que participa.
* A árvore de decisão é treinada sobre **itens individuais**, usando esses ((g\_i, h\_i)) agregados, exatamente como faria numa regressão ou classificação.

👉 Ou seja: os **pares só existem no treino** para gerar gradientes. No momento de construir a árvore, o algoritmo ainda enxerga **itens** com gradientes/hessianos associados.

***

## 10.5 Exemplo prático – 2 features e 3 boosts

### Setup

Tabela com labels e features:

| Item | Relevância (label) | Momentum | Vol |
| ---- | ------------------ | -------- | --- |
| A    | 3                  | 0.6      | 0.1 |
| B    | 2                  | 0.2      | 0.4 |
| C    | 1                  | -0.4     | 0.7 |

* Ordem desejada: A ≻ B ≻ C

Hiperparâmetros:

$$
\lambda = 1.0, \quad \eta = 0.3
$$

Inicialização:

$$
f_A = f_B = f_C = 0
$$

***

### Boost 1 (Árvore 1)

**Gradientes iniciais** (todos pares com (f=0)):

* A: (g\_A=-1.0, h\_A=0.5)
* B: (g\_B=0.0, h\_B=0.5)
* C: (g\_C=+1.0, h\_C=0.5)

**Split**: momentum > 0 (A,B) vs ≤ 0 (C).

* Esquerda: (G=-1.0, H=1.0) ⇒ (w\_L=+0.5)
* Direita: (G=+1.0, H=0.5) ⇒ (w\_R=-0.6667)

**Scores atualizados**:

* A, B: +0.15
* C: -0.20

***

### Boost 2 (Árvore 2)

Recalcula-se (g,h) com os novos scores.

* A: (g\_A=-0.9134, h\_A=0.4925)
* B: (g\_B=+0.0866, h\_B=0.4925)
* C: (g\_C=+0.8268, h\_C=0.4950)

**Split**: vol ≤ 0.3 (A) vs > 0.3 (B,C).

* A: (w\_L=+0.6120)
* (B,C): (w\_R=-0.4619)

**Scores atualizados**:

* A: 0.3336
* B: 0.0114
* C: -0.3386

***

### Boost 3 (Árvore 3)

Novos gradientes:

* A: (g\_A=-0.7582, h\_A=0.4674)
* B: (g\_B=+0.0068, h\_B=0.4861)
* C: (g\_C=+0.7514, h\_C=0.4663)

**Split**: momentum > 0 (A,B) vs ≤ 0 (C).

* (A,B): (w\_L=+0.3846)
* (C): (w\_R=-0.5125)

**Scores finais**:

* A: 0.4490
* B: 0.1268
* C: -0.4923

**Ranking obtido**: A ≻ B ≻ C (correto).

***

## 10.6 O que a árvore realmente enxerga

* A árvore não vê “+1/−1” explícitos.
* Ela usa **somatórios** (G=\sum g\_i), (H=\sum h\_i) nos nós.
* Testa splits e escolhe o de maior **gain**:

$$
Gain = \tfrac{1}{2}\Bigg( \frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda} - \frac{(G_L+G_R)^2}{H_L+H_R+\lambda} \Bigg) - \gamma
$$

* O peso da folha é:

$$
w^* = -\frac{G}{H+\lambda}
$$

* A atualização do score é:

$$
f \leftarrow f + \eta \cdot w^*
$$

***

## 10.7 Como funciona o **predict**

* Na inferência, **não existem pares**.
* Para cada item, percorre-se **todas as árvores** e coleta-se o peso da folha onde ele cai.
* A predição final é:

$$
f(x) = \sum_{t=1}^{T} \eta \cdot w^{(t)}_{\text{folha}(x)}
$$

* O ranking é obtido **ordenando os itens por seus scores finais** dentro de cada query.

***

✅ **Resumo do Capítulo 10:**

* O rank:pairwise transforma ranking em problema de pares.
* A loss é baseada em log-loss aplicada à diferença de scores.
* Cada par gera gradientes e hessianos, que são agregados por item.
* Árvores são treinadas como em regressão/classificação, mas com (g,h) vindos dos pares.
* No predict, não há pares: apenas scores somados e ordenação.
* Exemplo detalhado com 2 features e 3 boosts mostra o processo completo até o ranking final.
