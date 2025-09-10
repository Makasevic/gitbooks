# XGBoost para Classificação

## 9.1 Função de perda em classificação binária

O caso mais comum é a **log-loss** (entropia cruzada):

$$
\ell(y_i, \hat{p}_i) = - \Big( y_i \log(\hat{p}_i) + (1 - y_i) \log(1 - \hat{p}_i) \Big)
$$

**Definições**

$$
y_i \in \{0,1\} \quad \text{(rótulo verdadeiro)}
$$

$$
\hat{p}_i = \text{probabilidade prevista de } y_i = 1
$$

No boosting, trabalhamos em termos de **logits**:

$$
\hat{y}_i = \text{logit associado à observação}
$$

A probabilidade é obtida aplicando a **sigmoid**:

$$
\hat{p}_i = \sigma(\hat{y}_i) = \frac{1}{1 + e^{-\hat{y}_i}}
$$

***

## 9.2 Gradientes e Hessianos

Para a log-loss, o gradiente e o hessiano em relação ao logit são definidos como:

$$
g_i = \hat{p}_i - y_i
$$

$$
h_i = \hat{p}_i (1 - \hat{p}_i)
$$

👉 Diferente da regressão, aqui o hessiano **não é constante** — ele depende da incerteza da probabilidade.

***

## 9.3 Exemplo numérico – passo a passo

Dataset fictício: prever se o ativo deve ser **comprado (1)** ou **não comprado (0)** com base em uma feature (momentum).

| Ativo | Classe real (y) | Feature (momentum) |
| ----- | --------------- | ------------------ |
| A     | 1               | 0.5                |
| B     | 0               | 0.2                |
| C     | 1               | -0.3               |

***

### Passo 1 – Score inicial

No início, o XGBoost define o **logit inicial** a partir da taxa de positivos:

$$
\hat{y}^{(0)} = \log \frac{\text{positivos}}{\text{negativos}} = \log \frac{2}{1} \approx 0.693
$$

As probabilidades iniciais são:

$$
\hat{p}_i^{(0)} = \sigma(0.693) \approx 0.667
$$

***

### Passo 2 – Gradientes e Hessianos

Para cada amostra:

$$
g_i = \hat{p}_i - y_i, \qquad h_i = \hat{p}_i (1 - \hat{p}_i)
$$

*   **A**

    $$
    g_A = 0.667 - 1 = -0.333, \quad h_A = 0.222
    $$
*   **B**

    $$
    g_B = 0.667 - 0 = 0.667, \quad h_B = 0.222
    $$
*   **C**

    $$
    g_C = 0.667 - 1 = -0.333, \quad h_C = 0.222
    $$

***

### Passo 3 – Treinando a primeira árvore

Suponha que o split seja _momentum > 0 vs ≤ 0_.

*   **Grupo 1 (A, B):**

    $$
    G = 0.334, \quad H = 0.444
    $$
*   **Grupo 2 (C):**

    $$
    G = -0.333, \quad H = 0.222
    $$

**Peso ótimo de cada folha:**

$$
w^* = - \frac{G}{H + \lambda}
$$

**Assumindo:**

$$
\lambda = 1
$$

*   **Grupo 1**

    $$
    w^* = -\frac{0.334}{0.444+1} \approx -0.231
    $$
*   **Grupo 2**

    $$
    w^* = -\frac{-0.333}{0.222+1} \approx 0.273
    $$

***

### Passo 4 – Atualizando predições

**Learning rate:**

$$
\eta = 0.5
$$

*   **A**

    $$
    \hat{y}^{(1)} = 0.693 + 0.5 \times (-0.231) \approx 0.577
    $$
*   **B**

    $$
    \hat{y}^{(1)} = 0.693 + 0.5 \times (-0.231) \approx 0.577
    $$
*   **C**

    $$
    \hat{y}^{(1)} = 0.693 + 0.5 \times 0.273 \approx 0.830
    $$

Convertendo para probabilidades:

*   **A**

    $$
    \hat{p}^{(1)} = \sigma(0.577) \approx 0.641
    $$
*   **B**

    $$
    \hat{p}^{(1)} = \sigma(0.577) \approx 0.641
    $$
*   **C**

    $$
    \hat{p}^{(1)} = \sigma(0.830) \approx 0.697
    $$

***

## 9.4 Interpretação dos outputs

* O output interno do XGBoost em classificação é um **logit**.
* Ele é transformado em probabilidade pela **sigmoid**.
* A decisão final depende de um **threshold** (por padrão, 0.5).

📌 Importante: se o dataset for **desbalanceado**, o threshold ideal pode **não ser 0.5** — pode ser ajustado para refletir custos diferentes de falso positivo e falso negativo.

***

## 9.5 Multiclasse

Para classificação multiclasse, o XGBoost usa a **softmax**:

$$
p_k = \frac{e^{\hat{y}_k}}{\sum_j e^{\hat{y}_j}}
$$

Cada classe tem sua própria árvore (ou conjunto de árvores).

***

✅ **Resumo do Capítulo 9:**

* A função de perda é a log-loss, com gradiente:

$$
g_i = \hat{p}_i - y_i
$$

e hessiano:

$$
h_i = \hat{p}_i (1 - \hat{p}_i)
$$

* O modelo trabalha em logits, convertidos em probabilidades pela sigmoid.
* A atualização segue o mesmo esquema de boosting da regressão, mas agora ajustando logits.
* O threshold de decisão pode ser ajustado em datasets desbalanceados.
* Para múltiplas classes, usa-se softmax.
