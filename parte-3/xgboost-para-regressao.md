# XGBoost para Regressão

## 8.1 Função de perda em regressão

O caso mais comum é o **erro quadrático médio (MSE)**:

$$
\ell(y_i, \hat{y}_i) = (y_i - \hat{y}_i)^2
$$

**Definições**

$$
y_i = \text{valor real}
$$

$$
\hat{y}_i = \text{predição}
$$

No boosting, usamos **gradiente** e **hessiano** dessa loss:

$$
g_i = \frac{\partial \ell}{\partial \hat{y}_i} = -2\, (y_i - \hat{y}_i)
$$

$$
h_i = \frac{\partial^2 \ell}{\partial \hat{y}_i^{\,2}} = 2
$$

👉 Repare que aqui o **hessiano é constante (2)**, o que simplifica bastante os cálculos.

Outras opções possíveis em regressão:

* **MAE (erro absoluto)**: mais robusto a outliers.
* **Quantile loss**: usada em previsão intervalar.

***

## 8.2 Exemplo numérico – passo a passo

Vamos prever o preço de 3 ações a partir de uma feature simplificada (ex.: momentum).

| Ativo | Retorno real (y) | Feature (momentum) |
| ----- | ---------------- | ------------------ |
| A     | 0.10             | 0.5                |
| B     | 0.05             | 0.2                |
| C     | -0.02            | -0.3               |

***

### Passo 1 – Score inicial

No início, o XGBoost define a predição como a média global:

$$
\hat{y}^{(0)} = \frac{0.10 + 0.05 - 0.02}{3} \approx 0.043
$$

***

### Passo 2 – Gradientes e Hessianos

Para cada amostra (usando a média global acima):

$$
g_i = -2\, (y_i - \hat{y}^{(0)}), \qquad h_i = 2
$$

*   **A**

    $$
    g_A = -2\,(0.10 - 0.043) = -0.114
    $$
*   **B**

    $$
    g_B = -2\,(0.05 - 0.043) = -0.014
    $$
*   **C**

    $$
    g_C = -2\,(-0.02 - 0.043) = 0.126
    $$

***

### Passo 3 – Treinando a primeira árvore

O modelo tenta separar pelas features. Suponha que faça o split: _momentum > 0 vs ≤ 0_.

*   **Grupo 1 (A, B):**

    $$
    G = -0.128, \quad H = 4
    $$
*   **Grupo 2 (C):**

    $$
    G = 0.126, \quad H = 2
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
    w^* = -\frac{-0.128}{4+1} = 0.026
    $$
*   **Grupo 2**

    $$
    w^* = -\frac{0.126}{2+1} = -0.042
    $$

***

### Passo 4 – Atualizando predições

**Learning rate:**

$$
\eta = 0.5
$$

*   **A**

    $$
    \hat{y}^{(1)} = 0.043 + 0.5 \times 0.026 \approx 0.056
    $$
*   **B**

    $$
    \hat{y}^{(1)} = 0.043 + 0.5 \times 0.026 \approx 0.056
    $$
*   **C**

    $$
    \hat{y}^{(1)} = 0.043 + 0.5 \times (-0.042) \approx 0.022
    $$

***

### Passo 5 – Iterações seguintes

O processo se repete:

1. Recalcular gradientes e hessianos com as novas predições.
2. Treinar uma nova árvore para ajustar os resíduos.
3. Atualizar as predições com o fator (\eta).

Após **3 boosts**, o modelo já consegue aproximar bem os valores reais.

***

## 8.3 Interpretação dos outputs

* O output final em regressão é **um valor contínuo**.
* Não há transformação adicional (como sigmoid ou softmax).
* O ajuste vem da **soma das correções** feitas por cada árvore.

***

## 8.4 Papel do learning rate

Sem learning rate:

$$
\eta = 1 \quad \Rightarrow \quad \text{cada árvore aplica a correção completa } w^*
$$

Isso faria o modelo aprender rápido, mas arriscaria **overfitting**.

Com amortecimento:

$$
0 < \eta < 1 \quad \Rightarrow \quad \text{cada ajuste é suavizado (passo menor)}
$$

* Aprendizado mais lento, porém mais estável.
* Permite usar mais árvores sem explodir a variância.

***

✅ **Resumo do Capítulo 8:**

* Em regressão, o XGBoost normalmente usa MSE como função de perda.
* O hessiano é constante (2), simplificando cálculos.
* O modelo começa pela **média global** e corrige iterativamente os resíduos.
* O output final é um **valor contínuo**, obtido pela soma das correções.
* O **learning rate** controla a suavização do aprendizado.
