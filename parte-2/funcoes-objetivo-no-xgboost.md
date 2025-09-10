# Funções Objetivo no XGBoost

## 5.1 Estrutura geral

Todo modelo do XGBoost é guiado por uma **função objetivo** (_objective function_), que tem duas partes:

$$
\mathcal{L} = \sum_{i=1}^n \ell(y_i, \hat{y}_i) + \Omega(f)
$$

$$
\ell(y_i, \hat{y}_i) = \text{erro entre previsão e valor real}
$$

$$
\Omega(f) = \text{penalização por complexidade do modelo}
$$

👉 A loss direciona o aprendizado, enquanto a regularização controla o overfitting.

***

## 5.2 Expansão de Taylor (2ª ordem)

Para otimizar rápido, o XGBoost aproxima a loss usando uma **expansão de Taylor de 2ª ordem** em torno da previsão atual $f\_{m-1}(x)$:

$$
\ell(y_i, f_{m-1}(x_i) + f_m(x_i)) \approx \ell(y_i, f_{m-1}(x_i)) + g_i f_m(x_i) + \tfrac{1}{2} h_i f_m(x_i)^2
$$

**Definições:**

$$
g_i = \frac{\partial \ell}{\partial f}(y_i, f_{m-1}(x_i)) = \text{gradiente (1ª derivada da loss)}
$$

$$
h_i = \frac{\partial^2 \ell}{\partial f^2}(y_i, f_{m-1}(x_i)) = \text{hessiano (2ª derivada da loss)}
$$

📌 Isso dá ao modelo duas informações cruciais:

* **Direção da correção (gradiente)** → indica se a previsão deve subir ou descer.
* **Tamanho do passo (hessiano)** → indica o quão forte deve ser essa correção.

👉 É como ter um GPS que não só mostra para onde ir, mas também a velocidade ideal para chegar.

***

## 5.3 Peso ótimo de uma folha

Quando os gradientes e hessianos são acumulados em cada nó, o XGBoost calcula o **peso ótimo da folha**:

$$
w^* = -\frac{G}{H + \lambda}
$$

$$
G = \sum g_i = \text{soma dos gradientes do nó}
$$

$$
H = \sum h_i = \text{soma dos hessianos do nó}
$$

$$
\lambda = \text{termo de regularização L2}
$$

📌 Interpretação: o modelo ajusta o valor da folha na direção contrária ao gradiente (reduzindo erro), mas ponderado pela “curvatura” da loss (hessiano) e pela regularização.

Isso torna o update **eficiente e estável**, mesmo em dados ruidosos.

***

## 5.4 Ganho de um split

Para decidir se vale a pena dividir um nó, o XGBoost calcula o **ganho esperado**:

$$
Gain = \tfrac{1}{2} \left( \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda} \right) - \gamma
$$

📌 Um split só é aceito se o ganho for positivo e superar $\gamma$.

***

## 5.5 Funções objetivo comuns

O XGBoost suporta várias losses, cada uma adaptada a um tipo de problema:

* **Regressão**
  * Squared Error (MSE):

$$
\ell = (y - \hat{y})^2 = \text{erro quadrático médio}
$$

* MAE (erro absoluto): mais robusto a outliers.
* **Classificação binária**
  * Log-loss:

$$
\ell = -[y \log p + (1-y)\log(1-p)] = \text{entropia cruzada}
$$

* Output passa por função sigmoide → probabilidades entre 0 e 1.
* **Classificação multiclasse**
  * Softmax + entropia cruzada.
  * O modelo aprende scores que são convertidos em probabilidades via softmax.
* **Rankeamento**
  * Funções baseadas em pares (pairwise loss).
  * LambdaRank / LambdaMART → aproximam métricas como NDCG via gradiente.

***

## 5.6 O papel do learning rate

Mesmo com a expansão de Taylor ajustando passo e direção, o XGBoost introduz o **learning rate ($\eta$)**:

$$
F_m(x) = F_{m-1}(x) + \eta \cdot w^*
$$

$$
\eta = \text{learning rate (suaviza o impacto de cada árvore)}
$$

📌 É melhor usar um **learning rate pequeno** com mais árvores do que um learning rate alto com poucas.

***

✅ **Resumo do Capítulo 5:**

* Toda função objetivo = **Loss + Regularização**.
* O XGBoost usa a **2ª ordem (gradiente + hessiano)** para aprender mais rápido e de forma estável.
* Cada folha recebe um **peso ótimo** proporcional ao gradiente e inversamente proporcional ao hessiano.
* O **ganho de um split** garante que só divisões úteis sejam feitas.
* Funções objetivo variam conforme a tarefa: regressão, classificação ou ranking.
* O **learning rate** suaviza as correções, controlando o ritmo do aprendizado.
