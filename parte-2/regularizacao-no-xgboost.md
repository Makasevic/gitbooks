# Regularização no XGBoost

## 6.1 Por que regularizar?

O XGBoost é extremamente poderoso, mas justamente por isso tende a **overfitting** se não for controlado.\
A regularização é o conjunto de técnicas que **impõe restrições à complexidade do modelo**, forçando-o a ser mais simples e, portanto, mais generalizável.

👉 Intuição: sem regularização, o modelo “decora” o treino; com regularização, ele “aprende” padrões mais robustos.

***

## 6.2 Regularização L1 e L2

O XGBoost adiciona penalizações diretamente nos **pesos das folhas**:

* **L1 (lasso):** penaliza o valor absoluto dos pesos.
  * Favorece modelos mais esparsos (folhas com valores próximos de zero).
* **L2 (ridge):** penaliza o quadrado dos pesos.
  * Favorece modelos mais estáveis, suavizando valores extremos.

A função de regularização é:

$$
\Omega(f) = \gamma T + \tfrac{1}{2}\lambda \sum_{j=1}^{T} w_j^2 + \alpha \sum_{j=1}^{T} |w_j|
$$

$$
T = \text{número de folhas}
$$

$$
w_j = \text{peso da folha j}
$$

$$
\lambda = \text{coeficiente de regularização L2}
$$

$$
\alpha = \text{coeficiente de regularização L1}
$$

📌 Isso significa que cada árvore é penalizada tanto pelo **número de folhas** quanto pelo **tamanho dos valores nas folhas**.

### Como definir L1 e L2 na prática?

* **L2**

$$
\lambda = \text{parâmetro de regularização L2}
$$

$$
\lambda = 0 \quad \Rightarrow \quad \text{os pesos das folhas podem explodir (overfitting)}
$$

$$
\lambda \text{ muito grande} \quad \Rightarrow \quad \text{todos os pesos ficam próximos de zero (underfitting)}
$$

$$
\text{Heurística: começar com } \lambda = 1 \text{ (default) e aumentar se houver instabilidade ou overfitting}
$$

* **L1**

$$
\alpha = \text{parâmetro de regularização L1}
$$

$$
\alpha = 0 \quad \Rightarrow \quad \text{nenhuma penalização absoluta}
$$

$$
\alpha \text{ muito grande} \quad \Rightarrow \quad \text{muitas folhas são zeradas}
$$

$$
\text{Heurística: começar com } \alpha = 0 \text{ e aumentar apenas se houver muitos splits irrelevantes}
$$

📌 Em resumo:

* **L2 (lambda)** → quase sempre ativo, regula estabilidade.
* **L1 (alpha)** → mais situacional, bom para simplificar árvores em datasets ruidosos.

***

## 6.3 min\_child\_weight

Este parâmetro define o **peso mínimo (soma dos hessianos)** necessário para que um nó possa ser dividido.

$$
\text{Se } \sum h_i < \text{min\_child\_weight} \quad \Rightarrow \quad \text{não divide o nó}
$$

**Mas o significado muda dependendo da função objetivo:**

* **Regressão:**\
  O hessiano de cada observação é **1**.

$$
\text{Ex.: } \text{min\_child\_weight} = 10 \quad \Rightarrow \quad \text{exige pelo menos 10 observações no nó}
$$

* **Classificação:**\
  O hessiano depende da probabilidade prevista. Em log-loss binária:

$$
h_i = p_i (1 - p_i)
$$

$$
\text{Ex.: } \text{min\_child\_weight} = 10 \quad \Rightarrow \quad \text{soma dos pesos efetivos } h_i \text{ deve ser pelo menos 10}
$$

* **Rankeamento:**\
  Mais difícil de interpretar, pois o hessiano vem de funções de ranking.\
  Boa prática: olhar o **cover da raiz**:

$$
Cover(\text{root}) = \sum_{i=1}^{n} h_i
$$

$$
\text{Definir } \text{min\_child\_weight} \text{ como fração de } Cover(\text{root})
$$

📌 Isso dá uma escala razoável para o parâmetro em problemas de ranking.

***

## 6.4 gamma (custo do split)

O parâmetro **gamma (\gamma)** define o **ganho mínimo necessário** para que um split seja aceito.

$$
\text{Split aceito somente se } Gain > \gamma
$$

### Como definir \gamma?

* **Regressão (MSE):**

$$
\gamma \text{ pequeno (0–1) é suficiente, pois os ganhos são grandes}
$$

* **Classificação (log-loss):**

$$
\gamma \text{ deve ser muito baixo (0–2), pois os ganhos são naturalmente pequenos}
$$

* **Rankeamento:**

$$
\gamma \text{ pode ser definido como fração do ganho do split da raiz}
$$

📌 Assim como no `min_child_weight`, use o **próprio dataset como referência**.

***

## 6.5 max\_depth

O parâmetro **max\_depth** limita a **profundidade máxima das árvores**.

* Árvores profundas → maior variância, mais risco de overfitting.
* Árvores rasas → maior viés, mas generalizam melhor.

📌 No XGBoost, `max_depth` geralmente varia entre **3 e 10**, mas depende do problema.

***

## 6.6 subsample e colsample

O XGBoost também inclui **aleatoriedade controlada** na construção das árvores:

* **subsample:** fração de linhas usada em cada árvore.
* **colsample\_bytree:** fração de colunas usada em cada árvore.

👉 Isso funciona como uma injeção de _bagging_ dentro do boosting, reduzindo variância e tornando o modelo mais robusto.

***

## 6.7 Restrições monotônicas 🔹

Um recurso poderoso do XGBoost é a possibilidade de impor **restrições de monotonicidade** em variáveis específicas.

Exemplo: sabemos que **aumentar a renda de um cliente deve sempre aumentar sua probabilidade de aprovação de crédito**.

Podemos forçar o modelo a respeitar essa relação:

* `monotone_constraint = +1` → a predição cresce com a feature.
* `monotone_constraint = -1` → a predição decresce com a feature.

📌 Na prática, o XGBoost restringe os splits possíveis em cada nó de forma que o gradiente respeite a monotonicidade desejada.

👉 Isso combina **conhecimento de domínio** com aprendizado estatístico.

***

## 6.8 Relação entre regularização e viés/variância

Cada hiperparâmetro de regularização desloca o modelo em direção a mais **viés** ou mais **variância**:

* **Mais regularização:**
  * Árvores menores, menos splits, valores suavizados.
  * Maior viés, menor variância.
* **Menos regularização:**
  * Árvores maiores, mais splits, valores extremos.
  * Menor viés, maior variância.

📌 O segredo é encontrar o **equilíbrio certo** para o dataset.

***

✅ **Resumo do Capítulo 6:**

* O XGBoost regula sua complexidade via **L1, L2, número de folhas e profundidade**.
* `min_child_weight` → interpretação diferente para regressão, classificação e ranking (usar cover da raiz).
* `gamma` → custo mínimo do split; heurística baseada no ganho da raiz.
* **L1 e L2** → controlam magnitude dos pesos; ponto de partida recomendado:

$$
\lambda = 1, \quad \alpha = 0
$$

* Restrições monotônicas permitem injetar conhecimento de domínio no modelo.
* Regularização controla o **trade-off viés/variância**, essencial para evitar overfitting.
