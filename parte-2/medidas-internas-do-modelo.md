# Medidas internas do modelo

## 7.1 Por que entender as métricas internas?

Quando treinamos um modelo no XGBoost, cada split e cada folha têm métricas associadas:

* **Gain**
* **Cover**
* **Weight (ou Frequency)**

Essas métricas ajudam a interpretar **por que uma feature foi escolhida** e qual sua relevância no modelo.\
Elas não são métricas de acurácia ou erro final, mas sim **indicadores internos de qualidade dos splits**.

***

## 7.2 Derivação do Gain

O **Gain** mede o **aumento esperado na função objetivo** quando fazemos um split em determinado ponto.\
Essa fórmula vem diretamente da **expansão de Taylor de 2ª ordem** usada no XGBoost.

### Passo 1 – Expansão de Taylor

$$
\ell(y_i, f_{m-1}(x_i) + f_m(x_i)) \approx \ell(y_i, f_{m-1}(x_i)) + g_i f_m(x_i) + \tfrac{1}{2} h_i f_m(x_i)^2
$$

onde:

$$
g_i = \frac{\partial \ell}{\partial f}(y_i, f_{m-1}(x_i)) \quad \text{gradiente}
$$

$$
h_i = \frac{\partial^2 \ell}{\partial f^2}(y_i, f_{m-1}(x_i)) \quad \text{hessiano}
$$

***

### Passo 2 – Função objetivo em uma folha

Definições dos símbolos:

$$
I_j = \text{o conjunto de exemplos que caem na folha } j
$$

$$
w_j = \text{o peso (valor) atribuído à folha } j
$$

Com isso, a função objetivo aproximada para a folha (j) é:

$$
Obj_j(w_j) \approx \sum_{i \in I_j} \Big( g_i w_j + \tfrac{1}{2} h_i w_j^2 \Big) + \tfrac{1}{2} \lambda w_j^2 + \gamma
$$

*

$$
\lambda = \text{regularização L2}
$$

$$
\gamma = \text{custo de criar uma folha}
$$

***

### Passo 3 – Peso ótimo da folha

$$
w_j^* = - \frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}
$$

***

### Passo 4 – Ganho de um split

Se dividirmos um nó em esquerda (L) e direita (R), o ganho é a redução da perda obtida:

$$
Gain = \tfrac{1}{2} \left( 
\frac{G_L^2}{H_L + \lambda} + 
\frac{G_R^2}{H_R + \lambda} - 
\frac{(G_L + G_R)^2}{H_L + H_R + \lambda} 
\right) - \gamma
$$

onde:

$$
G_L = \sum_{i \in I_L} g_i, \quad H_L = \sum_{i \in I_L} h_i
$$

$$
G_R = \sum_{i \in I_R} g_i, \quad H_R = \sum_{i \in I_R} h_i
$$

O split só é aceito se:

$$
Gain > \gamma
$$

***

## 7.3 Cover (cobertura)

O **Cover** mede o “peso total” dos exemplos que passam por um nó ou folha:

$$
Cover = \sum_{i \in \text{nó}} h_i
$$

* **Em regressão:**

$$
h_i = 1 \quad \Rightarrow \quad \text{o cover é aproximadamente o número de observações}
$$

* **Em classificação:**

$$
h_i = p_i (1 - p_i) \quad \Rightarrow \quad \text{o cover reflete a incerteza média dos exemplos}
$$

* **Em rank:pairwise:**

$$
\text{o cover vem da soma dos hessianos definidos pela função de ranking}
$$

Splits com cover alto **afetam muitos exemplos**; splits com cover baixo **afetam poucos exemplos** (tendem a ser menos estáveis).



💡 E por que o Cover é a soma dos hessianos e não dos gradientes?

Essa definição não é arbitrária. Ela vem diretamente da expansão de Taylor usada pelo XGBoost para aproximar a loss.

Na expansão de 2ª ordem:

$$
\ell(y_i, f(x_i) + \Delta f(x_i)) \approx \ell(y_i, f(x_i)) + g_i \cdot \Delta f(x_i) + \tfrac{1}{2} h_i (\Delta f(x_i))^2
$$

* O **gradiente** indica a direção da correção (positivo ou negativo).
* O **hessiano** indica a curvatura da loss, ou seja, o "peso de confiança" dessa direção.

Se usássemos só gradientes para medir _cover_, erros positivos e negativos poderiam se cancelar, mascarando a quantidade de informação real.\
Já os hessianos são sempre positivos e capturam a intensidade do ajuste.

***

## 7.4 Weight (frequência)

O **Weight** é simplesmente a contagem de **quantas vezes uma feature foi usada em splits** ao longo de todas as árvores.

* Mede apenas a frequência, **não** o ganho.
* Pode dar uma visão enviesada: uma feature pode aparecer muito, mas em splits de baixo ganho.

***

## 7.5 Relação com Feature Importance

O XGBoost permite calcular importância de features de três formas principais:

1. **Gain** → importância proporcional ao ganho acumulado.
2. **Cover** → importância proporcional à cobertura acumulada.
3. **Weight (Frequency)** → importância proporcional à frequência de uso.

Geralmente, **Gain** é a mais usada, mas pode ser enviesada em datasets com muitas categorias.\
**SHAP values** (Capítulo 11) são uma forma mais robusta de medir importância real.

***

## 7.6 Intuição prática

* **Gain** → quanto o split ajudou a reduzir erro.
* **Cover** → quantos dados o split afetou.
* **Weight** → quantas vezes a feature foi usada.

Juntas, essas métricas mostram **como o modelo toma decisões internamente**.

***

✅ **Resumo do Capítulo 7:**

* O Gain vem da expansão de Taylor e mede a redução da perda de um split.
* O Cover mede o peso dos exemplos em cada nó.
* O Weight mede a frequência de uso da feature.
* Essas métricas são base para a Feature Importance nativa do XGBoost.
