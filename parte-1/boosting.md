# Boosting

## 3.1 Ensemble Learning: o poder do coletivo

Uma árvore de decisão sozinha pode ser instável e ter baixo poder preditivo.

A solução é usar **conjuntos de modelos (ensembles)**, que combinam várias árvores para melhorar desempenho.

Existem dois tipos principais de ensembles:

* **Bagging (Bootstrap Aggregating):**
  * Cada árvore é treinada de forma independente, em amostras diferentes dos dados.
  * As predições são combinadas (média ou votação).
  * Exemplo: **Random Forest**.
  * Vantagem: reduz variância.
  * Limitação: árvores não aprendem umas com as outras.
* **Boosting:**
  * As árvores são treinadas de forma **sequencial**, onde cada nova árvore tenta corrigir os erros das anteriores.
  * As predições são combinadas de forma **aditiva** (somadas).
  * Vantagem: reduz viés e melhora muito a performance.
  * Limitação: maior risco de overfitting (precisa de regularização).

***

## 3.2 Intuição do Boosting

Imagine um aluno aprendendo matemática:

* Primeiro faz uma prova sem estudar → erra muito.
* Depois, estuda os erros cometidos → melhora.
* Em seguida, revisa apenas os pontos que ainda estão confusos → melhora mais.

📌 Cada passo não refaz tudo do zero, mas **foca no que ainda falta aprender**.

Esse é o espírito do boosting: cada árvore acrescenta **uma correção** ao modelo existente.

***

## 3.3 AdaBoost – O Boosting original

O **AdaBoost** foi uma das primeiras implementações práticas de boosting:

* Cada observação começa com um peso igual.
* Após a primeira árvore, os exemplos mal classificados recebem **peso maior**.
* A próxima árvore foca nos exemplos “difíceis”.
* No final, cada árvore tem uma importância proporcional ao seu desempenho.

👉 Funciona muito bem em classificação binária, mas não é tão flexível para outros problemas.

***

## 3.4 Gradient Boosting – O salto conceitual

O **Gradient Boosting** generaliza a ideia:

* Em vez de reponderar exemplos, usa o **gradiente da função de perda** para decidir como cada nova árvore deve corrigir o modelo.
* A cada iteração, o modelo é atualizado de forma incremental:

$$
F_m(x) = F_{m-1}(x) + \eta \, h_m(x)
$$

$$
F_m(x) = \text{modelo após $m$ iterações}
$$

$$
h_m(x) = \text{nova árvore adicionada}
$$

$$
\eta = \text{learning rate, que controla o tamanho do passo}
$$

📌 Intuição: cada nova árvore não tenta prever o valor final direto, mas **contribui com um pequeno ajuste** para reduzir o erro do modelo atual.

***

## 3.5 Boosting e Overfitting

Boosting é muito poderoso, mas também propenso a **overfitting**.

Por isso controlamos sua complexidade com:

*   **Learning rate**

    $$
    \eta = \text{fator que diminui o impacto de cada nova árvore}
    $$
* **Número de árvores:** mais árvores = mais precisão, até certo ponto.
* **Profundidade das árvores:** árvores rasas (stumps) capturam apenas padrões simples e reduzem risco de memorizar ruído.

👉 O XGBoost é justamente uma implementação que acrescenta **regularização forte** para tornar boosting robusto.

***

## 3.6 Conexão com o XGBoost

O XGBoost é uma versão otimizada do Gradient Boosting que traz:

* Regularização explícita (L1 e L2).
* Eficiência (uso de histogramas, paralelização, GPU).
* Tratamento nativo de valores faltantes.
* Suporte a regressão, classificação e rankeamento em grande escala.

📌 O boosting é o **coração conceitual**. O XGBoost é a versão **industrial e turbinada**.

***

## 3.7 Como o boosting prevê em novos dados

Uma dúvida comum: como usar o modelo final, se ele contém muitas árvores diferentes?

O processo é simples:

1. O novo dado é passado pela **árvore 1** → cai numa folha → gera uma predição parcial.
2. O mesmo dado passa pela **árvore 2** → nova predição parcial.
3. Isso se repete até a **árvore M**.
4. O resultado final é a **soma de todas as contribuições** (ajustadas pelo learning rate).

$$
\hat{y}(x) = \sum_{m=1}^M \eta \, h_m(x)
$$

📌 Importante:

* As árvores foram **treinadas em sequência**, mas **na predição elas atuam em paralelo**.
* Não usamos só a última árvore: usamos **todas**.
* Cada árvore dá sua contribuição independente, e o modelo final é a soma dessas partes.

***

✅ **Resumo do Capítulo 3:**

* Ensembles combinam modelos → bagging reduz variância, boosting reduz viés.
* AdaBoost usa reponderação de exemplos; Gradient Boosting usa gradientes da loss.
* Cada nova árvore é uma correção incremental ao modelo.
* Boosting é forte, mas precisa de controle (learning rate, profundidade, nº de árvores).
* O XGBoost é a versão otimizada e regularizada do Gradient Boosting.
* Em predição, todas as árvores contribuem: a previsão final é a soma de todas as saídas.
