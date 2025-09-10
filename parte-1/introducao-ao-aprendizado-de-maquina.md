# Introdução ao Aprendizado de Máquina

## 1.1 O que é aprendizado de máquina supervisionado

O **aprendizado de máquina (machine learning)** é a área da ciência da computação que cria modelos capazes de **aprender padrões a partir de dados**.

Diferente de algoritmos tradicionais, onde o programador escreve todas as regras, em machine learning nós fornecemos **exemplos (dados de treino)** e o modelo descobre automaticamente as relações subjacentes.

* **Input (features):** são as variáveis de entrada que descrevem um objeto ou situação.\
  Exemplo: preço de uma casa pode depender de `metragem`, `número de quartos`, `bairro`.
* **Output (target):** é a variável que queremos prever.\
  Exemplo: o valor final de venda da casa.

Quando o modelo aprende **a partir de pares (input, output)**, chamamos isso de **aprendizado supervisionado**.

***

## 1.2 Regressão, Classificação e Rankeamento

Dentro do aprendizado supervisionado, existem três grandes tipos de problemas que o XGBoost consegue resolver:

* **Regressão**\
  O objetivo é prever um **valor contínuo**.
  * Exemplo: prever o preço de uma ação amanhã.
  * Função de perda comum: erro quadrático médio (MSE).
  * Output do modelo: um número real (pode ser qualquer valor).
* **Classificação**\
  O objetivo é prever uma **classe** entre várias possibilidades.
  * Exemplo: identificar se um e-mail é spam ou não.
  * Função de perda comum: log-loss (entropia cruzada).
  * Output do modelo: probabilidade de cada classe (ex.: 80% spam, 20% não spam).
* **Rankeamento**\
  O objetivo é ordenar itens de acordo com sua relevância.
  * Exemplo: em um buscador, dado o termo “restaurante japonês”, queremos que os melhores resultados apareçam no topo.
  * Função de perda: baseada em pares de comparação (LambdaRank, NDCG).
  * Output do modelo: um **score relativo** que define a ordem dos itens.

👉 Embora pareçam diferentes, todos esses problemas podem ser resolvidos pela **mesma base algorítmica do XGBoost**: árvores de decisão combinadas em boosting. O que muda é a **função de perda (loss function)** que direciona o aprendizado.

***

## 1.3 Conceito de erro, bias e variância

Para avaliar modelos de ML, precisamos entender o que significa **errar** e por que isso acontece. Dois conceitos centrais são **bias** e **variância**:

* **Bias (viés):** erro sistemático. O modelo é **simples demais** para capturar a realidade.\
  Exemplo: prever o preço de casas apenas pela metragem, ignorando bairro e número de quartos.
* **Variância:** erro devido à sensibilidade excessiva ao treino. O modelo é **complexo demais** e se adapta até ao ruído.\
  Exemplo: prever o preço de casas memorizando cada imóvel específico do treino.

O ideal é encontrar um **equilíbrio** entre bias e variância, evitando tanto a simplificação exagerada quanto a supercomplexidade.

***

## 1.4 Overfitting e underfitting

Dois termos práticos que derivam de bias/variância:

* **Underfitting:**\
  O modelo é **simples demais**, não captura os padrões.\
  Exemplo: uma reta tentando ajustar dados que seguem uma parábola.
* **Overfitting:**\
  O modelo é **complexo demais**, aprende até os ruídos do treino e perde capacidade de generalizar.\
  Exemplo: uma curva extremamente tortuosa que passa exatamente por todos os pontos do treino, mas erra nos novos.

### Analogia rápida

Pense em aprender a jogar futebol:

* **Underfitting:** você só aprende a chutar de um jeito, sempre erra porque não se adapta.
* **Overfitting:** você decora cada chute específico do treino, mas quando a bola vem de outro jeito no jogo, você falha.

O **XGBoost** é poderoso porque possui **mecanismos de regularização** para controlar o overfitting e encontrar esse equilíbrio.

***

## ✅ Resumo do Capítulo 1

* Machine learning supervisionado aprende a partir de pares (input, output).
* Existem três grandes problemas: **regressão, classificação e rankeamento**.
* Todo modelo precisa lidar com o equilíbrio entre **bias e variância**.
* **Overfitting** e **underfitting** são extremos a serem evitados — e o XGBoost oferece ferramentas próprias para isso.
