# Grandes volumes de dados

## 15.1 O desafio de escalar o boosting

O XGBoost é poderoso, mas pode ser exigente em termos de tempo e memória.\
Quando lidamos com **milhões ou bilhões de observações**, é essencial adotar estratégias para treinar de forma eficiente sem perder qualidade.

***

## 15.2 Uso de GPU

Uma das formas mais eficazes de acelerar o treino é usar **GPU** com o parâmetro `tree_method="gpu_hist"`:

* Constrói histogramas em paralelo na GPU.
* Acelera drasticamente o treino em bases massivas.
* Escala bem mesmo com centenas de features.

📌 Exemplos práticos de uso:

* Previsão de séries financeiras com milhões de registros históricos.
* Modelos de risco de crédito com grandes bases transacionais.

***

## 15.3 Paralelização em CPU

Mesmo sem GPU, o XGBoost já é paralelizado por padrão:

* O parâmetro `nthread` define quantos núcleos de CPU serão usados.
* Cada árvore pode ser construída em paralelo por feature.
* O método `hist` aproveita melhor múltiplos núcleos que o `exact`.

👉 Para máquinas multicore, configurar `nthread` corretamente pode reduzir muito o tempo de treino.

***

## 15.4 Treinamento distribuído

Para volumes realmente massivos, o XGBoost pode ser distribuído em clusters:

* **Dask-XGBoost (Python):** integração com Dask para paralelização distribuída.
* **XGBoost4J-Spark (Java/Scala):** integração com Apache Spark.
* Permite treinar em **bilhões de linhas**, repartindo o dataset entre nós do cluster.

📌 Exemplo de aplicação:

* Recomendação de produtos em e-commerce com centenas de milhões de interações de usuários.

***

## 15.5 Estratégias práticas de eficiência

Além de GPU e clusters, existem ajustes práticos que ajudam em datasets grandes:

* **Subsample e colsample\_bytree:**
  * Usar apenas parte das observações ou features em cada árvore.
  * Reduz custo computacional e ainda ajuda na regularização.
* **Pré-processamento eficiente:**
  * Remover features redundantes ou constantes.
  * Usar compressão de dados numéricos (ex.: `float32` em vez de `float64`).
* **Batching:**
  * Em alguns casos, é possível dividir os dados em batches, treinar modelos separados e combiná-los (stacking/ensembles).

***

## 15.6 Quando evitar boosting em big data

Embora otimizado, o XGBoost pode ser **overkill** em cenários massivos onde:

* O problema é simples e uma **regressão linear** já atinge boa performance.
* A latência de predição precisa ser mínima (milissegundos).
* O custo de tuning do modelo supera os ganhos de acurácia.

📌 Muitas vezes, a simplicidade vence em problemas de altíssima escala.

***

✅ **Resumo do Capítulo 15:**

* O XGBoost suporta **GPU** e paralelização em CPU para acelerar o treino.
* Pode ser distribuído via **Dask** ou **Spark**, escalando para bilhões de observações.
* Estratégias como **subsample**, **colsample** e pré-processamento reduzem custo sem perder qualidade.
* Em alguns cenários, modelos mais simples podem ser preferíveis a boosting em big data.
