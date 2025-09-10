# Custom objective e custom metric

## 14.1 Por que customizar?

O XGBoost já oferece funções de perda (objectives) e métricas de avaliação prontas, como:

* **reg:squarederror** → regressão
* **binary:logistic** → classificação
* **rank:pairwise** → rankeamento

Mas em muitas situações do mundo real, a métrica que realmente importa não está diretamente disponível.\
Exemplos: prever quantis (Value-at-Risk), otimizar métricas robustas contra outliers ou aproximar métricas de ranking complexas como NDCG.

👉 A flexibilidade do XGBoost permite que o usuário defina suas próprias funções de perda e métricas de avaliação.

***

## 14.2 Custom objective

Um **custom objective** deve retornar dois vetores para cada observação:

* **gradiente (g):** primeira derivada da loss em relação ao score.
* **hessiano (h):** segunda derivada da loss.

O XGBoost usa esses valores para construir as árvores de forma exata, assim como faria com seus objectives nativos.

📌 Estrutura em Python:

```python
def custom_loss(y_pred, dtrain):
    y_true = dtrain.get_label()
    # calcular gradiente g
    # calcular hessiano h
    return grad, hess
```

### Exemplo: Quantile Regression

A regressão quantílica busca prever um quantil específico (ex.: mediana, percentil 95).\
A loss é assimétrica: penaliza mais os erros acima ou abaixo do alvo, dependendo do quantil.

```python
def quantile_loss(alpha=0.5):
    def loss(y_pred, dtrain):
        y_true = dtrain.get_label()
        diff = y_true - y_pred
        grad = np.where(diff < 0, -alpha, 1 - alpha)
        hess = np.ones_like(y_true)
        return grad, hess
    return loss
```

👉 Esse tipo de loss é muito usado em risco financeiro (ex.: cálculo de VaR).

***

## 14.3 Custom metric

Já as **custom metrics** não influenciam o treino, apenas a forma como monitoramos o desempenho do modelo.\
Devem retornar uma tupla:

* **nome** da métrica
* **valor** calculado
* **booleano** indicando se valores maiores são melhores (`True`) ou menores (`False`).

📌 Estrutura em Python:

```python
def custom_metric(y_pred, dtrain):
    y_true = dtrain.get_label()
    return 'metric_name', valor, is_higher_better
```

### Exemplo: MAPE (Mean Absolute Percentage Error)

```python
def mape(y_pred, dtrain):
    y_true = dtrain.get_label()
    return 'mape', np.mean(np.abs((y_true - y_pred) / y_true)), False
```

👉 O MAPE é útil em previsão de demanda e séries temporais financeiras.

***

## 14.4 Outros exemplos práticos

* **Fair loss:** dá menos peso a outliers, suavizando o impacto de grandes erros.
* **Huber loss:** combina MSE e MAE, sendo mais robusta que o erro quadrático puro.
* **Custom ranking metric:** é possível aproximar métricas como NDCG ou MAP via gradientes diferenciáveis.

***

## 14.5 Cuidados ao usar objectives customizados

* A função precisa retornar gradiente e hessiano corretos — se mal implementados, o modelo pode divergir.
* Hiperparâmetros de regularização podem precisar de ajuste extra.
* Métricas customizadas podem aumentar o custo computacional.

📌 Sempre valide se o custom objective realmente melhora a métrica de interesse no conjunto de teste.

***

✅ **Resumo do Capítulo 14:**

* O XGBoost permite definir **custom objectives** (com gradiente e hessiano) e **custom metrics** (avaliação).
* Isso dá enorme flexibilidade para adaptar o modelo a problemas específicos.
* Exemplos incluem quantile regression, MAPE, Huber e métricas de ranking.
* É preciso cuidado ao implementar, garantindo derivadas corretas e interpretabilidade.
