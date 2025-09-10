# Validação e Treinamento

## 13.1 O desafio da validação em boosting

Modelos de boosting, como o XGBoost, aprendem de forma **sequencial**, onde cada árvore corrige erros das anteriores.\
Isso traz poder preditivo, mas também torna a validação mais sensível:

* Pequenos erros no split de treino/validação podem influenciar fortemente o modelo.
* Overfitting pode surgir rápido se a validação não for representativa.
* Em séries temporais, é ainda mais crítico respeitar a ordem cronológica.

📌 Diferente de regressões lineares ou árvores isoladas, o boosting precisa de **validação bem planejada** para refletir sua capacidade de generalização.

***

## 13.2 Divisão treino/validação/teste

A divisão tradicional em **treino/validação/teste** continua válida, mas com cuidados adicionais:

* **Treino:** usado para construir as árvores.
* **Validação:** monitora o erro durante o treino, ajusta hiperparâmetros e controla overfitting.
* **Teste:** avalia apenas no final, para garantir que o modelo generaliza.

👉 Em **séries temporais**, não é permitido embaralhar os dados.\
O ideal é usar **expanding window** ou **rolling window**, onde a validação é sempre feita em blocos futuros.

***

## 13.3 Early stopping

O XGBoost oferece o recurso de **early stopping** (`early_stopping_rounds`):

* O modelo é treinado com muitas árvores (ex.: 2000).
* A cada iteração, mede-se a métrica na validação.
* Se não houver melhora após X rodadas consecutivas, o treino para.

✔️ **Vantagem:** economiza tempo e evita overfitting.\
❌ **Risco:** se a validação não é representativa, o modelo pode parar cedo demais.

📌 Em datasets pequenos, é comum usar validação cruzada em vez de early stopping para ter maior segurança.

***

## 13.4 Cross-validation no XGBoost

O XGBoost possui a função `xgb.cv`, que executa **k-fold cross-validation** automaticamente:

* Divide o dataset em k partes.
* Treina k vezes, cada vez deixando uma parte para validação.
* Retorna métricas médias e desvio-padrão.

✔️ Útil em datasets pequenos.\
❌ Custo computacional alto em boosting, já que o treino é multiplicado por k.

👉 Em **séries temporais**, deve-se usar **time-series CV** (expanding/rolling window), em vez de embaralhar os dados.

***

## 13.5 Ordem de prioridade no tuning

O XGBoost tem muitos hiperparâmetros, mas alguns são mais importantes que outros.\
A ordem de tuning recomendada é:

1. **n\_estimators** (número de árvores): define a complexidade do ensemble.
2. **learning\_rate (eta):** controla o peso de cada árvore (taxa de aprendizado).
3. **max\_depth / min\_child\_weight:** controlam a complexidade individual das árvores.
4. **subsample / colsample\_bytree:** trazem aleatoriedade e reduzem overfitting.
5. **lambda, alpha (L2, L1):** regularização nas folhas.
6. **gamma:** custo para criar novos splits.

📌 Essa ordem ajuda a evitar grid search desnecessário e reduz custo de tuning.

***

## 13.6 Boas práticas

* Sempre usar **validação alinhada ao problema real** (temporal ou por grupos).
* Não confiar em métricas de treino — olhar sempre para validação.
* Ajustar o número de árvores e learning rate juntos (trade-off).
* Usar **early stopping** como guia, mas confirmar em um **teste final**.

***

✅ **Resumo do Capítulo 13:**

* O boosting precisa de validação cuidadosa, mais ainda em séries temporais.
* O split treino/validação/teste deve respeitar a estrutura dos dados.
* Early stopping ajuda a evitar overfitting, mas depende da qualidade da validação.
* Cross-validation é útil em bases pequenas, mas caro.
* O tuning deve seguir uma ordem de prioridade, evitando buscas cegas.
