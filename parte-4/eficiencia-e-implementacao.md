# Eficiência e Implementação

## 12.1 Estratégias de crescimento da árvore: tree\_method

O XGBoost oferece diferentes métodos de construção de árvores, definidos pelo parâmetro `tree_method`. Cada um possui vantagens e desvantagens:

* **exact (método exato):**
  * Avalia todos os pontos de corte possíveis de cada feature.
  * Garante encontrar o split ótimo em cada nó.
  * ✔️ Alta precisão.
  * ❌ Muito lento e consome memória em datasets grandes.
* **hist (histogram):**
  * Cria histogramas de bins para cada feature.
  * Em vez de avaliar todos os valores, testa apenas os limites dos bins.
  * ✔️ Muito mais rápido e eficiente em memória.
  * ✔️ É o padrão para bases grandes.
  * ❌ Pode introduzir aproximações (mínima perda de precisão).
* **gpu\_hist:**
  * Implementação paralelizada do método hist em GPU.
  * ✔️ Ganhos de velocidade significativos em milhões de observações.
  * ✔️ Escala bem para datasets massivos.

📌 Em aplicações reais, `hist` ou `gpu_hist` são quase sempre preferidos ao `exact`.

***

## 12.2 Tratamento nativo de sparsidade

Muitos datasets têm colunas esparsas (cheias de zeros ou valores ausentes). Isso acontece, por exemplo, após aplicar **one-hot encoding** em variáveis categóricas.

O XGBoost lida com sparsidade de forma **nativa**, sem precisar preencher valores:

* Durante o treino, sempre que encontra valores nulos ou zeros em uma feature, o algoritmo **testa enviar essas observações para a esquerda e para a direita**.
* Calcula o **gain** em cada opção.
* O caminho que gera maior redução na função de perda é escolhido como **branch default**.
* Essa decisão é armazenada no nó da árvore.

👉 Isso significa que você não precisa imputar manualmente valores esparsos: o modelo aprende sozinho como tratá-los.

***

## 12.3 Handling de missing values

O mesmo raciocínio vale para **valores faltantes (NaN):**

* Durante o treino: o XGBoost avalia se mandar os NaN para a esquerda ou para a direita maximiza o ganho de informação.
* Durante a inferência: sempre que o modelo encontra um NaN, ele segue o **branch default** aprendido.

📌 Assim, tanto zeros em dados esparsos quanto valores ausentes explícitos são tratados como parte do processo de aprendizado, e não como algo a ser corrigido manualmente.\
É como se o XGBoost tratasse os valores faltantes como uma “condição especial” da feature e aprendesse qual caminho é mais proveitoso.

***

## 12.4 Comparação com LightGBM e CatBoost

O ecossistema de Gradient Boosted Trees possui outros competidores importantes:

* **LightGBM (Microsoft):**
  * Usa histogramas como padrão.
  * Suporte forte a categorias sem necessidade de one-hot.
  * Mais leve em memória.
  * Muito rápido em datasets grandes.
* **CatBoost (Yandex):**
  * Projetado para lidar com variáveis categóricas diretamente.
  * Utiliza _ordered boosting_, que reduz overfitting.
  * Muito eficiente em datasets mistos (numérico + categórico).
* **XGBoost:**
  * Mais flexível e versátil (suporta regressão, classificação e ranking).
  * Possui ecossistema robusto e suporte a custom objectives.
  * Muito usado em competições de Machine Learning.

📌 Em resumo:

* Se o dataset tem **muitas categorias**, CatBoost tende a se destacar.
* Se o dataset é **muito grande e tabular**, LightGBM pode ser mais rápido.
* Se você precisa de **versatilidade e controle fino**, XGBoost segue como padrão-ouro.

***

✅ **Resumo do Capítulo 12:**

* O parâmetro `tree_method` define como as árvores são construídas (`exact`, `hist`, `gpu_hist`).
* O XGBoost lida nativamente com sparsidade e valores faltantes testando os dois caminhos (esquerda/direita) e aprendendo o branch default que maximiza o gain.
* Comparado a LightGBM e CatBoost, o XGBoost é o mais versátil e customizável, embora nem sempre o mais rápido.
