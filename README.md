# Índice

***

### **Parte I – Fundamentos do Aprendizado de Máquina**

1. **Introdução**
   * O que é aprendizado de máquina supervisionado
   * Diferença entre regressão, classificação e rankeamento
   * Conceito de erro, bias e variância
   * Overfitting e underfitting: por que precisamos regularizar
2. **Árvores de decisão**
   * Estrutura de uma árvore (nós, folhas, splits)
   * Funções de custo para splits: MSE, entropia, Gini
   * Critérios de parada (max\_depth, min\_samples, ganho mínimo)
   * Exemplo prático: construindo uma árvore à mão com 6 pontos
3. **Boosting**
   * O que é ensemble learning (bagging vs boosting)
   * AdaBoost e a ideia de reponderar exemplos difíceis
   * Gradient Boosting: usar gradientes para corrigir erros
   * Exemplo visual de boosting: três árvores corrigindo erros sequenciais

***

### **Parte II – O XGBoost em profundidade**

4. **Arquitetura e Filosofia do XGBoost**
   * Por que o XGBoost foi criado
   * O que o diferencia de outros GBMs (regularização, velocidade, sparsidade)
   * Fluxo interno do algoritmo (do input ao output)
   * Filosofia do design
5. **Funções objetivo (Objective functions)**
   * Estrutura geral: Loss + Regularização
   * Conceito de gradiente e hessiano (primeira e segunda derivada)
   * Como o XGBoost usa a segunda ordem para otimizar mais rápido
   * Diferença entre loss de regressão, classificação e ranking
6. **Regularização no XGBoost**
   * L1 (lasso) vs L2 (ridge) no espaço das folhas
   * min\_child\_weight: mínimo de “peso” para criar um split
   * gamma: custo para criar uma nova divisão
   * max\_depth e sua relação com variância
   * subsample e colsample (bagging dentro do boosting)
   * Relação entre regularizadores e o viés/variância
   * 🔹 Restrições monotônicas (aplicação prática e impacto no gradiente)
7. **Medidas internas do modelo**
   * Gain: ganho de informação de um split
   * Cover: “peso” dos exemplos que passam por um nó
   * Weight: número de ocorrências em cada folha
   * Como essas medidas aparecem na feature importance

***

### **Parte III – Aplicações práticas do XGBoost**

8. **XGBoost para Regressão**
   * Função de perda: Squared Error, MAE
   * Passo a passo com exemplo numérico:
     * 1ª árvore: ajustando aos resíduos iniciais
     * 2ª árvore: correção do erro da primeira
     * 3ª árvore: refinamento final
   * Como o learning\_rate “suaviza” cada correção
   * Como interpretar os outputs (valores contínuos)
9. **XGBoost para Classificação**
   * Função de perda: log-loss (binary logistic)
   * Como o modelo gera scores → logits → probabilidades via sigmoid
   * Exemplo prático com 3 boosts em um dataset pequeno
   * Probabilidade vs. decisão (threshold 0.5 e casos desbalanceados)
   * Multiclass: softmax e generalização do processo
10. **XGBoost para Rank:pairwise**
    * Conceito de ranking (queries, docs, ordem de relevância)
    * Como o XGBoost forma pares para aprender preferências
    * LambdaRank: como aproximar métricas de ranking (NDCG) via gradiente
    * Exemplo prático com um grupo de 3 documentos:
      * Construindo os pares
      * Calculando gradiente e hessiano
      * Ajustando os scores com 2 boosts
    * Diferença entre rank:pairwise e rank:ndcg

***

### **Parte IV – Interpretabilidade e Ajuste Fino**

11. **Interpretabilidade**
    * Importância de features: Gain, Cover, Frequency
    * Limitações dessas métricas internas
    * SHAP values no XGBoost (como calcular e interpretar)
    * Exemplos visuais: waterfall plots, force plots
12. **Eficiência e Implementação**
    * tree\_method: exact, hist, gpu\_hist
    * Como o XGBoost lida com dados esparsos
    * Handling de valores faltantes (branch default)
    * Comparação com LightGBM e CatBoost
13. **Validação e Treinamento**
    * Split treino/validação/teste no contexto de boosting
    * Early stopping: vantagens e riscos
    * Cross-validation: quando usar e quando não
    * Hiperparâmetros mais importantes para tuning (ordem de prioridade)

***

### **Parte V – Avançado**

14. **Custom objective e custom metric**
    * Como implementar sua própria função de perda
    * Exemplos práticos (quantile regression, fair loss, etc.)
    * Como definir métricas customizadas de avaliação
15. **Grandes volumes de dados**
    * Uso de GPU (gpu\_hist, gpu\_exact)
    * Técnicas de paralelização
    * Treinamento distribuído com dask/xgboost4j
16. **Conclusão**
    * Vantagens e limitações do XGBoost
    * Comparação com redes neurais e outros modelos modernos
    * Onde usar XGBoost e onde evitar
    * Futuro dos GBMs no ecossistema de ML
