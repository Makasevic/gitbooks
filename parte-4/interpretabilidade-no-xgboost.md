# Interpretabilidade no XGBoost

## 11.1 Importância de features: Gain, Cover e Frequency

O XGBoost armazena, durante o treino, estatísticas internas que permitem avaliar a importância relativa de cada feature. As três métricas mais usadas são:

* **Gain (ganho de informação):**\
  Mede o quanto uma feature contribuiu para reduzir a função de perda. Sempre que um split é feito com uma feature, calcula-se o _gain_ obtido. A soma total dos ganhos dessa feature em todas as árvores define sua importância.\
  👉 É a métrica mais usada, pois conecta diretamente com a melhoria da loss.
* **Cover (abrangência):**\
  Mede quantas observações (ou “peso total”) passaram por splits feitos com aquela feature. Se uma feature aparece em nós que cobrem muitas amostras, terá Cover alto.\
  👉 Relaciona-se à “popularidade” da feature na árvore.
* **Frequency (frequência):**\
  Conta o número de vezes que a feature foi usada em splits ao longo do modelo.\
  👉 Simples, mas pode ser enganosa: uma feature pode aparecer várias vezes, mas em splits pouco relevantes.

📌 Exemplo intuitivo:\
Se estivermos prevendo preço de imóveis, é possível que **metragem** apareça pouco, mas com _gain_ alto (split decisivo), enquanto **bairro** apareça muitas vezes, mas com _gain_ menor.

***

## 11.2 Limitações dessas métricas internas

Embora úteis, essas métricas têm limitações importantes:

* **Bias por granularidade:** Features contínuas com muitos valores possíveis tendem a ser escolhidas mais vezes do que features categóricas discretas.
* **Interação entre features:** Se duas variáveis são altamente correlacionadas, o XGBoost pode usar apenas uma delas — a outra parecerá “pouco importante” mesmo sendo relevante.
* **Dificuldade de interpretar Cover/Frequency:** Elas não dizem diretamente o quanto a feature melhora a performance do modelo, apenas como foi usada.

👉 Conclusão: métricas internas ajudam, mas não devem ser a única forma de avaliar importância de variáveis. É por isso que surgiram abordagens mais robustas, como o **SHAP**.

***

## 11.3 SHAP values no XGBoost

O SHAP (SHapley Additive exPlanations) traz uma abordagem baseada na **Teoria dos Jogos de Shapley**.

* **Ideia:** Cada feature é vista como um “jogador” que contribui para a predição.
* O valor SHAP de uma feature é a média ponderada de sua contribuição marginal em todos os subconjuntos possíveis de features.

📌 Isso garante três propriedades desejáveis:

1. **Eficiência:** A soma das contribuições é igual à predição do modelo.
2. **Simetria:** Features com impacto igual recebem a mesma importância.
3. **Consistência:** Se o modelo muda para valorizar mais uma feature, o SHAP não diminui sua importância.

### TreeSHAP

O cálculo exato dos valores de Shapley seria inviável em árvores grandes (precisaria avaliar todas as combinações de features).\
O algoritmo **TreeSHAP** permite calcular SHAP values de forma eficiente em modelos baseados em árvores, como o XGBoost.

***

## 11.4 Exemplos visuais

* **Waterfall plot:**\
  Mostra como cada feature empurra a predição para cima ou para baixo, em relação a uma base (valor médio).
* **Force plot:**\
  Visualização interativa em que cada feature é representada como uma “força” que empurra o score final em direção positiva ou negativa.
* **Summary plot (beeswarm):**\
  Mostra a distribuição dos valores SHAP de cada feature ao longo de todas as observações.\
  👉 Permite identificar não só a importância média de cada variável, mas também se ela atua de forma linear ou não.

📌 Exemplo prático:

* Em um modelo de crédito, o SHAP pode mostrar que **renda** tem impacto positivo (aumenta chance de aprovação) e **inadimplência passada** impacto negativo.
* Mais importante: é possível ver o impacto **para cada indivíduo**, permitindo explicações personalizadas.

***

## 11.5 Comparando métricas internas e SHAP

* **Métricas internas (Gain, Cover, Frequency):** rápidas, fáceis, dão ideia geral.
* **SHAP:** mais lento, mas fornece interpretações locais e globais consistentes.

👉 Na prática:

* Use métricas internas para **depuração rápida** e seleção preliminar de features.
* Use SHAP para **explicações finais** e quando precisar de confiança interpretativa (auditoria, relatórios, clientes).

***

✅ **Resumo do Capítulo 11:**

* O XGBoost gera métricas internas de importância: Gain, Cover e Frequency.
* Essas métricas ajudam, mas têm limitações — podem enganar em variáveis correlacionadas ou muito granulares.
* O SHAP values trazem uma visão sólida baseada em Teoria dos Jogos, calculada de forma eficiente pelo TreeSHAP.
* Visualizações como waterfall, force e summary plots permitem interpretar predições individuais e globais.
* Conclusão: combine métricas internas (rápidas) e SHAP (robusto) para interpretar o XGBoost com confiança.
