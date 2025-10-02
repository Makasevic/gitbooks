# Intuição

## O que o modelo faz numa classificação?

**Ponto de partida (bias)**: o modelo começa com um “palpite neutro” em logit. Se o bias for 0, a probabilidade inicial fica próxima de 50%.

**Dividir para entender**: a árvore cria divisões sempre que isso reduz o erro. Por exemplo, o split “idade < 30” não é mágico; ele foi escolhido porque era o corte que mais ajudava naquele nó.

**Chegar à folha**: cada caminho termina em uma folha que indica quanto ajustar o logit.

* Folha positiva: observou mais compradores que o esperado → aumente a probabilidade.
* Folha negativa: observou menos compradores que o esperado → reduza a probabilidade.

**Intuição do sinal**: a folha compara observado versus esperado naquele grupo. Se observados > esperados, ajusta para cima; se observados < esperados, ajusta para baixo. O “−” na fórmula garante que o ajuste vá na direção de corrigir o erro.

## Por que os cortes são esses?

O modelo testa vários pontos em cada feature e escolhe o que mais melhora a previsão naquele momento (maior ganho). O primeiro split escolhe a melhor “chave”; nos ramos seguintes, escolhem-se as próximas “chaves”. É ganho de informação, não uma regra fixa.

## O que significa o valor da folha?

É o empurrão no logit do grupo.

* “+”: o grupo era mais comprador do que o modelo previa.
* “−”: o grupo era menos comprador do que o modelo previa.

A magnitude cresce com o erro observado versus esperado e é amortecida por incerteza, regularização e pelo *learning rate*.

## Onde entra o `pred_contribs` (o “quem empurrou”)?

A amostra percorre um caminho (splits) até cair em uma folha. O `pred_contribs` pega o efeito total da folha e o atribui às features que decidiram o caminho.

* Feature apareceu no caminho → recebe parte do crédito/culpa.
* Feature não apareceu → contribuição zero nessa árvore.

Somando todas as árvores, obtém-se a contribuição total por feature.

Há sempre uma coluna *bias*. Bias + soma das contribuições = logit final. Aplicando a sigmoide, obtemos a probabilidade do *predict*.

Não é a média dos pesos dos nós. É a lógica dos valores de Shapley para árvores (TreeSHAP): “quanto a predição esperada muda quando revelo esta feature no caminho?”. É uma atribuição justa por construção da árvore.

## Um único exemplo mental

* Bias = 0 (probabilidade inicial ~50%).
* Caminho: idade < 30 → salário < 5k → folha negativa.

Leitura: “entre jovens de baixa renda, veio menos comprador do que o esperado” → abaixe o logit.

O `pred_contribs` reparte essa queda entre idade e salário (splits do caminho); as demais features ficam com 0.

Identidade: bias + contribuições = logit final → sigmoide → probabilidade.

## Exemplo (com árvore)

Contexto: classificar “compra” (1) vs “não compra” (0). Bias = 0 (chute inicial ≈ 50%). Folhas em logit.

```
                     idade < 30?
                   /            \
             Sim  /              \  Não
                /                \
        salario < 5k?            FOLHA = +0.4
          /       \
        Sim       Não
   FOLHA = -0.5   FOLHA = +0.8
```

### Leitura das folhas

* +0.4 (idade ≥ 30): mais comprador que o esperado → sobe o logit.
* −0.5 (idade < 30 & salário < 5k): menos comprador que o esperado → desce o logit.
* +0.8 (idade < 30 & salário ≥ 5k): bem mais comprador → sobe bastante.

### Três caminhos típicos

**A) Jovem e baixa renda — idade = 25, salário = 4k → folha −0.5**

* Intuição: abaixar o logit.
* `pred_contribs` (ideia): idade = −0.2, salário = −0.3, demais = 0, bias = 0 → logit = −0.5 → probabilidade ≈ 38%.

**B) Jovem e renda alta — idade = 25, salário = 8k → folha +0.8**

* Intuição: elevar o logit.
* `pred_contribs`: idade = +0.3, salário = +0.5, demais = 0, bias = 0 → logit = +0.8 → probabilidade ≈ 69%.

**C) Mais velho — idade = 40, salário = 10k → folha +0.4**

* Intuição: elevar o logit.
* `pred_contribs`: idade = +0.4, salário = 0, bias = 0 → logit = +0.4 → probabilidade ≈ 60%.

Em todos os casos: bias + contribuições = logit; sigmoide(logit) = probabilidade do *predict* (sem `output_margin`).

## Em uma frase

O XGBoost aprende cortes que reduzem o erro, ajusta o palpite nas folhas conforme observado versus esperado e o `pred_contribs` contabiliza, por feature, quem ao longo do caminho empurrou a previsão para cima ou para baixo — garantindo que a soma bata exatamente o logit (que vira a probabilidade).
