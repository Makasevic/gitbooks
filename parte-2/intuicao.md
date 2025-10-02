# Intuição do XGBoost

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

## Como as próximas árvores corrigem os erros das anteriores

O XGBoost treina as árvores **sequencialmente**. Cada nova árvore enxerga onde o modelo atual está errando e tenta compensar.

1. **Calcular o erro atual:** depois de somar todas as árvores já treinadas, o algoritmo calcula o gradiente/hessiano (indicadores do erro) para cada observação.
2. **Aprender “onde ajustar”:** a próxima árvore é treinada com esses sinais. Ela procura splits que expliquem bem os resíduos — os lugares onde a previsão ficou muito acima ou abaixo do real.
3. **Aplicar com cuidado (learning rate):** o peso de cada folha é multiplicado pelo *learning rate* (shrinkage). Assim, mesmo que uma árvore proponha um ajuste grande, o modelo só adiciona uma fração controlada. Isso evita oscilações e permite refinar aos poucos.
4. **Repetir até estabilizar:** árvore após árvore, os erros vão diminuindo. Quando o ganho marginal fica pequeno (early stopping ou número máximo de árvores), o treino para.

📌 Intuição geral: **cada árvore dá um empurrãozinho extra na direção correta**, corrigindo os resíduos deixados pela soma das anteriores. O learning rate garante que esses empurrões sejam graduais.

💡 Quer ver como atribuir esse efeito final a cada feature? O capítulo de Interpretabilidade (Parte 4) detalha o uso de `pred_contribs`/TreeSHAP para decompor a predição.

## Um único exemplo mental

* Bias = 0 (probabilidade inicial ~50%).
* Caminho: idade < 30 → salário < 5k → folha negativa.

Leitura: “entre jovens de baixa renda, veio menos comprador do que o esperado” → abaixe o logit.

A soma das árvores anteriores + o ajuste dessa folha gera o novo logit. Se os erros persistirem, a próxima árvore analisará justamente esse grupo para fazer correções adicionais (em outra região do espaço ou refinando o mesmo caminho).

Identidade: bias + contribuições das árvores = logit final → sigmoide → probabilidade.

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
* Com poucas árvores, o logit final pode ficar ≈ −0.5 (probabilidade ≈ 38%). Se isso ainda não bastar para corrigir o erro, as próximas árvores podem criar novos splits focando nesse público para reduzir o resíduo remanescente.

**B) Jovem e renda alta — idade = 25, salário = 8k → folha +0.8**

* Intuição: elevar o logit.
* Após aplicar o learning rate, o logit pode ficar ≈ +0.8 (probabilidade ≈ 69%). Árvores futuras só vão mexer muito se ainda houver diferença entre previsão e observado para esse grupo.

**C) Mais velho — idade = 40, salário = 10k → folha +0.4**

* Intuição: elevar o logit.
* O logit tende a ficar ≈ +0.4 (probabilidade ≈ 60%). Se esse segmento continuar com erro sistemático, novas árvores podem criar splits adicionais (ex.: idade ≥ 60) para refinar o ajuste.

Em todos os casos: bias + soma das folhas das árvores = logit; sigmoide(logit) = probabilidade do *predict* (sem `output_margin`).

## Em uma frase

O XGBoost aprende cortes que reduzem o erro, ajusta o palpite nas folhas conforme observado versus esperado e, árvore após árvore (com *learning rate*), vai corrigindo resíduos até que o logit final esteja coerente com os dados — que depois vira probabilidade via sigmoide.
