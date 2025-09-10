# Medidas internas do modelo

## 7.1 Gain (ganho de informação)

O **Gain** mede o quanto um split reduz a função de perda.\
Sempre que o XGBoost avalia dividir um nó, ele calcula:

* A perda antes do split.
* A perda após dividir em dois nós filhos.
* O ganho é a diferença: quanto menor ficou o erro.

👉 Splits com maior _gain_ são preferidos.

***

## 7.2 Cover (abrangência)

O **Cover** mede o "peso" dos exemplos que passam por um nó.\
Diferente de árvores clássicas (onde seria só a contagem de exemplos), no XGBoost ele é definido como:

$$
\text{Cover} = \sum_{i \in \text{nó}} h_i
$$

onde (h\_i) é o hessiano (2ª derivada da loss) de cada exemplo.

👉 Isso significa que nós com maior _cover_ concentram mais "curvatura da loss", ou seja, mais informação útil para ajustar o modelo.

***

## 7.3 Weight (peso ótimo da folha)

Cada folha da árvore precisa ter um valor de saída (w), que será adicionado à predição dos exemplos que caírem nela.\
Esse valor não é arbitrário: ele é escolhido para minimizar a loss local.

A função objetivo aproximada na folha é:

$$
\text{Loss}_{leaf}(w) = \sum_{i \in \text{leaf}} g_i w + \tfrac{1}{2} \sum_{i \in \text{leaf}} h_i w^2 + \tfrac{1}{2} \lambda w^2
$$

onde:

* (g\_i) = gradiente (1ª derivada da loss).
* (h\_i) = hessiano (2ª derivada da loss).
* (\lambda) = regularização L2.

Derivando em relação a (w) e igualando a zero:

$$
w^* = - \frac{\sum g_i}{\sum h_i + \lambda}
$$

👉 Ou seja:

* O gradiente dá a direção da correção.
* O hessiano e a regularização controlam o tamanho do passo.

***

## 7.4 Por que o Cover é a soma dos hessianos?

Essa definição não é arbitrária. Ela vem diretamente da expansão de Taylor usada pelo XGBoost para aproximar a loss.

Na expansão de 2ª ordem:

$$
\ell(y_i, f(x_i) + \Delta f(x_i)) \approx \ell(y_i, f(x_i)) + g_i \cdot \Delta f(x_i) + \tfrac{1}{2} h_i (\Delta f(x_i))^2
$$

* O **gradiente** ((g\_i)) indica a direção da correção (positivo ou negativo).
* O **hessiano** ((h\_i)) indica a curvatura da loss, ou seja, o "peso de confiança" dessa direção.

Se usássemos só gradientes para medir _cover_, erros positivos e negativos poderiam se cancelar, mascarando a quantidade de informação real.\
Já os hessianos são sempre positivos e capturam a intensidade do ajuste.

Por isso:

* O _cover_ é definido como (\sum h\_i), refletindo a informação acumulada do nó.
* O **peso ótimo da folha** combina (\sum g\_i) e (\sum h\_i):

$$
w^* = - \frac{\sum g_i}{\sum h_i + \lambda}
$$

👉 Assim, temos uma interpretação clara:

* _Gain_ mede a melhoria da loss.
* _Cover_ mede a quantidade de informação (curvatura) em um nó.
* _Weight_ define o ajuste ótimo, equilibrando direção (gradiente) e confiança (hessiano + regularização).

***

✅ **Resumo do Capítulo 7:**

* _Gain_ escolhe os splits mais promissores.
* _Cover_ mede a importância estatística de um nó pela soma dos hessianos.
* _Weight_ define o valor ótimo da folha, usando gradiente e hessiano.
