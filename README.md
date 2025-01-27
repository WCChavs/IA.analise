# Previsão de Score de Crédito

Este projeto tem como objetivo prever o score de crédito de novos clientes utilizando Inteligência Artificial. A partir de uma base de dados contendo informações sobre clientes, como profissão, histórico de pagamento, e comportamento de crédito, o modelo de Machine Learning é treinado para classificar o score de crédito como **Boa**, **OK** ou **Ruim**.

## Passo a Passo

### Passo 0: Entender o Desafio da Empresa

O objetivo é prever o score de crédito de novos clientes com base em dados históricos. As empresas precisam dessas previsões para avaliar a probabilidade de um cliente pagar suas dívidas em dia. O modelo desenvolvido visa melhorar a análise de risco e otimizar a concessão de crédito.

### Passo 1: Importação da Base de Dados

Primeiro, a base de dados é carregada utilizando a biblioteca **Pandas**. A base contém informações dos clientes como profissão, comportamento de pagamento, e o histórico de crédito.

```bash
%pip install pandas
```

Após isso, a tabela é exibida para análise.

```python
import pandas as pd

# Carregar a base de dados
tabela = pd.read_csv(r"C:\Python-Projetos\Python-IA\clientes.csv")
display(tabela)
```

### Passo 2: Preparação da Base de Dados para IA

A base de dados precisa ser processada para que o modelo de IA consiga entender e trabalhar com as informações. Algumas colunas que contêm dados categóricos (texto) precisam ser convertidas para valores numéricos. Para isso, é utilizado o **LabelEncoder** da biblioteca **sklearn**.

```python
from sklearn.preprocessing import LabelEncoder

# Codificar a coluna 'profissao'
codificador_profissao = LabelEncoder()
tabela["profissao"] = codificador_profissao.fit_transform(tabela["profissao"])

# Codificar outras colunas categóricas
codificador_credito = LabelEncoder()
tabela["mix_credito"] = codificador_credito.fit_transform(tabela["mix_credito"])

codificador_pagamento = LabelEncoder()
tabela["comportamento_pagamento"] = codificador_pagamento.fit_transform(tabela["comportamento_pagamento"])

display(tabela.info())
```

### Passo 3: Treinamento do Modelo de Inteligência Artificial

Agora que a base de dados está pronta, dividimos os dados em **treinamento** e **teste**. Com os dados de treinamento, o modelo é treinado para prever o score de crédito.

Dois modelos são testados: **RandomForestClassifier** (árvore de decisão) e **KNeighborsClassifier** (vizinhos próximos).

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# Separar dados em variáveis independentes (x) e dependentes (y)
y = tabela["score_credito"]
x = tabela.drop(columns=["score_credito", "id_cliente"])

# Separar em treino e teste
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3)

# Criar e treinar os modelos
modelo_arvoredecisao = RandomForestClassifier()
modelo_knn = KNeighborsClassifier()

modelo_arvoredecisao.fit(x_treino, y_treino)
modelo_knn.fit(x_treino, y_treino)
```

### Passo 4: Avaliação do Melhor Modelo

Agora, os modelos são avaliados com base na **acurácia**, utilizando os dados de teste. O modelo com a melhor performance é selecionado.

```python
from sklearn.metrics import accuracy_score

# Fazer previsões com os modelos
previsao_arvoredecisao = modelo_arvoredecisao.predict(x_teste)
previsao_knn = modelo_knn.predict(x_teste)

# Calcular a acurácia
display(accuracy_score(y_teste, previsao_arvoredecisao))
display(accuracy_score(y_teste, previsao_knn))
```

### Passo 5: Previsão para Novos Clientes

O modelo com a melhor acurácia é utilizado para prever o score de crédito de novos clientes. Para isso, os novos dados precisam ser tratados da mesma forma que os dados de treinamento, ou seja, as colunas categóricas precisam ser codificadas.

```python
# Carregar a base de dados de novos clientes
tabela_novos_clientes = pd.read_csv("novos_clientes.csv")

# Codificar as colunas categóricas
tabela_novos_clientes["profissao"] = codificador_profissao.transform(tabela_novos_clientes["profissao"])
tabela_novos_clientes["mix_credito"] = codificador_credito.transform(tabela_novos_clientes["mix_credito"])
tabela_novos_clientes["comportamento_pagamento"] = codificador_pagamento.transform(tabela_novos_clientes["comportamento_pagamento"])

# Fazer as previsões para os novos clientes
nova_previsao = modelo_arvoredecisao.predict(tabela_novos_clientes)
display(nova_previsao)
```

### Conclusão

O modelo selecionado (no caso, o **RandomForestClassifier**) pode agora ser utilizado para prever o score de crédito de novos clientes com base nas informações fornecidas.

## Tecnologias Utilizadas

- **Pandas**: Para manipulação de dados e leitura dos arquivos CSV.
- **Scikit-learn**: Para a implementação dos modelos de Machine Learning e pré-processamento dos dados.
- **Python**: Linguagem de programação utilizada para desenvolver o código.

## Como Rodar

1. Instale as dependências:
   ```bash
   pip install pandas scikit-learn
   ```

2. Execute o script Python que contém o código.

3. Certifique-se de que os arquivos CSV (`clientes.csv` e `novos_clientes.csv`) estão no caminho correto.

4. Acesse os resultados das previsões e acurácia diretamente no terminal ou interface Python.

## Melhorias Futuras

- **Validação Cruzada**: Implementar validação cruzada para melhorar a avaliação dos modelos.
- **Otimização de Parâmetros**: Realizar uma busca de hiperparâmetros (Grid Search) para encontrar a melhor configuração para os modelos.
- **Explicabilidade do Modelo**: Utilizar técnicas como SHAP ou LIME para explicar as previsões do modelo.

## Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para mais detalhes.
