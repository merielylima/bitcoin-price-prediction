#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
    Aluna: Meriely Eline Lima Gaia
    Matrícula: 201633840047

Para realização deste trabalho foi utilizado o ambiente computacional Jupyter Notebook e as seguintes configurações
    - Ubuntu 20.04.1 LTS   
    - Python 3.7
        numpy 1.19.2
        pandas 1.1.3
        sklearn 0.24.1
        plotly 4.14.3
Foram utilizados os seguintes métodos preditivos:
    1. Árvore de Decisão 
    2. Regressão Linear

    Referências: 
        https://www.youtube.com/watch?v=hOLSGMEEwlI&ab_channel=ComputerScience
        https://minerandodados.com.br/como-criar-dashboards-em-python/
        https://minerandodados.com.br/analisando-dados-da-bolsa-de-valores-com-python/
        https://ichi.pro/pt/negociacao-algoritmica-em-python-medias-moveis-simples-8012299612427
        
'''
import warnings
warnings.filterwarnings('ignore') 


# In[2]:


# Importar bibliotecas
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import plotly.graph_objs as go
import plotly
import plotly.offline as py
from plotly.offline import plot, iplot
plotly.offline.init_notebook_mode(connected=True)
import heapq


# In[3]:


# armazenar os dados em um dataframe
df1 = pd.read_csv('BTC-USD.csv')
df1.head() # mostra os pprimeiros 5 valores


# In[4]:


'''
Gráfico de Candlestick: Cada vela contém os valores de abertura, alta, baixa e fechamento.
Quando o valor de fechamento da ação é menor que o valor de abertura temos uma vela vermelha, 
se o valor fechar acima do preço de abertura a vela é verde. 
'''
# Visualizar comportamento através do gráfico de candlestick
fig = go.Figure(data=[go.Candlestick(x=df1['Date'],
                open=df1['Open'],
                high=df1['High'],
                low=df1['Low'],
                close=df1['Close'])])

#fig.update_layout(xaxis_rangeslider_visible=False)
fig.update_layout(
    title = 'Preço dos bitcoins durante os anos de 2018 à 2020',
    yaxis_title = 'Preço USD ($)',
    xaxis_title = 'Data')
fig.show()


# In[5]:


# Mudar nome das colunas e fechar dados para jan 2019 à outubro 2020
df1.columns = ['data', 'abertura', 'max', 'min', 'fechamento', 'ajus_fechamento', 'volume']
df1 = df1[0:725] 
df1.tail().round(2) # mostra os 5 ultimos valores


# In[6]:


#obter o número de dias de negociação
df1.shape[0]


# In[7]:


df1.describe()


# In[8]:


# Pegar apenas os preços de fechamento
df = df1[['fechamento']] # df é apenas a coluna fechamento
df.tail(4)


# In[9]:


# visualização do dados de preço de fechamento
fig_fechamento = {
    'x': df1.data,
    'y': df.fechamento,
    'type': 'scatter',
    'mode': 'lines',
    'line': {
        'width': 1,
        'color': 'blue'
    }
}
# informar todos os dados e gráficos em uma lista
data = [fig_fechamento]
 
# configurar o layout do gráfico
layout = go.Layout(title='Preço de fechamento dos bitcoins durante os anos de 2018 à 2020 ',

                   # Definindo exibicao dos eixos x e y
                   yaxis={'title':'Valor do Bitcoin', 
                          'tickformat':'.', 
                          'tickprefix':'$ '},
                   xaxis={'title': 'Dias',
                          })
 
# instanciar objeto Figure e plotar o gráfico
fig = go.Figure(data=data, layout=layout)
fig.show()


# In[10]:


# Criar uma variável para prever 'x' dias no futuro
dias_futuro = 31

# Criar uma nova coluna (alvo) deslocada 'x' unidades/dias para cima
df['predicao'] = df[['fechamento']].shift(-dias_futuro)# df é a coluna fechamento + a coluna predição
df.tail(4) 


# In[11]:


#criar a feature dos dados (X) e converta-la em uma matriz numpy + remover as últimas 'x' linhas/dias

X = np.array(df.drop(['predicao'], 1))[:-dias_futuro]
print("Tamanho depois da remoção do dias", X.shape)
X.round(2)


# In[12]:


#criar o conjunto de dados de destino (y) e converte-lo em uma matriz numpy 
# obter todos os valores do alvo (y), exceto as últimas 'x' linhas/dias
y = np.array(df['predicao'])[:-dias_futuro]
y.round(2)


# In[13]:


# Dividir (split) os dados em 75% para treinamento e 25% para teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)


# In[14]:


X_train.shape


# In[15]:


X_test.shape


# In[16]:


# ---------- Criar os modelos ---------- # 

#Criar o modelo de regressão da árvore de decisão
arvore = DecisionTreeRegressor().fit(X_train, y_train)

#Criar o modelo de regressão linear

linear = LinearRegression().fit(X_train, y_train)


# In[17]:


#testando o modelo a fim retornar a acurácia da previsão
print("Acurácia do modelo de arvore de decisão (R quadrado) = {}".format(arvore.score(X_test, y_test)))
print("Acurácia do modelo de regressão linerar (R quadrado) = {}".format(linear.score(X_test, y_test)))


# In[18]:


# obter as últimas 'x' linhas do conjunto de dados
x_futuro = df.drop(['predicao'], 1)[: -dias_futuro]
x_futuro = x_futuro.tail(dias_futuro)
x_futuro.head()


# In[19]:


# converte-lo em uma matriz numpy 
x_futuro = np.array(x_futuro)
print("Tamanho: ", x_futuro.shape)
x_futuro.round(2)


# In[26]:


# Mostrar o modelo de regressão linear 
predicao_linear = linear.predict(x_futuro)
print("\nVetor com os valores do modelo de regressão linear: \n",predicao_linear.round(2))

# Mostrar o modelo de arvores de decisão
predicao_arvore = arvore.predict(x_futuro)
print("\nVetor com valores do modelo de arvores de decisão: \n",predicao_arvore.round(2))


# In[31]:


num_max_1 = np.argpartition(predicao_arvore, -5)[-5:]
print("1. ARVORE DE DECISÃO - OUTUBRO 2020\n")
print(" ----- MÁXIMO -----")
print("Nos dias {}".format(num_max_1), "o preço do bitcoin estará mais valorizado.")
for i in range(4):
    index = num_max_1[i]
    valor = predicao_arvore[index]
  
    print("No dia",num_max_1[i], "o valor será de: $", valor.round(2))
    
num_min_1 = np.argpartition(predicao_arvore, 5)
print("\n ---- MÍNIMO -----")
print("Nos dias ",num_min_1[0:4], "o preço do bitcoin estará mais desvalorizado.")
for i in range(4):
    index = num_min_1[i]
    valor = predicao_arvore[index]
  
    print("No dia",num_min_1[i], "o valor será de: $", valor.round(2))


# In[32]:


num_max_2 = np.argpartition(predicao_linear, -5)[-5:]
print("2. REGRESSÃO LINEAR - OUTUBRO 2020\n")
print(" ----- MÁXIMO -----")
print("Nos dias",num_max_2, "o preço do bitcoin estará mais valorizado.")
for i in range(4):
    index = num_max_2[i]
    valor = predicao_linear[index]
  
    print("No dia",num_max_2[i], "o valor será de: $", valor.round(2))
 
num_min_2 = np.argpartition(predicao_linear, 5)
print("\n ---- MÍNIMO -----")
print("Nos dias ",num_min_2[0:4], "o preço do bitcoin estará mais desvalorizado.")
for i in range(4):
    index = num_min_2[i]
    valor = predicao_linear[index]
  
    print("No dia",num_min_2[i], "o valor será de: $", valor.round(2))


# In[23]:


# criar variável para mostrar apenas os valores referentes ao mes de outubro de 2020
ultimas_datas = df1.data[694:725]
ultimas_datas.to_frame().head()


# In[24]:


# Visualizar os dados obtidos de forma iterativa a partir do modelo de Regressão Linear

validacao_linear = df[X.shape[0]:]
validacao_linear['predicoes_linear'] = predicao_linear
original = {
    'x': ultimas_datas,
    'y': validacao_linear.fechamento,
    'type': 'scatter',
    'mode': 'lines',
    'line': {
        'width': 1,
        'color': 'blue'
    },
    'name': 'Original'
}
predicao = {
    'x': ultimas_datas,
    'y': validacao_linear.predicoes_linear,
    'type': 'scatter',
    'mode': 'lines',
    'line': {
        'width': 1,
        'color': 'green'
    },
    'name': 'Predição'
}

# informar todos os dados e gráficos em uma lista
data = [original, predicao]
 
# configurar o layout do gráfico
layout = go.Layout(title='Modelo de Regressão Linear',

                   # Definindo exibicao dos eixos x e y
                   yaxis={'title':'Valor de Fechamento USD($)', 
                          'tickformat':'.', 
                          'tickprefix':'$ '},
                   xaxis={'title': 'Dias',
                          })
 
# instanciar objeto Figure e plotar o gráfico
fig = go.Figure(data=data, layout=layout)
fig.show()


# In[25]:


## Visualizar os dados obtidos de forma iterativa a partir do modelo de arvore de decisão

validacao_arvore = df[X.shape[0]:]
validacao_arvore['predicoes_arvore'] = predicao_arvore

original = {
    'x': ultimas_datas,
    'y': validacao_arvore.fechamento,
    'type': 'scatter',
    'mode': 'lines',
    'line': {
        'width': 1,
        'color': 'blue'
    },
    'name': 'Original'
}
predicao = {
    'x': ultimas_datas,
    'y': validacao_arvore.predicoes_arvore,
    'type': 'scatter',
    'mode': 'lines',
    'line': {
        'width': 1,
        'color': 'green'
    },
    'name': 'Predição'
}

# informar todos os dados e gráficos em uma lista
data = [original, predicao]
 
# configurar o layout do gráfico
layout = go.Layout(title='Modelo de Arvore de Decisão',

                   # Definindo exibicao dos eixos x e y
                   yaxis={'title':'Valor de Fechamento USD($)', 
                          'tickformat':'.', 
                          'tickprefix':'$ '},
                   xaxis={'title': 'Dias',
                          })
 
# instanciar objeto Figure e plotar o gráfico
fig = go.Figure(data=data, layout=layout)
fig.show()

