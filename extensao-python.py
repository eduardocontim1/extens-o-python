import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose

# Configurar estilo dos gráficos
sns.set(style="whitegrid")

# Carregar os dados de vendas
# Suponha que os dados estejam em um arquivo CSV com colunas 'Data' e 'Vendas'
data = pd.read_csv('vendas.csv', parse_dates=['Data'], index_col='Data')

# Verificar as primeiras linhas do dataframe
print(data.head())

# Resample para obter dados mensais se não estiverem nesse formato
data_monthly = data.resample('M').sum()

# Analisar padrões sazonais
result = seasonal_decompose(data_monthly['Vendas'], model='additive')

# Plotar os componentes
plt.figure(figsize=(12, 8))

plt.subplot(4, 1, 1)
plt.plot(data_monthly['Vendas'], label='Vendas Mensais')
plt.legend(loc='upper left')
plt.title('Vendas Mensais')

plt.subplot(4, 1, 2)
plt.plot(result.trend, label='Tendência', color='orange')
plt.legend(loc='upper left')
plt.title('Componente de Tendência')

plt.subplot(4, 1, 3)
plt.plot(result.seasonal, label='Sazonalidade', color='green')
plt.legend(loc='upper left')
plt.title('Componente Sazonal')

plt.subplot(4, 1, 4)
plt.plot(result.resid, label='Resíduos', color='red')
plt.legend(loc='upper left')
plt.title('Componente Residual')

plt.tight_layout()
plt.show()

# Ajustar e treinar um modelo ARIMA para prever as vendas
from statsmodels.tsa.arima.model import ARIMA

# Dividir os dados em treino e teste
train = data_monthly['Vendas'][:'2023-06']
test = data_monthly['Vendas']['2023-07':]

model = ARIMA(train, order=(1, 1, 1))
model_fit = model.fit()

# Fazer previsões
forecast = model_fit.forecast(steps=len(test))

# Plotar previsões
plt.figure(figsize=(10, 6))
plt.plot(train, label='Treino')
plt.plot(test, label='Teste', color='orange')
plt.plot(test.index, forecast, label='Previsão', color='green')
plt.legend(loc='upper left')
plt.title('Previsão de Vendas')
plt.show()

# Exportar previsões para um arquivo CSV
forecast_df = pd.DataFrame({'Data': test.index, 'Previsão': forecast})
forecast_df.to_csv('previsoes_vendas.csv', index=False)
