#!/usr/bin/env python
# coding: utf-8

# Definição do problema

# Base de dados:
# Input (X): Temperatura
# Output (Y): Lucro diário em dólares

# Etapa 1: Importação das bibliotecas

# In[4]:


import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

print(tf.__version__)


# Etapa 2: Importação da base de dados
# 

# In[9]:


sales_df = pd.read_csv('SalesData.csv')


# In[10]:


sales_df


# In[11]:


sales_df.info()


# In[12]:


sales_df.describe()


# Etapa 3: Visualização da base de dados

# In[14]:


sns.scatterplot(x=sales_df['Temperature'], y=sales_df['Revenue'])
plt.show()


# Etapa 4: Criação das variáveis da base de dados

# In[15]:


X_train = sales_df['Temperature']
y_train = sales_df['Revenue']


# In[16]:


X_train.shape


# Etapa 5: Criação e construção do modelo

# In[17]:


model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units = 10, input_shape = [1]))
model.add(tf.keras.layers.Dense(units = 1))


# In[18]:


model.summary()


# In[19]:


model.compile(optimizer=tf.keras.optimizers.Adam(0.1), loss = 'mean_squared_error')


# In[20]:


epochs_hist = model.fit(X_train, y_train, epochs = 1000)


# Etapa 6: Avaliação do modelo

# In[21]:


epochs_hist.history.keys()


# In[24]:


plt.plot(epochs_hist.history['loss'])
plt.title('Model Loss Progress During Training')
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.legend(['Training Loss']);


# In[23]:


model.get_weights()


# In[26]:


# Previsões com o modelo treinado
temp = 5
temp_array = np.array([[temp]])  # Converter a entrada em um array NumPy 2D
revenue = model.predict(temp_array)
print('Revenue Predictions Using Trained ANN =', revenue)


# In[27]:


plt.scatter(X_train, y_train, color = 'gray')
plt.plot(X_train, model.predict(X_train), color = 'red')
plt.ylabel('Revenue [dollars]')
plt.xlabel('Temperature [degC]')
plt.title('Revenue Generated vs. Temperature @Ice Cream Stand');


# In[ ]:


Etapa 7: Confirmar os resultados usando sklearn


# In[28]:


X_train.shape


# In[29]:


X_train = X_train.values.reshape(-1,1)


# In[30]:


X_train.shape


# In[31]:


y_train = y_train.values.reshape(-1,1)


# In[32]:


y_train.shape


# In[33]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# In[34]:


regressor.coef_


# In[35]:


regressor.intercept_


# In[36]:


plt.scatter(X_train, y_train, color = 'gray')
plt.plot(X_train, regressor.predict(X_train), color = 'red')
plt.ylabel('Revenue [dollars]')
plt.xlabel('Temperature [degC]')
plt.title('Revenue Generated vs. Temperature @Ice Cream Stand');


# In[37]:


temp = 5
revenue = regressor.predict([[temp]])
print('Revenue Predictions Using Trained ANN =', revenue)


# In[ ]:




