#!/usr/bin/env python
# coding: utf-8

# # Regresión multivariable
# *Ignacio Díaz Blanco, Universidad de Oviedo, 2023*
# 
# 
# Ejemplo básico de regresión multivariable utilizando el método `Ridge()` de `scikit-learn`
# 

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# ## Generación de datos de ejemplo
# 
# Generamos datos de ejemplo de un modelo con tres variables independientes $x_1, x_2, x_3$, un término afín independiente $b$
# $$
# y = a_1 x_1 + a_2 x_2 + a_3 x_3 + b + \epsilon
# $$
# al cual se le ha añadido ruido de distribución normal $\epsilon$ que representa la incertidumbre en los datos (ej. ruidos en los sensores)

# In[2]:


N = 1000

x1 = np.random.randn(N)
x2 = np.random.randn(N)
x3 = np.random.randn(N)

# ruido del sensor y
epsilon = 0.2*np.random.randn(N)

[a1, a2, a3, b] = [1.5, -2.1, 3.2, 0.7]

X = np.column_stack((x1,x2,x3))
y = a1*x1 + a2*x2 + a3*x3 + b + epsilon

# visualizamos los datos en una tabla
df = pd.DataFrame(np.column_stack((X,y)),columns=('x1','x2','x3','y'))
print(df)


# ## Aprendizaje del modelo utilizando scikit-learn

# In[3]:


from sklearn.linear_model import Ridge

# creamos el modelo
modelo = Ridge(alpha=0.0001)

# ajustamos el modelo a los datos
modelo.fit(X,y)

# imprimimos los parámetros del modelo
print(f'         coeficientes  =  {modelo.coef_}')
print(f'término independiente  =  {modelo.intercept_}')


# los coeficientes y el término independiente se aproximan bastante bien a los reales

# ### Inferencia del modelo con datos de test

# In[4]:


Ntest = 10

# datos de test para las tres variables independientes (sensores)
x1test = np.random.randn(Ntest)
x2test = np.random.randn(Ntest)
x3test = np.random.randn(Ntest)

# valores que daría el modelo ideal
ytest = a1*x1test + a2*x2test + a3*x3test + b

# datos de entrada al modelo
Xtest = np.column_stack((x1test,x2test,x3test))

# predicción del modelo
ypred = modelo.predict(Xtest)

# diferencia entre el valor ideal y la predicción
error = ytest - ypred

# mostramos los datos en una tabla
df = pd.DataFrame(np.column_stack((ytest,ypred,error)),columns=['ytest','ypred','error'])
print(df)


# ## Aprendizaje del modelo con la expresión matricial
# 
# Estimamos el modelo utilizando la expresión
# 
# $$
# \hat {\mathbf W} = (\mathbf X^T \mathbf X - \lambda\mathbf I)^{-1}\mathbf X^T\mathbf Y
# $$
# 
# tomando la matriz de regresores los valores de $x$ con una columna extra de $1's$ que permite obtener el término independiente en el modelo 
#  
# $$
# \mathbf X = 
# \left(
# \begin{matrix}
# x^1_1 & x^1_2 & x^1_3 & 1 \\
# x^2_1 & x^2_2 & x^2_3 & 1 \\
# \vdots\\
# x^n_1 & x^n_2 & x^n_3 & 1 \\
# \end{matrix}
# \right)
# \qquad
# \mathbf Y = 
# \left(
# \begin{matrix}
# y^1 \\
# y^2 \\
# \vdots \\
# y^n \\
# \end{matrix}
# \right)
# \qquad 
# {\rm de\; forma\; que}
# \qquad
# \mathbf Y = \mathbf X\mathbf W
# \qquad
# {\rm donde}
# \qquad
# \mathbf W = \left[a_1, a_2, a_3, b\right]
# $$
# 

# In[5]:


alpha = 0.0001
X = np.column_stack((x1,x2,x3,np.ones(N)))
Y = np.column_stack((y,))
W = np.linalg.inv(X.T@X + alpha*np.eye(4))@X.T@Y

# podemos obtener los coeficientes del modelo
print('valores estimados: ')
print(f'coeficientes = {W[:-1].T}')
print(f'término independiente = {W[-1]}')


# ### Inferencia del modelo con datos de test

# In[6]:


# datos de entrada al modelo
Xtest = np.column_stack((x1test,x2test,x3test,np.ones(Ntest)))
ypred = Xtest@W

# diferencia entre el valor ideal y la predicción
error = ytest - ypred.ravel()

# mostramos los datos en una tabla
df = pd.DataFrame(np.column_stack((ytest,ypred,error)),columns=['ytest','ypred','error'])
print(df)


# sale exactamente igual.
# 

# <p class=""><a href="http://creativecommons.org/licenses/by-sa/4.0/"><img src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" style="border-width:0"></a><br><span>Ejemplo de Regresión multivariable</span> by <a href="http://isa.uniovi.es/~idiaz">Ignacio Díaz Blanco</a> is licensed under a <a href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Reconocimiento-CompartirIgual 4.0 Internacional License</a>.</p>
