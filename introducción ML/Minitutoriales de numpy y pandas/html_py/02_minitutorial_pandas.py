#!/usr/bin/env python
# coding: utf-8

# # Mini-tutorial de Pandas
# **Por Ignacio Díaz Blanco. Área de Ingeniería de Sistemas y Automática. Universidad de Oviedo.**
# 
# En este mini-tutorial, se muestran algunas operaciones y funcionalidades útiles de Pandas. Pandas es una librería extensísisma para importación, tratamiento, filtrado, análisis y representación de datos en formato matricial (tipo tabla de Excel), incluyendo operaciones de agrupación y agregación multivía propias de cubos de datos. Pandas es, además, una herramienta fantástica para tareas como la importación de datos, así como para la gestión de fechas y datos faltantes, auténticos quebradero de cabeza si se abordan sin librerías.
# 
# La documentación oficial de Pandas http://pandas.pydata.org/pandas-docs/stable/, en el momento de redactar esto, tiene en torno a 1500 páginas, por lo que es virtualmente imposible cubrir todas las funciones de esta potente herramienta. Sin embargo, la forma de trabajar con ella es bastante intuitiva y basada en operaciones de alto nivel. El enfoque recomendado para trabajar con Pandas es ponerse "manos a la obra" partiendo de ejemplos sencillos siguiendo un tutorial (este es un ejemplo, pero hay otros muchos en internet) y consultando la documentación cuando se desee realizar una operación que no se conoce o no está cubierta en los tutoriales.

# ## Ejemplo básico
# Empezaremos con un ejemplo básico, tratando una tabla de datos aleatorios, de 10 registros (filas) con 3 atributos o variables (columnas)

# In[1]:


# IMPORTACIÓN DE MÓDULOS
# importamos el módulo de pandas y también pylab
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime

# con esto, las gráficas de matplotlib se muestran en linea en el notebook
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# creación de un DataFrame básico
df = pd.DataFrame(np.random.randn(10,3), columns=['x','y','valor'])

# generamos un array de 10 fechas (timestamps)
t = pd.date_range('2016/02/25', periods=10, freq='2h')

# asignamos un índice de timestamps
df.index = t


# In[3]:


# MOSTRAMOS EL DATAFRAME COMPLETO
df


# In[4]:


# FORMAS RESUMIDAS DE MOSTRAR EL DATAFRAME
# df.head() permite ver solo los primeros registros del dataframe 
df.head()


# In[5]:


# SELECCIÓN DE REGISTROS
# mostrar sólo los datos de las 4 de la mañana
df[df.index.hour == 4]


# In[6]:


# mostrar sólo los datos con valor > 0.5
df[df['valor']>0.5]


# In[7]:


# Resumen del dataframe con unos pocos estadísticos descriptivos
df.describe()


# In[8]:


# generar gráficas del dataframe
import matplotlib.pyplot as plt
df.plot()


# In[9]:


plt.figure(figsize=(12,5))
plt.subplot(1,3,1)
df['x'].plot(kind='bar',title='barras verticales')

plt.subplot(1,3,2)
df['x'].plot(kind='barh',title='barras horizontales')

plt.tight_layout()


# In[10]:


# DESCRIPTORES ESTADÍSTICOS (media, desviación típica, etc.)

# medias (por columnas)
df.mean()


# In[11]:


# lo mismo puede hacerse con la keyword "axis": 0=columnas, 1=filas
df.mean(axis=0)


# In[12]:


#medias (por filas)
df.mean(axis=1)


# In[13]:


# aplicar una función a cada dato"
df.apply(np.sin)


# In[14]:


# aplicar una función a cada columna (por defecto)
df.apply(np.sum)


# In[15]:


# aplicar una función a cada columna usando axis=0 (sale lo mismo que antes)
df.apply(np.sum,axis=0)


# In[16]:


# aplicar una función a cada fila
df.apply(np.sum,axis=1)


# In[17]:


# ordenar el DataFrame por una columna concreta

df.sort_values(by='x')

# otra foram (aunque parece que quedará "obsoleta")
# df.sort(columns='x')


# In[18]:


# ... y dibujarla
plt.figure(figsize=(10,10))
plt.plot(df.sort_values(by='x').values)
plt.legend(df.columns)


# ## Importar datos (archivo 

# In[19]:


# IMPORTACIÓN DE LOS DATOS

df = pd.read_csv('../../datos/Activa_columna.csv')
df


# In[20]:


# ASIGNACIÓN DE FECHAS: el contexto temporal es muy importante

# CREAMOS UN RANGO DE FECHAS (timestamps)
# Sabemos lo siguiente de nuestro fichero *.csv:
# fecha inicial: 1 de marzo de 2010
# número de datos: 262800 (un año entero de datos, a intervalos de 2 minutos)
t = pd.date_range('2010-03-01',periods=262800,freq='2min')


# lo asignamos al índice del dataframe
df.index = t

# creamos "helper columns": categorizan los datos
df['diaSemana'] = df.index.weekday
df['hora'] = df.index.hour
df['mes'] = df.index.month


# mostramos los 10 primeros registros
df.head(10)


# In[21]:


# REMUESTREO (resample)
# remuestreamos a intervalos 1 hora, de los que tomamos la media como valor agregado
dfr = df.resample('1h').mean()


# mostramos los 10 primeros registros (ahora el periodo de muestreo es 1 hora)
dfr.head(10)


# In[22]:


# Dibujamos la potencia activa
plt.figure(figsize=(15,5))
dfr['Demanda'].plot()


# In[23]:


# Dibujamos la potencia activa, solo de octubre (observar, por ejemplo, la fiesta del 12 de Octubre)
plt.figure(figsize=(15,5))
dfr['Demanda']['2010-10'].plot()


# In[24]:


# ... idem para un rango de fechas: de junio a septiembre (inclusive)
plt.figure(figsize=(15,5))
dfr['Demanda']['2010-06-01':'2010-09-30'].plot()

# ... Pandas, gestiona adecuadamente los datos faltantes o erróneos: 
# simplemente, no se dibujan... no se cuentan al hacer medias, etc.


# In[25]:


# AGRUPACIÓN+AGREGACIÓN (groupby+aggfun)

# Agregación por horas tipo "mean()"
# permite obtener potencia media consumida en el año para cada hora

# creamos un objeto especial que contiene la "agrupación" por horas
dfg = df.groupby('hora')

# ahora aplicamos una agregación "media" para cada grupo (debe devolvernos 24 valores)
dfg['Demanda'].mean()


# In[26]:


# Podemos hacer el "groupby" y la agregación en una sola instrucción, y además dibujarlas

plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
df.groupby('hora').mean()['Demanda'].plot(kind='bar')

plt.subplot(1,3,2)
df.groupby('diaSemana').mean()['Demanda'].plot(kind='bar')

plt.subplot(1,3,3)
df.groupby('mes').mean()['Demanda'].plot(kind='bar')


# In[27]:


# PIVOT TABLE (tabla pivote)

# Una tabla pivote implica 
#
# 1) una agrupación 2D (o más dimensiones) de los datos conforme 
#    a variables distintas (por ejemplo, "dia de la semana" y "hora"). 
#    Cada casilla de la tabla describe una combinación de las dos variables de agrupación 
#    (ej. "lunes a las 10:00" o bien "martes a las 20:00") 
#
# 2) La agregación de todos los registros vinculados a cada casilla (media, suma, máximo, etc.)


# creamos una tabla pivote de horas vs. dia de la semana
# una tabla pivote es una agregación por más de una variable
dfp = df.pivot_table(columns='hora',index='diaSemana')['Demanda']
dfp


# In[28]:


# La tabla pivote aporta una información muy interesante y elocuente 
# sobre el comportamiento general de los datos (en este caso, la demanda eléctrica)


# In[29]:


# Una forma interesante de representar la tabla pivote es mediante una visualización "heatmap" ("mapa de calor")
plt.figure(figsize=(12,5))
plt.title(u'Pivot table: medias por (horas,días)')
plt.imshow(dfp,interpolation='none')
horas = range(24)
etiqhoras = map(lambda x: str(x)+'h',horas)
dias = range(7)
etiqdias = ['Lu','Ma','Mi','Ju','Vi','Sa','Do']
plt.xticks(horas,etiqhoras,rotation=90)
plt.yticks(dias,etiqdias)
plt.colorbar()
plt.draw()
plt.show()


# ## Filtrado
# Todos los análisis anteriores podemos aplicarlos a un subconjunto de los datos, generalmente tomando una "rodaja" (*slice*) en una de las variables. Por ejemplo podemos crear un nuevo dataframe que contenga solamente los datos del mes de abril y hacer sobre él todas las operaciones anteriores (medias por horas, días de la semana, etc.) que permitirán describir, lógicamente,  el comportamiento de la demanda en dicho mes. Esta operación de selección de un subconjunto de los datos suele denominarse *filtrado*.

# In[30]:


# Ejemplo: calcular y visualizar el consumo promedio por horas durante el mes de abril
plt.figure(figsize=(12,5))
consumo_por_horas_en_abril = df.loc['2010-04'].groupby('hora').mean()['Demanda']

print(consumo_por_horas_en_abril)

dfg = df.loc['2010-04'].groupby('hora')
plt.stackplot(dfg.groups.keys(),dfg.mean()['Demanda'])


# ### Filtrado iterativo, mes a mes
# En el siguiente ejemplo realizamos un bucle que itere por meses y que muestre los perfiles de consumo medio horario correspondientes a cada mes

# In[31]:


# FILTRADO MES A MES

# consultar la página https://docs.python.org/2/library/datetime.html 
# formatos de fechas

# generamos 12 timestamps: uno por mes, empezando por el timestamp del primer registro (1 de marzo de 2010), 
# generando un total de 12 timestamps y seleccionando intervalos de 1 mes
t = pd.date_range('2010-03-01',periods=12,freq='1ME')

# generamos una lista de cadenas con los nombres de los meses a partir de los timestamps anteriores
# utilizamos los códigos estándar de %Y (=año con 4 cifras) y %m (= mes con 2 cifras)... hay más códigos 
# de estos... pueden consultarse en internet.
tstring = [datetime.datetime.strftime(i,'%Y/%m') for i in t]


# dibujamos los patrones de consumo por meses
plt.figure(figsize=(15,10))

for i,mes in enumerate(tstring):
    plt.subplot(3,4,i+1)

    # filtramos el dataFrame: nos quedamos solo con el mes en curso
    dfMes = df.loc[mes]

    # agregamos (medias) por dias de la semana... y dibujamos gráfica de barras
    dfMes.groupby('diaSemana').mean()['Demanda'].plot(kind='bar')

    # muy importante: igualamos la escala "y" en todos los subplots para hacerlos comparables
    plt.axis(ymax=100)
    plt.xticks(dias,etiqdias)
    plt.title(t[i].strftime('%B'))
    plt.xlabel('')
    plt.ylabel('Potencia activa (kW)')

plt.tight_layout()


# Esto son datos reales de consumo. Se observan patrones reconocibles, como el menor consumo en fin de semana.

# In[32]:


get_ipython().run_cell_magic('html', '', '<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Licencia de Creative Commons" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br /><span xmlns:dct="http://purl.org/dc/terms/" property="dct:title">Mini-tutorial de Pandas</span> by <a xmlns:cc="http://creativecommons.org/ns#" href="http://isa.uniovi.es/~idiaz" property="cc:attributionName" rel="cc:attributionURL">Ignacio Díaz Blanco</a> is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Reconocimiento-CompartirIgual 4.0 Internacional License</a>.\n')

