#!/usr/bin/env python
# coding: utf-8

# # Minitutorial de numpy y matplotlib
# *Ignacio Díaz Blanco, 2019-2026. Universidad de Oviedo*
# 
# ## Introducción
# ### Python como herramienta de análisis de datos
# <p>Utilizaremos python. Algunas razones:</p>
# <ul>
# <li>Es <strong>software libre</strong>. Se puede descargar de forma gratuita y existen varias distribuciones (enthought python distribution, python-xy, anaconda, ...). Recomendamos especialmente Anaconda, de continuum analytics
# <ul>
# <li><a href="https://www.continuum.io">Página web de Continuum Analytics</a></li>
# <li><a href="https://www.anaconda.com/distribution/">Página de descargas de Anaconda</a></li>
# </ul></li>
# <li>Python tiene una enorme cantidad de <strong>módulos</strong> (librerías). Algunos de los que utilizaremos:
# <ul>
# <li><strong>scipy</strong>: engloba paquetes como numpy, matplotlib, etc. que permiten hacer computación numérica con una filosofía similar a matlab. La potencia de python como lenguaje y la adopción por una comunidad muy amplia y activa, ha originado una expansión tal, que supera a matlab en muchas facetas.</li>
# <li><strong>Pandas</strong>: muy potente para la etapa de preparación de datos y la extracción de características. Tiene innumerables opciones para importar datos, gestionar timestamps (fechas), remuestrear, imputar datos faltantes, fusión de tablas, etc. Además tiene una potente gama de funciones de agrupación y agregación que permiten extraer descriptores de un problema dado.</li>
# <li><strong>scikit-learn</strong> (sklearn): permite acceder a algoritmos avanzados de <em>machine learning</em> (regresión, interpolación, clasificación, manifold learning, redes neuronales, etc.) de una forma sencilla</li>
# </ul></li>
# <li>Python es superior a Matlab en varios aspectos:
# <ul>
# <li>El propio lenguaje python, es más avanzado que el de Matlab (dicts, listas, etc.)</li>
# <li>Visualización de datos (mayavi, matplotlib, otras)</li>
# <li>Preparación de datos (pandas)</li>
# <li>Existen módulos para prácticamente cualquier cosa (Matlab es más limitado)</li>
# </ul></li>
# </ul>
# 
# ### Filosofía de la asignatura
# <ul>
# <li><strong>El ecosistema de Python para análisis de datos es inmenso</strong> y no puede ser abordado en un curso</li>
# <li>El enfoque que adoptaremos será el de <strong>&quot;saber hacer&quot;</strong>, a partir de ejemplos</li>
# <li>Refrescaremos la <strong>programación básica en python</strong> (muchos ya lo habéis visto en cursos bajos):
# <ul>
# <li><a href="https://docs.python.org/2/tutorial/introduction.html">An informal introduction to python</a></li>
# <li><a href="http://docs.scipy.org/doc/numpy-dev/user/quickstart.html">numpy: quickstart tutorial</a></li>
# </ul></li>
# <li>Posteriormente, tras cada sesión de teoría en la que se exponen conceptos, haremos <strong>tutoriales</strong> e iremos implementando esas ideas en <strong>aplicaciones concretas a través de ejemplos</strong>.</li>
# <li>El objetivo es que vosotros seáis capaces de elaborar <strong>soluciones propias a partir de estos ejemplos</strong>.</li>
# </ul>
# 

# ## Tutorial guiado

# In[1]:


# coding: utf-8
# importamos los módulos que vamos a utilizar (python tiene cientos)
import numpy as np
import matplotlib.pyplot as plt

# comando "mágico" específico para notebooks de Python,
# permite que las figuras se coloquen en línea, en el propio documento
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# creación de un vector (tensor de orden 1 [una matriz sería un tensor de orden 2, etc.])
v = np.array([1,2,3])
v


# In[3]:


# podemos imprimirlo en consola
print("vector: ", v)


# In[4]:


# con "shape", obtenemos su tamaño
print("tamaño del vector: ", v.shape)


# In[5]:


# copiar el vector creando otro objeto nuevo (reservando memoria para él)
w = v.copy()

# ¡Ojo!: si hiciéramos w=v (sin usar .copy()) no haríamos una copia, sino una referencia.
# En este caso, un cambio en v repercutiría automáticamente en w.
c = v
v[0] = 100

print("vector c: ", c)
print("vector v: ", v)
print("vector w: ", w)


# In[6]:


# vemos que c ha adoptado el cambio aplicado a v, porque es una referencia
# sin embargo, w no ha cambiado, porque es una copia con su propio espacio en memoria

# obtener el tamaño de un vector
w.shape


# In[7]:


# vemos que w tiene solo una dimensión (un eje o "axis"). Es un vector, un tensor de orden 1

# podemos cambiar el tamaño de un vector, matriz o tensor, siempre que no afectemos al número de elementos
w.shape = (3,1)

# ahora, w es una matriz de 3 por 1... tensor de orden 2, pero con dimensiones de un vector (un vector columna)
print(w)
print('')
print("tamaño de w: ", w.shape)


# In[8]:


# creamos ahora un vector de 12 elementos
w = np.array([1,2,3,4,5,6,7,8,9,10,11,12])
w


# In[9]:


# podemos redimensionar el vector de 12 dimensiones a una matriz de 3 por 4
h = w.reshape(3,4)
h


# In[10]:


v = np.array([[1,2,3]])
print("vector: ", v)
print("tamaño del vector", v.shape)
print("¡¡más que un vector, es una matriz de 1,3 !! (un vector fila)")


# In[11]:


# trasponer una matriz
print ('mismo vector columna: \n', v.T)
print ('cuyo tamaño es ', v.T.shape)


# In[12]:


# podemos crear una matriz directamente con np.array()
p = np.array([[1,2,3],[4,5,6]])
p


# In[13]:


# también calcular su traspuesta
p.T


# In[14]:


# aplicamos una función a todos los elementos de un vector o matriz
np.sin(p)


# ## Crear vectores y matrices típicos

# In[15]:


# crear un vector de unos
np.ones(5)


# In[16]:


# o una matriz de unos (2 x 3)
np.ones((2,3))


# In[17]:


# crear una matriz de ceros
np.zeros((3,5))


# In[18]:


# vector de números aleatorios con distribución uniforme [0,1]
np.random.rand(5)


# In[19]:


# matriz de numeros aleatorios distrib. uniforme [0,1]
np.random.rand(2,3)


# In[20]:


# matriz de números aleatorios distrib. normal N(0,1)
np.random.randn(2,3)


# In[21]:


# vector con 10 valores linealmente espaciados entre -1 y 3
np.linspace(-1,3,10)


# In[22]:


# generar vector de valores entre -1 y 3, espaciados 0.5 entre sí
np.arange(-1,3,0.5)


# ## Gráficas básicas con matplotlib
# Matplotlib es una librería muy extensa de python para crear gráficos 2D de muchos tipos (plot, scatter, barras, etc.) que incluye numerosas opciones de configuración. Entre otras cosas, permite introducir en las gráficas ecuaciones de LaTeX, generar trazas con transparencia, salvar las figuras en múltiples formatos (pdf, png, jpg,...), etc.

# In[23]:


# generamos un vector de x equiespaciados
x = np.linspace(-5,5,1000)

# generamos los valores de ordenadas (y)
y = np.sinc(x)

# abrimos una figura nueva
f = plt.figure(figsize=(10,10))
plt.plot(x,y,linewidth=5,alpha=0.2)

# podemos utilizar LaTeX (si lo tenemos instalado en el sistema)
# plt.rc('text', usetex=True)
plt.xlabel(r'$\int_0 f(x) dx$')
plt.ylabel(r'$\int_0 f(x) dx$')
plt.text(2,0.2,r'$\int_0 f(x) dx$')
plt.grid(True)
# f.savefig('./figuras/minitutorial-numpy-matploblib-figura-1.pdf')


# In[24]:


# Ejemplos de subplots
f = plt.figure(figsize=(10,10))
f.clf()
plt.subplot(2,2,1)
plt.plot(x,y)
plt.subplot(2,2,2)
plt.plot(x,x**2)
plt.subplot(2,2,3)
plt.plot(x,np.sin(x))
plt.subplot(2,2,4)
plt.plot(x,x)


# ## Trazado de familias de curvas con plot
# ### Familia de curvas representada mediante distintos subplots (uno por curva)

# In[25]:


f = plt.figure(figsize=(15,15))
f.clf()
for k,v in enumerate(np.linspace(2,5,4)):
    plt.subplot(2,2,k+1)
    plt.plot(x,np.sinc(x*v))
    plt.grid(True)
    plt.title(u'funcion sinc(%1.1f x)'%(v))


# ### Familia de curvas en la misma figura

# In[26]:


x = np.linspace(-5,5,1000)
y = np.sinc(x)

f = plt.figure(1,figsize=(10,10))
f.clf()
for k in np.arange(1,4,0.5):
    plt.plot(x,np.sinc(x*k),linewidth=5,alpha=0.2)
plt.grid(True)    
# f.savefig('./figuras/minitutorial-numpy-matploblib-figura-2.png')


# ### Gráfica de dispersión (scatterplot)

# In[27]:


# generamos datos aleatorios
# apilamos horizontalmente dos nubes gaussianas de puntos 2D: una centrada en (0,0) y otra en (3,3)
p = np.hstack((np.random.randn(2,500),np.random.randn(2,200)+3))

# obtenemos un tercer valor z = f(x,y) = x*y
z = p[0,:]*p[1,:]

# creamos una figura (contenedor) para alojar la gráfica de dispersión
f = plt.figure(figsize=(10,10))

# dibujamos la gráfica. En ella es posible definir para cada punto
# x,y: definen las coordenadas 2D del punto
# c: define el color
# s: define el tamaño
# linewidths: define el grosor de la linea que rodea cada círculo
# alpha: define el nivel de transparencia (0=transparente, 1=opaco)
plt.scatter(p[0,:],p[1,:],c=z,s=np.exp(z/4.),linewidths=0,alpha=0.3)

# añadimos etiquetas de texto a cada punto
for i in range(p.shape[1]):
    plt.text(p[0,i],p[1,i],'%1.1f'%(z[i]),fontsize=3)

# opcionalmente, podemos guardar la figura en formato pdf
# f.savefig('./figuras/minitutorial-numpy-matploblib-figura-3.jpg')


# ## Retículas 2D y mapas de calor
# Los mapas de calor o *heatmaps* nos permiten representar funciones del tipo z = f(x,y) asignando a cada punto (x,y) de la gráfica un color dado por z según una escala de color. Este tipo de representación es muy útil, por ejemplo, en mapas. Para hacer el *heatmap* debemos generar de alguna forma, una retícula de puntos 2D uniforme, que cubren los valores de x e y; luego darles un color según z de acuerdo con una escala de color.

# In[28]:


# generamos sendos intervalos de 30 valores entre -5 y 5 para x e y
x = np.linspace(-5,5,30)
y = np.linspace(-5,5,30)

# Con meshgrid, generamos matrices xi, yi de 30x30 con las coordenadas x e y de los puntos de la retícula
# La función meshgrid, en realidad nos hace el producto cartesiano entre x e y, es decir genera matrices 
# que tienen todas las combinaciones posibles de los puntos de x con los puntos de y
xi,yi = np.meshgrid(x,y)

# las matrices resultantes son de tamaño 30 x 30
xi.shape, yi.shape


# In[29]:


# si queremos utilizar un scatter, debemos generar los 30x30 puntos 
# podemos hacerlo a partir de las matrices xi, yi
# la función ravel(), aplasta las matrices xi, yi redimensionándolas de 30x30 a 1x900
xi_plana = xi.ravel()
yi_plana = yi.ravel()

xi_plana.shape, yi_plana.shape


# In[30]:


# luego la función vstack las apila en vertical, generando una matriz de 2 x 900
p = np.vstack((xi_plana,yi_plana))
# cada columna de p puede verse como 900 columnas de vectores 2D
p.shape


# In[31]:


# hacemos el scatterplot 
plt.scatter(p[0,:],p[1,:],c=p[0,:]*p[1,:],linewidths=0)

# añadimos una barra de color que nos da la escala empleada como referencia visual
plt.colorbar()


# In[32]:


# otra forma parecida generando directamente una imagen a partir de las matrices xi,yi
plt.imshow(xi*yi)
plt.colorbar()


# In[33]:


# Generamos una imagen con líneas de contorno
cs = plt.contour(xi,yi,xi*yi,levels=np.arange(-25,25,1))

# etiquetamos solo algunas de ellas (de 5 en 5) con sus valores
plt.clabel(cs,np.array([-15,-10,-5,0,5,10,15]))
plt.colorbar()


# In[34]:


get_ipython().run_cell_magic('html', '', '<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Licencia de Creative Commons" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br /><span xmlns:dct="http://purl.org/dc/terms/" property="dct:title">Minitutorial de numpy y matplotlib</span> by <a xmlns:cc="http://creativecommons.org/ns#" href="http://isa.uniovi.es/~idiaz" property="cc:attributionName" rel="cc:attributionURL">Ignacio Díaz Blanco</a> is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Reconocimiento-CompartirIgual 4.0 Internacional License</a>.\n')

