# Ejemplos de Machine Learning
## Descripción
Este repo contiene ejemplos y pequeñas demostraciones con datos reales, enfocados a la ingeniería, que sirven de apoyo para cursos y seminarios de introducción al *machine learning* (aprendizaje automático). Se abordan diversos conceptos básicos de ML, incluyendo: 
- regresión
- clasificación
- reducción de la dimensionalidad 
- identificación de sistemas dinámicos

El material puede ser actualizado, incorporando nuevos ejemplos, correcciones, o mejoras.

## Instalación

Los ejemplos están todos en Python y usan librerías habituales como `numpy`, `matplotlib`, `scikit-learn`, etc.  
Para garantizar **reproducibilidad** y evitar conflictos entre proyectos, este repositorio utiliza **`uv`** como gestor de entornos y dependencias (basado en `pyproject.toml` y `uv.lock`).

### Requisitos
- Python ≥ 3.10
- `uv` instalado (una sola vez en tu sistema)

Puedes instalar `uv` siguiendo las instrucciones oficiales:
https://docs.astral.sh/uv/

### Pasos de instalación

1. Clona el repositorio:
   ```bash
   git clone https://github.com/idiazblanco/ejemplos-curso-ML.git
   cd ejemplos-curso-ML
   ```

2. Crea el entorno virtual del proyecto e instala las dependencias exactas:
   ```bash
   uv sync
   ```

   Esto creará automáticamente un entorno virtual local (`.venv`) con las versiones de librerías fijadas en `uv.lock`.

3. Ejecuta los ejemplos en Jupyter Notebook:

   La mayoría de los ejemplos del repositorio están en formato **Jupyter Notebook (`.ipynb`)**.  
   Para trabajar con ellos, lanza Jupyter usando el entorno gestionado por `uv`:

   ```bash
   uv run jupyter lab
   ```

   o, alternativamente:

   ```bash
   uv run jupyter notebook
   ```

   Desde el navegador, abre los notebooks y ejecútalos con el kernel asociado al entorno del proyecto (`.venv`).
