# Data Science Challenge

El siguiente repositorio contiene los archivos y codigos necesarios para cumplir con los objetivos de challenge de mercado libre, en donde se debe calsificar los productos en las categorias nuevos o usados

## Estructura
Se cuenta distintos archivos y con 4 carpetas, descritas a continuacion.
### Carpetas
* `data` en esta carpeta debe estar ubicado el archivo .json suministrado para la prueba
* `Funciones` esta carpeta contiene las funciones utilizadas para la tansformacion de los datos del modelo especificamente en el archivo `funciones.py`
* `notebooks` contiene 3 notebooks nuemrados, donde se realiza el paso de limpieza de variables y analisis exploratorio (EDA), transfomracion de datos y eleccion y calibracion de modelos, al interior de cada uno de estos hay conclusiones y analisis sobre la informacion dada
  * `01.clean_var.iptnb`: notebook de jupyter con la limpieza, transformacion y creacion de variables a partir del insumo para el modelo, su conclusion es que variables deben conservarse, eliminarse y transformarse para su uso y evaluacion en el modelo de clasificacion
  * `02.clean_var.iptnb`: contiene la version base de las transformaciones necesarias para aplicar en el entrenamiento del modelo de clasificacion, su conclusion es un dataset de entrenamiento y pruebas listo para ingestar en un modelo
  * `03.model_tunning.iptnb`: contiene la evaluacion de modelos de clasifficacion binaria para predecir si un item es nuevo o usado, su conclusion es la eleccion de un modelo de prediccion que tiene el mejor desempeño segun las metricas estipuladas enel mismo.
* `imagenes` carpeta que contiene las imagenes empleadas en los notebooks de la carpeta anterior
### Archivos
* `.gitignore`: archivo con los archivos que no se les hace seguimiento en el repo
* `new_or_used.py`: codigo python con la transformacion de los datos, entrenamiento del modelo y calculo de estadisticas de desempeño
* `requirements.txt`: documento con los requisiros de librerias para el modelo


