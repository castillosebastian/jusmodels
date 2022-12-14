<p align="left">
  <img src="https://github.com/castillosebastian/jusmodels/blob/master/data-science.png" width="300">
</p>

# Justmodels 

Este es un repositorio que contiene el *pipeline* completo para implementar modelos de *machine learning* en la predicción de series temporales. En el caso particular que nos sirve de ejemplo (ver documentación en la carpeta `/rmardowns` o en mi [webpage](https://castillosebastian.github.io/)) empleamos datos del servicio de justicia. 

El repositorio está armado de tal forma que los *scripts* puedan correrse de dos formas:   

- a través se sus documentos rmarkdown disponibles en `/rmarkdowns` , o
- mediante los archivos .R en `/source`.   

# Publicaciones del Repositorio

Los documentos RMardown generado en este repositorio puede consultarse on line aquí:

- [Feature engineering](https://rpubs.com/ClaudioSebastianCastillo/964233) 
- [Workflows](https://rpubs.com/ClaudioSebastianCastillo/963182)
- [Hyperparameter Tuning](https://rpubs.com/ClaudioSebastianCastillo/963694)
- [Ensambles y Meta-learners](https://rpubs.com/ClaudioSebastianCastillo/964231)


## Ambiente de trabajo

Intenté configurar el ambiente de trabajo de tal forma que fuera reproducible y que el repositorio en general sea fácil de implementar. Sin embargo resalto que la ejecución tanto de los scripts como de los documentos RMarkdown requieren un número importante de librerías y sus respectivas dependencias. Puedo anticiparles que, salvo que se disponga de una *tarde tranquila* para instalar todo y jugar con el código, la reproducción del procesamiento aquí presentado puede darles un dolor de cabeza. Para la próxima intentaré ser menos pretencioso :).


## Estructura del Projecto

```
- data    - dataset de sentencias dictadas por materia en primera instancia 				
- src     - script para ejecución en serie del pipeline completo 
- exp     - experimentos corridos en el desarrollo de modelos. Iniciamos repo con 1.
- env     - carpeta de configuración de ambiente
```

## Pasos para su ejecución

1. Clone repo.

        > git clone https://github.com/castillosebastian/justmodels.git
    
2. Crear ambiente r and python (en mi caso /usr/local/bin/python3) :
        
        > renv::init()
        > renv::use_python()

3. Dataset: el repo viene con datos de ejemplo pero está pensado para correr con cualquier set de datos. 

        > /data o descargar (ver ejemplo en main.R)

4. Correr `<PROJECT_HOME>/main.R` para setear variables y path

5. Jugar y romper a voluntad!

## To do:

- Implementar calibración de `lightgbm` compatibla con `modeltime_calibrate()`
- Reentrenar set de datos/particiones para predicciones en `future_dataset`

## Reconocimientos

- Este repositorio está fuertemente inspirado en el trabajo en *MLOps* realizado con un gran [Profesor Gustavo Denicolay](https://github.com/castillosebastian/labo) y los post de Series Temporales de [Matt Dancho](https://business-science.github.io/modeltime.ensemble/index.html) y [Boris Guarisma ](https://blog.bguarisma.com/). He procurado extender las implementaciones presentadas por este último, incorporando algoritmos (eg.lightgbm) o métodos (eg. optimización bayesiana) impulsados por Denicolay. Denicolay merece todos los créditos por los buenos logros, los errores me pertenecen.    

## Bibliografía empleada

- time_series: [1](https://otexts.com/fpp3/), [2](https://wires.onlinelibrary.wiley.com/doi/epdf/10.1002/widm.1475)
- fourier series: [1](https://conceptosclaros.com/transformada-de-fourier/)
- custom metric functions: [1](https://towardsdatascience.com/custom-loss-functions-for-gradient-boosting-f79c1b40466d),[2](https://github.com/manifoldai/mf-eng-public/blob/master/notebooks/custom_loss_lightgbm.ipynb),
- metric: [binary_logloss](https://towardsdatascience.com/intuition-behind-log-loss-score-4e0c9979680a#:~:text=Log%2Dloss%20is%20indicative%20of,is%20the%20log%2Dloss%20value.)
- hpyer parameters optimization grid: [1](https://github.com/Microsoft/LightGBM/issues/695)  
- ACF/PACF: [1](https://towardsdatascience.com/identifying-ar-and-ma-terms-using-acf-and-pacf-plots-in-time-series-forecasting-ccb9fd073db8)
- xgboost: [1](https://towardsdatascience.com/machine-learning-part-18-boosting-algorithms-gradient-boosting-in-python-ef5ae6965be4)
- lightgbm: [1](https://proceedings.neurips.cc/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf), [2](https://lightgbm.readthedocs.io/en/v3.3.2/),  [3](https://towardsdatascience.com/lightgbm-vs-xgboost-which-algorithm-win-the-race-1ff7dd4917d)
