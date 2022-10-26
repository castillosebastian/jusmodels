# Justmodel 



## Project Structure

```
- data          - dataset de casos resueltos con sentencia 				
- src			
- exp           - 
```

## Acknowledgment

- Este repositorio está fuertemente inspirado en el trabajo en *MLOps* realizado con el [Profesor Gustavo Denicolay](https://github.com/castillosebastian/labo) y los post de Series Temporales de [Boris Guarisma ](https://blog.bguarisma.com/). He procurado extender las implementaciones presentadas por este último en dirección de mejorar los resultados de las predicciones, incorporando algoritmos (eg.lightgbm) o métodos (eg. optimización bayesiana) impulsados por Denicolay. Éste último merece todos los créditos por los buenos logros, los errores me pertenecen.    






## Steps to execute:

1. Clone the source code from github under <PROJECT_HOME> directory.

        > git clone https://github.com/castillosebastian/justmodels.git
    
2. Create r and python (/usr/local/bin/python3) env:
        
        > renv::init()
        > renv::use_python()

3.  Dataset 

        > HOME_DIR /data o descargar ej.mainR

4. Setea variables y mas `<PROJECT_HOME>/main.R` 



        
        
        
10. Important Bib

- time_series: [1](https://otexts.com/fpp3/), [2](https://wires.onlinelibrary.wiley.com/doi/epdf/10.1002/widm.1475)
- custom metric functions: [1](https://towardsdatascience.com/custom-loss-functions-for-gradient-boosting-f79c1b40466d),[2](https://github.com/manifoldai/mf-eng-public/blob/master/notebooks/custom_loss_lightgbm.ipynb),
- metric: [binary_logloss](https://towardsdatascience.com/intuition-behind-log-loss-score-4e0c9979680a#:~:text=Log%2Dloss%20is%20indicative%20of,is%20the%20log%2Dloss%20value.)
- hpyer parameters optimization grid: [1](https://github.com/Microsoft/LightGBM/issues/695) 
