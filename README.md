# Justmodel 



## Project Structure

```
- data          - dataset de casos resueltos con sentencia 				
- src			
- exp           - 
```

## Acknowledgment

- I have borrowed the initial project structure and framework code from [arnabbiswas1's](https://github.com/arnabbiswas1/kaggle_pipeline_tps_aug_22) open sourced code.

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

- custom metric functions: [1](https://towardsdatascience.com/custom-loss-functions-for-gradient-boosting-f79c1b40466d),[2](https://github.com/manifoldai/mf-eng-public/blob/master/notebooks/custom_loss_lightgbm.ipynb),
- metric: [binary_logloss](https://towardsdatascience.com/intuition-behind-log-loss-score-4e0c9979680a#:~:text=Log%2Dloss%20is%20indicative%20of,is%20the%20log%2Dloss%20value.)
- hpyer parameters optimization grid: [1](https://github.com/Microsoft/LightGBM/issues/695) 
