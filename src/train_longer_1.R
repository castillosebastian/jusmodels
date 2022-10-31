# lib------
source("~/jusmodels/main.R")
pacman::p_load(tidyverse,
               timetk,
               tsibble,
               tsibbledata,
               fastDummies,
               skimr, 
               tidymodels, 
               recipes,
               modeltime,
               future, 
               doFuture,
               plotly,
               treesnip)
options(scipen=999)
options(tidymodels.dark = TRUE)

# bayes opt: important
# https://towardsdatascience.com/the-beauty-of-bayesian-optimization-explained-in-simple-terms-81f3ee13b10f
# r package: https://github.com/yanyachen/rBayesianOptimization 
# https://www.tidymodels.org/learn/work/bayes-opt/
# https://r4ds.github.io/bookclub-tmwr/bayesian-optimization.html
# https://www.kaggle.com/code/athosdamiani/lightgbm-with-tidymodels
# parallell procesing
# https://business-science.github.io/modeltime/articles/parallel-processing.html

# todo------
# implementar HT with rBayesianOptimization 

# Load data y process enviroment----------
artifacts <- read_rds(paste0(HOME_DIR, "/exp/001/feature_engineering_artifacts_list.rds"))
splits            <- artifacts$splits
recipe_spec       <- artifacts$recipes$recipe_spec
materia        <- artifacts$data$materia

wflw_artifacts <- read_rds(paste0(HOME_DIR, "/exp/001/workflows_artifacts_list.rds"))
 
# Cross-validation plan----
set.seed(123)
resamples_kfold <- training(splits) %>% 
  vfold_cv(v = 10)

# Registers the doFuture parallel processing
registerDoFuture()

# detect CPU / threads (or vCores)
n_cores <- parallel::detectCores()


model_spec_prophet_boost_tune <- prophet_boost(
  mode = "regression",
  # growth = NULL,
  changepoint_num = tune(), # important parameter
  prior_scale_changepoints = tune(), # ver: probar recommended tuning range is 0.001 to 0.5
  #changepoint_range = NULL,
  seasonality_yearly = FALSE,
  seasonality_weekly = FALSE,
  seasonality_daily = FALSE,
  trees = tune(),
  mtry = tune(),
  min_n = tune(),
  tree_depth = tune(),
  learn_rate = tune(),
  loss_reduction = tune(),
  # sample_size = NULL,
  # stop_iter = NULL
  ) %>%
  set_engine("prophet_xgboost")

wflw_spec_prophet_boost_tune <- workflow() %>%
  add_model(model_spec_prophet_boost_tune) %>%
  add_recipe(artifacts$recipes$recipe_spec)


prophet_boost_set_1 <- extract_parameter_set_dials(wflw_spec_prophet_boost_tune)

prophet_boost_set_1 <-
  prophet_boost_set_1 %>%
  update(mtry = mtry(c(5, 45)))

tune_results_prophet_boost_1 <- wflw_spec_prophet_boost_tune %>%
  tune_bayes(
    resamples = resamples_kfold,
    # To use non-default parameter ranges
    param_info = prophet_boost_set_1,
    # Generate semi-random to start
    initial = 10, # prophet_boost_initial is in garbage
    iter = 50,
    # How to measure performance? RSME raiz cuadrada del error cuadrado medio y R2 % de variabilidad
    metrics = metric_set(rmse, rsq),
    control = control_bayes(no_improve = 30, verbose = TRUE) # No parelelizable: ha confirmar
  )

# desactivar proc paralelo
plan(strategy = sequential)


wflw_fit_prophet_boost_tuned <- wflw_spec_prophet_boost_tune %>%
  finalize_workflow(
    select_best(tune_results_prophet_boost_1, "rmse", n=1)) %>%
  fit(training(splits))



## Save Prophet Boot tuning artifacts------
tuned_prophet_xgb <- list(

  # Workflow spec
  tune_wkflw_spec = wflw_spec_prophet_boost_tune, # best model workflow
  # Grid spec
  tune_bayes_param_set = list(
    round1 = prophet_boost_set_1#,
    # round2 = prophet_boost_set_2,
    # round3 = prophet_boost_set_3
    ),
  # Tuning Results
  tune_results = list(
    round1 = tune_results_prophet_boost_1#,
    # round2 = tune_results_prophet_boost_2,
    # round3 = tune_results_prophet_boost_3
    ),
  # Tuned Workflow Fit
  tune_wflw_fit = wflw_fit_prophet_boost_tuned,
  # from FE
  splits        = artifacts$splits,
  data          = artifacts$data,
  recipes       = artifacts$recipes,
  standardize   = artifacts$standardize,
  normalize     = artifacts$normalize

)

archivo_salida  <-  paste0(HOME_DIR,"/exp/001/tuned_prophet_xgb_2.rds")

tuned_prophet_xgb %>%
  write_rds(archivo_salida)

# XGBoost---------------------------------------------------------------------
# https://parsnip.tidymodels.org/reference/details_boost_tree_xgboost.html

## Identify tuning parameters----

model_spec_xgboost_tune <- parsnip::boost_tree(
  mode = "regression",
  trees = 2000,
  min_n = tune(),
  tree_depth = tune(),
  learn_rate = tune(),
  loss_reduction = tune(),
  mtry = tune()
  ) %>%
  set_mode("regression") %>%
  set_engine("xgboost")


wflw_spec_xgboost_tune <- workflow() %>%
  add_model(model_spec_xgboost_tune) %>%
  add_recipe(artifacts$recipes$recipe_spec %>%
               step_rm(mes)) # https://stackoverflow.com/questions/72548979/my-parsnip-models-doesnt-work-in-modeltime-calibrate-function-after-updating-pa


## Round 1: Hyperparamter Tuning ------------------------------------------------

xgboost_set <- extract_parameter_set_dials(model_spec_xgboost_tune)

xgboost_set_1 <-
  xgboost_set %>%
  update(mtry = mtry(c(5, 45)))

tune_results_xgboost_1 = wflw_spec_xgboost_tune %>%
  tune_bayes(
  resamples = resamples_kfold,
  # To use non-default parameter ranges
  param_info = xgboost_set_1,
  # Generate semi-random to start
  initial = 10, # prophet_boost_initial is in garbage
  iter = 50,
  # How to measure performance? RSME raiz cuadrada del error cuadrado medio y R2 % de variabilidad
  metrics = metric_set(rmse, rsq),
  control = control_bayes(no_improve = 30, verbose = TRUE, parallel_over =  'everything') # acá también se puede paralelizar
)


set.seed(123)
wflw_fit_xgboost_tuned <- wflw_spec_xgboost_tune %>%
  finalize_workflow(
    select_best(tune_results_xgboost_1, "rmse", n=1)) %>%
  fit(training(splits))


## Save Prophet Boot tuning artifacts------
tuned_xgboost <- list(

  # Workflow spec
  tune_wkflw_spec = wflw_spec_xgboost_tune, # best model workflow
  # Grid spec
  tune_bayes_param_set = list(
    round1 = xgboost_set_1),
  # Tuning Results
  tune_results = list(
    round1 = tune_results_xgboost_1),
  # Tuned Workflow Fit
  tune_wflw_fit = wflw_fit_xgboost_tuned,
  # from FE
  splits        = artifacts$splits,
  data          = artifacts$data,
  recipes       = artifacts$recipes,
  standardize   = artifacts$standardize,
  normalize     = artifacts$normalize

)

archivo_salida  <-  paste0(HOME_DIR,"/exp/001/tuned_xgboost_2.rds")

tuned_xgboost %>%
  write_rds(archivo_salida)


