# lib------
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

# Prophet Boost--------------------------------------------------------------

# repo: https://business-science.github.io/modeltime/reference/prophet_boost.html 
# implementation: 
## https://www.kaggle.com/code/gomes555/xgboost-bayes-opt-with-tidymodels
## https://www.hfshr.xyz/posts/2020-05-23-tidymodel-notes/
# tuning: 
# https://medium.com/grabngoinfo/hyperparameter-tuning-and-regularization-for-time-series-model-using-prophet-in-python-9791370a07dc
# https://medium.com/grabngoinfo/hyperparameter-tuning-and-regularization-for-time-series-model-using-prophet-in-python-9791370a07dc

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

# Prophet Boost
# Check parameters' range with 
# extract_parameter_set_dials(): you must check if there is any parameter 
# nparam[?] means that values range is missing for mtry parameter.

## Round 1: Hyperparamter Tuning ------------------------------------------------

# toggle on parallel processing

plan(strategy = cluster, workers  = parallel::makeCluster(n_cores))

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
    iter = 10,
    # How to measure performance? RSME raiz cuadrada del error cuadrado medio y R2 % de variabilidad
    metrics = metric_set(rmse, rsq),
    control = control_bayes(no_improve = 30, verbose = TRUE) # No parelelizable: ha confirmar
  )

# desactivar proc paralelo
plan(strategy = sequential)

# Resultados
tune_results_prophet_boost_1 %>% 
  show_best("rmse", n = Inf)

tune_results_prophet_boost_1 %>% 
  show_best("rsq", n = Inf)

# Gráficos
gr1<- tune_results_prophet_boost_1 %>%
  autoplot() +
  geom_smooth(se = FALSE)

ggplotly(gr1)

## Round 2: Hyperparamter Tuning -----------------------------------------------

# update learn_rate parameter range of values.

plan(strategy = cluster, workers  = parallel::makeCluster(n_cores))

set.seed(123)
prophet_boost_set_2 <-
  prophet_boost_set_1 %>%
  update(mtry = mtry(c(5, 45)), 
         learn_rate = learn_rate(c(-1.7, 0)))

tune_results_prophet_boost_2 <- wflw_spec_prophet_boost_tune %>%
  tune_bayes(
    resamples = resamples_kfold,
    # To use non-default parameter ranges
    param_info = prophet_boost_set_2,
    # Generate semi-random to start
    initial = 10, # prophet_boost_initial is in garbage
    iter = 10,
    # How to measure performance?
    metrics = metric_set(rmse, rsq),
    control = control_bayes(no_improve = 30, verbose = TRUE)
  )

plan(strategy = sequential)

tune_results_prophet_boost_2 %>% 
  show_best("rmse", n = Inf)

tune_results_prophet_boost_2 %>% 
  show_best("rsq", n = Inf)

gr2<- tune_results_prophet_boost_2 %>%
  autoplot() +
  geom_smooth(se = FALSE)

ggplotly(gr2)


## Round 3: Hyperparamter Tuning -----------------------------------------------

plan(strategy = cluster, workers  = parallel::makeCluster(n_cores))

set.seed(123)
prophet_boost_set_3 <-
  prophet_boost_set_2 %>%
  update(mtry = mtry(c(5, 45)), 
         learn_rate = learn_rate(c(-1.7, 0)), 
         trees = trees(c(1183, 1993)))

         
tune_results_prophet_boost_3 <- wflw_spec_prophet_boost_tune %>%
  tune_bayes(
    resamples = resamples_kfold,
    # To use non-default parameter ranges
    param_info = prophet_boost_set_3,
    # Generate semi-random to start
    initial = 10, # prophet_boost_initial is in garbage
    iter = 10,
    # How to measure performance?
    metrics = metric_set(rmse, rsq),
    control = control_bayes(no_improve = 30, verbose = TRUE)
  )

plan(strategy = sequential)

tune_results_prophet_boost_3 %>% 
  show_best("rmse", n = Inf)

tune_results_prophet_boost_3 %>% 
  show_best("rsq", n = Inf)

gr3<- tune_results_prophet_boost_3 %>%
  autoplot() +
  geom_smooth(se = FALSE)

ggplotly(gr3)

## Select and fit the best model(s)------

# Fitting round 3 best RMSE model
set.seed(123)
wflw_fit_prophet_boost_tuned <- wflw_spec_prophet_boost_tune %>%
  finalize_workflow(
    select_best(tune_results_prophet_boost_3, "rmse", n=1)) %>%
  fit(training(splits))

modeltime_table(wflw_fit_prophet_boost_tuned) %>%
  modeltime_calibrate(testing(splits)) %>%
  modeltime_accuracy()

# Fitting round 3 best RSQ model
set.seed(123)
wflw_fit_prophet_boost_tuned_rsq <- wflw_spec_prophet_boost_tune %>%
  finalize_workflow(
    select_best(tune_results_prophet_boost_3, "rsq", n=1)) %>%
  fit(training(splits))

modeltime_table(wflw_fit_prophet_boost_tuned_rsq) %>%
  modeltime_calibrate(testing(splits)) %>%
  modeltime_accuracy()

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

archivo_salida  <-  paste0(HOME_DIR,"/exp/001/tuned_prophet_xgb.rds")

tuned_prophet_xgb %>% 
  write_rds(archivo_salida)

# XGBoost---------------------------------------------------------------------
# https://parsnip.tidymodels.org/reference/details_boost_tree_xgboost.html

## Identify tuning parameters----

model_spec_xgboost_tune <- parsnip::boost_tree(
  mode = "regression",
  trees = 1000,
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
  iter = 5,
  # How to measure performance? RSME raiz cuadrada del error cuadrado medio y R2 % de variabilidad
  metrics = metric_set(rmse, rsq),
  control = control_bayes(no_improve = 30, verbose = TRUE, parallel_over =  'everything') # acá también se puede paralelizar
)

tune_results_xgboost_1 %>% 
  show_best("rmse", n = Inf)

tune_results_xgboost_1 %>% 
  show_best("rsq", n = Inf)

gr1<- tune_results_xgboost_1 %>%
  autoplot() +
  geom_smooth(se = FALSE)

ggplotly(gr1)

## Round 2: Hyperparamter Tuning ------------------------------------------------

xgboost_set_2 <-
  xgboost_set_1 %>%
  update(learn_rate = learn_rate(c(-3,-1)))

tune_results_xgboost_2 = wflw_spec_xgboost_tune %>% 
  tune_bayes(
    resamples = resamples_kfold,
    # To use non-default parameter ranges
    param_info = xgboost_set_2,
    # Generate semi-random to start
    initial = 10, # prophet_boost_initial is in garbage
    iter = 5,
    # How to measure performance? RSME raiz cuadrada del error cuadrado medio y R2 % de variabilidad
    metrics = metric_set(rmse, rsq),
    control = control_bayes(no_improve = 30,
                            verbose = TRUE,
                            parallel_over =  'everything') # acá también se puede paralelizar
  )

tune_results_xgboost_2 %>% 
  show_best("rmse", n = Inf)

tune_results_xgboost_2 %>% 
  show_best("rsq", n = Inf)

gr2<- tune_results_xgboost_2 %>%
  autoplot() +
  geom_smooth(se = FALSE)

ggplotly(gr2)


## Round 3: Hyperparamter Tuning ------------------------------------------------

xgboost_set_3 <-
  xgboost_set_1

tune_results_xgboost_3 = wflw_spec_xgboost_tune %>% 
  tune_bayes(
    resamples = resamples_kfold,
    # To use non-default parameter ranges
    param_info = xgboost_set_3,
    # Generate semi-random to start
    initial = 10, # prophet_boost_initial is in garbage
    iter = 10, # Ojo mayor iteraciones
    # How to measure performance? RSME raiz cuadrada del error cuadrado medio y R2 % de variabilidad
    metrics = metric_set(rmse, rsq),
    control = control_bayes(no_improve = 30,
                            verbose = TRUE,
                            parallel_over =  'everything') # acá también se puede paralelizar
  )

tune_results_xgboost_3 %>% 
  show_best("rmse", n = Inf)

tune_results_xgboost_3 %>% 
  show_best("rsq", n = Inf)

gr3<- tune_results_xgboost_3 %>%
  autoplot() +
  geom_smooth(se = FALSE)

ggplotly(gr3)

# Fitting round 3 best RMSE model -----
set.seed(123)
wflw_fit_xgboost_tuned <- wflw_spec_xgboost_tune %>%
  finalize_workflow(
    select_best(tune_results_xgboost_3, "rmse", n=1)) %>%
  fit(training(splits))

modeltime_table(wflw_fit_xgboost_tuned) %>%
  modeltime_calibrate(testing(splits)) %>%
  modeltime_accuracy()

# Fitting round 3 best RSQ model
wflw_fit_xgboost_tuned_rsq <- wflw_spec_xgboost_tune %>%
  finalize_workflow(
    select_best(tune_results_xgboost_3, "rsq", n=1)) %>%
  fit(training(splits))

modeltime_table(wflw_fit_xgboost_tuned_rsq) %>%
  modeltime_calibrate(testing(splits)) %>%
  modeltime_accuracy()

## Save Prophet Boot tuning artifacts------
tuned_xgboost <- list(
  
  # Workflow spec
  tune_wkflw_spec = wflw_spec_xgboost_tune, # best model workflow
  # Grid spec
  tune_bayes_param_set = list(
    round1 = xgboost_set_1,
    round2 = xgboost_set_2,
    round3 = xgboost_set_3),
  # Tuning Results
  tune_results = list(
    round1 = tune_results_xgboost_1,
    round2 = tune_results_xgboost_2,
    round3 = tune_results_xgboost_3),
  # Tuned Workflow Fit
  tune_wflw_fit = wflw_fit_xgboost_tuned,
  # from FE
  splits        = artifacts$splits,
  data          = artifacts$data,
  recipes       = artifacts$recipes,
  standardize   = artifacts$standardize,
  normalize     = artifacts$normalize
  
)

archivo_salida  <-  paste0(HOME_DIR,"/exp/001/tuned_xgboost.rds")

tuned_xgboost %>% 
  write_rds(archivo_salida)

-----------------------
# Ligthgbm--------------------------------------------------------------------

## ligthgbm - identify tunin parameters

model_spec_lightgbm_tune <- parsnip::boost_tree(
  mtry = tune(),
  trees = 100,
  min_n = tune(),
  tree_depth = tune(),
  loss_reduction = tune(),
  learn_rate = tune()) %>% # ojo la configuracion de este parámetro x transformacion log10 https://dials.tidymodels.org/reference/learn_rate.html#ref-examples
  set_mode("regression") %>%
  set_engine("lightgbm", objective = "root_mean_squared_error") 


wflw_spec_lightgbm_tune <- workflow() %>%
  add_model(model_spec_lightgbm_tune) %>%
  add_recipe(artifacts$recipes$recipe_spec)

## Round 1: ----------------------------------------------
# plan(strategy = cluster, workers  = parallel::makeCluster(n_cores)) # ojo no se puede paralelizar. Pruebo 6vCPU

lightgbm_set <- extract_parameter_set_dials(wflw_spec_lightgbm_tune)

lightgbm_set_1 <-
  lightgbm_set %>%
  recipes::update(mtry = mtry(c(1L,30L)))

tune_results_lightgbm_boost_1 <- wflw_spec_lightgbm_tune %>%
  tune_bayes(
    resamples = resamples_kfold,
    # To use non-default parameter ranges
    param_info = lightgbm_set_1,
    # Generate semi-random to start
    initial = 10,
    iter = 10,
    # How to measure performance?
    metrics = metric_set(rmse, rsq),
    control = control_bayes(no_improve = 30, verbose = TRUE, parallel_over =  'everything')
  )

#plan(strategy = sequential)

tune_results_lightgbm_boost_1 %>% 
  show_best("rmse", n = Inf)

tune_results_lightgbm_boost_1 %>% 
  show_best("rsq", n = Inf)

gr1<- tune_results_lightgbm_boost_1 %>%
  autoplot() +
  geom_smooth(se = FALSE)

ggplotly(gr1)

## Round 2: ----------------------------------------------

# treesnop/parsnip tiene posibilidad restringidas de parametrizar el lightgbm. Con formas de parametrización con bugs -ie.learn_rate-

# lightgbm_set_2 <- 
#   lightgbm_set_1 # %>% 
#   # update(learn_rate = learn_rate(range = c(-10, -1), trans = log10_trans()))
#          
# tune_results_lightgbm_boost_2 <- wflw_spec_lightgbm_tune %>%
#   tune_bayes(
#     resamples = resamples_kfold,
#     # To use non-default parameter ranges
#     param_info = lightgbm_set_2,
#     # Generate five at semi-random to start
#     initial = 10,
#     iter = 10,
#     # How to measure performance?
#     metrics = metric_set(rmse, rsq),
#     control = control_bayes(no_improve = 30, verbose = TRUE)
#   )
# 
# tune_results_lightgbm_boost_2 %>% 
#   show_best("rmse", n = Inf)
# 
# tune_results_lightgbm_boost_2 %>% 
#   show_best("rsq", n = Inf)
# 
# # tune_results_lightgbm_boost_2 %>% 
# #   show_best("rmse", n = Inf)%>% 
# #   select(1:7, 11) %>% 
# #   gt()
# 
# gr2<- tune_results_lightgbm_boost_2 %>%
#   autoplot() +
#   geom_smooth(se = FALSE)
# 
# ggplotly(gr2)


# Fitting the best model------------
set.seed(123)
wflw_fit_lightgbm_tuned <- wflw_spec_lightgbm_tune %>%
  finalize_workflow(
    select_best(tune_results_lightgbm_boost_1, "rmse", n=1)) %>%
  fit(training(splits))

modeltime_table(wflw_fit_lightgbm_tuned) %>%
  modeltime_calibrate(testing(splits)) %>%
  modeltime_accuracy()

# Fitting round 3 best RSQ model
wflw_fit_lightgbm_tuned_rsq <- wflw_spec_lightgbm_tune %>%
  finalize_workflow(
    select_best(tune_results_lightgbm_boost_1, "rsq", n=1)) %>%
  fit(training(splits))

modeltime_table(wflw_fit_lightgbm_tuned_rsq) %>%
  modeltime_calibrate(testing(splits)) %>%
  modeltime_accuracy()

## Save ligthgbm tuning artifacts------
tuned_lightgbm <- list(
  
  # Workflow spec
  tune_wkflw_spec = wflw_spec_lightgbm_tune, # best model workflow
  # Tune spec
  tune_bayes_param_set = list(
    round1 = lightgbm_set_1),
  # Tuning Results
  tune_results = list(
    round1 = tune_results_lightgbm_boost_1),
  # Tuned Workflow Fit
  tune_wflw_fit = wflw_fit_lightgbm_tuned,
  # from FE
  splits        = artifacts$splits,
  data          = artifacts$data,
  recipes       = artifacts$recipes,
  standardize   = artifacts$standardize,
  normalize     = artifacts$normalize
  )

archivo_salida  <-  paste0(HOME_DIR,"/exp/001/tuned_lightgbm.rds")

# Error: Attempting to use a Booster which no longer exists. This can happen if 
# you have called Booster$finalize() or if this Booster was saved with saveRDS(). 
# To avoid this error in the future, use saveRDS.lgb.Booster() or 
# Booster$save_model() to save lightgbm Boosters.

# ver https://stackoverflow.com/questions/72027360/how-to-save-tidymodels-lightgbm-model-for-reuse

tuned_lightgbm %>% 
  #saveRDS.lgb.Booster(archivo_salida,)
  saveRDS(archivo_salida) # ojo saveRDS sino errores de calibracion


# SVM-----

model_spec_svm_tune <- svm_rbf(cost = tune(), 
                                           rbf_sigma = tune()) %>% # param to be tuned
  set_mode("regression") %>% # binary response var
  set_engine("kernlab")

wflw_spec_svm_tune <- workflow() %>%
  add_model(model_spec_svm_tune) %>%
  add_recipe(artifacts$recipes$recipe_spec)

## Round 1:----

svm_set <- extract_parameter_set_dials(wflw_spec_svm_tune)

tune_results_svm_1 <- wflw_spec_svm_tune %>%
  tune_bayes(
    resamples = resamples_kfold,
    # To use non-default parameter ranges
    param_info = svm_set,
    # Generate five at semi-random to start
    initial = 10,
    iter = 10,
    # How to measure performance?
    metrics = metric_set(rmse, rsq),
    control = control_bayes(no_improve = 30, verbose = TRUE)
  )

tune_results_svm_1 %>% 
  show_best("rmse", n = Inf)

tune_results_svm_1 %>% 
  show_best("rsq", n = Inf)

gr1<- tune_results_svm_1 %>%
  autoplot() +
  geom_smooth(se = FALSE)

ggplotly(gr1)



# RandomForest----

model_spec_random_forest_tune <- rand_forest(
      mtry = tune(),
      trees = tune(),
      min_n = tune()
      ) %>%
  set_mode("regression") %>% 
  set_engine("ranger") 

wflw_spec_random_forest_tune <- workflow() %>%
  add_model(model_spec_random_forest_tune) %>%
  add_recipe(artifacts$recipes$recipe_spec %>% 
               step_rm(mes)) 

random_forest_set <- extract_parameter_set_dials(model_spec_random_forest_tune)

random_forest_set_1 <-
  random_forest_set %>%
  update(mtry = mtry(c(1, ncol(splits$data))))

tune_results_random_forest_1 = wflw_spec_random_forest_tune %>% 
  tune_bayes(
    resamples = resamples_kfold,
    # To use non-default parameter ranges
    param_info = random_forest_set_1,
    # Generate semi-random to start
    initial = 7, 
    iter = 5,
    # How to measure performance? RSME raiz cuadrada del error cuadrado medio y R2 % de variabilidad
    metrics = metric_set(rmse, rsq),
    control = control_bayes(no_improve = 30, verbose = TRUE, parallel_over =  'everything') # acá también se puede paralelizar
  )


tune_results_random_forest_1 %>% 
  show_best("rmse", n = Inf)

tune_results_random_forest_1 %>% 
  show_best("rsq", n = Inf)

gr1- tune_results_random_forest_1 %>%
  autoplot() +
  geom_smooth(se = FALSE)

ggplotly(gr1)

# Fitting round 3 best RMSE model -----
set.seed(123)
wflw_fit_random_forest_tuned <- wflw_spec_random_forest_tune %>%
  finalize_workflow(
    select_best(tune_results_random_forest_1, "rmse", n=1)) %>%
  fit(training(splits))

modeltime_table(wflw_fit_random_forest_tuned) %>%
  modeltime_calibrate(testing(splits)) %>%
  modeltime_accuracy()

# Fitting round 3 best RSQ model
wflw_fit_random_forest_tuned_rsq <- wflw_spec_random_forest_tune %>%
  finalize_workflow(
    select_best(tune_results_random_forest_1, "rsq", n=1)) %>%
  fit(training(splits))

modeltime_table(wflw_fit_random_forest_tuned_rsq) %>%
  modeltime_calibrate(testing(splits)) %>%
  modeltime_accuracy()

## Save RF tuning artifacts------
tuned_random_forest <- list(
  
  # Workflow spec
  tune_wkflw_spec = wflw_fit_random_forest_tuned, # best model workflow
  # Grid spec
  tune_bayes_param_set = list(
    round1 = random_forest_set_1
    # round2 = random_forest_set_2,
    # round3 = random_forest_set_3),
  ),
  # Tuning Results
  tune_results = list(
    round1 = tune_results_random_forest_1
    # round2 = tune_results_random_forest_2,
    # round3 = tune_results_random_forest_3),
  ),
  # Tuned Workflow Fit
  tune_wflw_fit = wflw_fit_random_forest_tuned,
  # from FE
  splits        = artifacts$splits,
  data          = artifacts$data,
  recipes       = artifacts$recipes,
  standardize   = artifacts$standardize,
  normalize     = artifacts$normalize
  
)

archivo_salida  <-  paste0(HOME_DIR,"/exp/001/tuned_random_forest.rds")

tuned_random_forest %>% 
  write_rds(archivo_salida)


# Modeltime & Calibration tables-----
# The objective is to add all non-tuned and tuned models into a single modeltime 
# table submodels_all_tbl.
# Please recall that we deal with workflows (not models directly) which 
# incoporates a model and a preprocessing recipe.
# Notice how we combined (combine_modeltime_tables()) the tables and updated 
# the description 
# (update_model_description()) for the tuned workflows.

submodels_tbl <- modeltime_table(
  wflw_artifacts$workflows$wflw_random_forest,
  wflw_artifacts$workflows$wflw_xgboost,
  #wflw_artifacts$workflows$wflw_prophet,
  wflw_artifacts$workflows$wflw_prophet_boost ,
  wflw_artifacts$workflows$wflw_lightgbm
)

submodels_all_tbl <- modeltime_table(
  tuned_random_forest$tune_wflw_fit,
  tuned_xgboost$tune_wflw_fit,
  #tuned_lightgbm$tune_wflw_fit,
  tuned_prophet_xgb$tune_wflw_fit
  ) %>%
  update_model_description(1, "RANGER - Tuned") %>%
  update_model_description(2, "XGBOOST - Tuned") %>%
  #update_model_description(3, "LIGHTGBM - Tuned") %>%
  update_model_description(4, "PROPHET W/ XGBOOST ERRORS - Tuned") %>%
  combine_modeltime_tables(submodels_tbl)


# Model evaluation results
# We calibrate all fitted workflows with the test dataset and display the accuracy 
# results for all non-tuned and tuned models. 
# You may also create an interactive table which facilitates sorting, 
# as shown in the code snippet below.

calibration_all_tbl <- submodels_all_tbl %>%
  modeltime_calibrate(testing(splits), quiet = FALSE)

calibration_all_tbl %>%
  modeltime_accuracy() %>%
  arrange(rmse)

# Integro LightGBM
# accuracy_all <- calibration_all_tbl %>% 
#   modeltime_accuracy() %>%
#   bind_rows(tuned_lightgbm$lightgbm_tuned_test_accuracy) %>% 
#   mutate(.model_desc = str_replace(.model_desc, "LIGHTGBM", "LIGHTGBM - Tuned")) %>% 
#   arrange(rmse) 

# Interactive table 
accuracy_all %>%
  table_modeltime_accuracy()

# Forecast plot
# We plot a forecast with test data using all models and industries, 
# the figure below is a zoom of the test data predictions vs. actual data.

calibration_all_tbl %>%
  modeltime_forecast(
    new_data    = testing(splits),
    actual_data = artifacts$data$data_prepared_tbl,
    keep_data   = TRUE 
  ) %>%
  filter(Industry == Industries[1]) %>%
  plot_modeltime_forecast(
    #.facet_ncol         = 4, 
    .conf_interval_show = FALSE,
    .interactive        = TRUE,
    .title = Industries[1]
  )

# Save all work
workflow_all_artifacts <- list(
  
  workflows = submodels_all_tbl,
  
  calibration = calibration_all_tbl
)

archivo_salida  <-  paste0(HOME_DIR,"/exp/001/workflows_NonandTuned_artifacts_list.rds")

workflow_all_artifacts %>%
  write_rds(archivo_salida)


