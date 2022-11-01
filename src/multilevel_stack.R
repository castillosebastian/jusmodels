# Ensembles

# lib
pacman::p_load(tidyverse,
               timetk,
               tsibble,
               tsibbledata,
               fastDummies,
               skimr, 
               tidymodels, 
               modeltime,# 03 MACHINE LEARNING
               future, # 04 HYPERPARAMETER TUNING
               doFuture,
               plotly,
               modeltime.ensemble) # 5 ENSEMBLES


# load data

workflows_tunednot <- read_rds(paste0(HOME_DIR, "/exp/001/workflows_NonandTuned_artifacts_list.rds"))

calibration_tbl <- workflows_tunednot$calibration

artifacts <- read_rds(paste0(HOME_DIR, "/exp/001/feature_engineering_artifacts_list.rds"))
splits            <- artifacts$splits
recipe_spec       <- artifacts$recipes$recipe_spec
materia        <- artifacts$data$materia

wflw_artifacts <- read_rds(paste0(HOME_DIR, "/exp/001/workflows_artifacts_list.rds"))




# Stacking Algorithms
# A stacking algorithm or meta-learner learns from predictions. The predictions
# come from k-fold cross-validations applied to each submodel. Once fed into the 
# meta-learner tunable specification a hyperpameter tuning will be peformed for 
# the meta-learner.

set.seed(123)
resamples_kfold <- training(splits) %>%
  drop_na() %>%
  vfold_cv(v = 10)


submodels_resamples_kfold_tbl <- calibration_tbl %>%
  modeltime_fit_resamples(
    resamples = resamples_kfold,
    control   = control_resamples(
      verbose    = TRUE, 
      allow_par  = TRUE,
    )
  )

# Parallel Processing ----
registerDoFuture()
n_cores <- parallel::detectCores()
n_cores <- n_cores-2
plan(
  strategy = cluster,
  workers  = parallel::makeCluster(n_cores)
)

# Meta-learners specification
# In this article we will use 3 meta-learners: Random Forest, XGBoost, and SVM.
# Create a stacking algorithm specification with ensemble_model_spec(). 
# Within the meta-learner specification we must provide the tunable parameters,
# the number of folds, and the size of the grid. We will also set grid search 
# control parameters by activating verbose and allowing parallel processing.
# For all meta-learners specifications we will define a k=10 folds and a
# grid size of 20.

# Random Forest stacking algorithm-----
# We perform tuning on 2 parameters: mtry and min_n.

ensemble_fit_ranger_kfold <- submodels_resamples_kfold_tbl %>%
  ensemble_model_spec(
    model_spec = rand_forest(
      mode = "regression",
      trees = tune(),
      min_n = tune()
    ) %>%
      set_engine("ranger"),
    kfolds  = 10, 
    grid    = 20,
    control = control_grid(verbose = TRUE, 
                           allow_par = TRUE)
  )

modeltime_table(ensemble_fit_ranger_kfold) %>%
  modeltime_accuracy(testing(splits))

# XGBoost stacking algorithm ------
# We perform tuning on 4 parameters: trees, tree_depth, learn_rate, loss_reduction.

ensemble_fit_xgboost_kfold <- submodels_resamples_kfold_tbl %>%
  ensemble_model_spec(
    model_spec = boost_tree(
      trees          = tune(),
      tree_depth     = tune(),
      learn_rate     = tune(),
      loss_reduction = tune(), 
      mode = "regression"
    ) %>%
      set_engine("xgboost"),
    kfolds = 10, 
    grid   = 20, 
    control = control_grid(verbose = TRUE, 
                           allow_par = TRUE)
  )
                  
modeltime_table(ensemble_fit_xgboost_kfold) %>%
  modeltime_accuracy(testing(splits))     

# SVM stacking algorithm ----
# We perform tuning on 3 parameters: cost, rbf_sigma, margin. Run ?svm_rbf for explanations.

ensemble_fit_svm_kfold <- submodels_resamples_kfold_tbl %>%
  ensemble_model_spec(
    model_spec = svm_rbf(
      mode      = "regression",
      cost      = tune(),
      rbf_sigma = tune(),  
      margin    = tune()
    ) %>%
      set_engine("kernlab"),
    kfold = 10, 
    grid  = 20, 
    control = control_grid(verbose = TRUE, 
                           allow_par = TRUE)
  )

modeltime_table(ensemble_fit_svm_kfold) %>%
  modeltime_accuracy(testing(splits))

# Multi-Level Stacks-------
# The previous stacking approaches can be mutualized in a multi-level stack.
# We can combine meta-learners into a higher level of the stack which will 
# be a normal (weighted) average ensemble 

# https://business-science.github.io/modeltime.ensemble/articles/nested-ensembles.html

# metalerners_fit <- list(
#   
#   ensemble_fit_ranger_kfold = ensemble_fit_ranger_kfold,
#   ensemble_fit_xgboost_kfold = ensemble_fit_xgboost_kfold, 
#   ensemble_fit_svm_kfold = ensemble_fit_svm_kfold
# )
# 
# archivo_salida  <-  paste0(HOME_DIR,"/exp/001/metalerners_fit.rds")
# 
# metalerners_fit %>%
#   write_rds(archivo_salida)



modeltime_table(
  ensemble_fit_ranger_kfold, 
  ensemble_fit_xgboost_kfold,
  ensemble_fit_svm_kfold
  ) %>%
  modeltime_accuracy(testing(splits)) %>% 
  arrange(rmse)


loadings_tbl <- modeltime_table(
  ensemble_fit_ranger_kfold, 
  ensemble_fit_xgboost_kfold,
  ensemble_fit_svm_kfold
  ) %>% 
  modeltime_calibrate(testing(splits)) %>%
  modeltime_accuracy() %>%
  mutate(rank = min_rank(-rmse)) %>%
  select(.model_id, rank)

stacking_fit_wt <- modeltime_table(
  ensemble_fit_ranger_kfold, 
  ensemble_fit_xgboost_kfold,
  ensemble_fit_svm_kfold
  ) %>%
  ensemble_weighted(loadings = loadings_tbl$rank)

stacking_fit_wt  %>% 
  modeltime_calibrate(testing(splits)) %>%
  modeltime_accuracy() %>%
  arrange(rmse)

# Forescast plot---
# Below the stacked ensemble algorithm's forecast plot on the test dataset for 
# all industries, it looks good ! except for one industry the model seems to 
# follow pretty well the trend and spikes of the actual data.

calibration_stacking <- stacking_fit_wt %>% 
  modeltime_table() %>%
  modeltime_calibrate(testing(splits))

calibration_stacking %>%
  modeltime_forecast(
    new_data    = testing(splits),
    actual_data = artifacts$data$data_prepared_tbl,
    keep_data   = TRUE 
  ) %>%
  group_by(materia) %>%
  plot_modeltime_forecast(
    .facet_ncol         = 1, 
    .conf_interval_show = FALSE,
    .interactive        = TRUE,
    .title = "Predicciones en test con modelos apilados"
  )

# Próximos 12 meses
# Toggle ON parallel processing
plan(
  strategy = cluster,
  workers  = parallel::makeCluster(n_cores)
)

# Reajustamos modelo

# ESTE PARTE DEL SCRIPT FALLA POR EL CAMBIO EN LOS OBJETS: SE DEBE REENTRENAR TODO
# Sin tocar las salidas de los script.
# paralelizar ahorrará mucho tiempo!
refit_stacking_tbl <- calibration_stacking %>%
  modeltime_refit(
    data = artifacts$data$data_prepared_tbl,
    resamples = artifacts$data$data_prepared_tbl %>%
      drop_na() %>%
      vfold_cv(v = 3)
  )

# 12-m predicciones
forecast_stacking_tbl <- refit_stacking_tbl %>%
  modeltime_forecast(
    new_data    = artifacts$data$future_tbl,
    actual_data = artifacts$data$data_prepared_tbl,
    keep_data = TRUE
  )

# plot 
lforecasts <- lapply(X = 1:length(materia), FUN = function(x){
  forecast_stacking_tbl %>%
    filter(materia == materia[x]) %>%
    #group_by(materia) %>%
    mutate(across(.value:.conf_hi,
                  .fns = ~standardize_inv_vec(x = .,
                                              mean = artifacts$standardize$std_mean[x],
                                              sd = artifacts$standardize$std_sd[x]))) %>%
    mutate(across(.value:.conf_hi,
                  .fns = ~expm1(x = .)))
})

forecast_stacking_tbl <- bind_rows(lforecasts)

forecast_stacking_tbl %>%
  group_by(materia) %>%
  plot_modeltime_forecast(.title = "Sentencias: Predicción 1 año",
                          .facet_ncol         = 1,
                          .conf_interval_show = FALSE,
                          .interactive        = TRUE)



# Código OK

# refit_calibration_tbl <- calibration_tbl %>% 
#   modeltime_refit(
#     data = artifacts$data$data_prepared_tbl,
#     resamples = artifacts$data$data_prepared_tbl %>%
#       vfold_cv(v = 3)
#   )
# 
# # 12-month forecast calculations with future dataset
# forecast_calibration_tbl <- refit_calibration_tbl %>%
#   modeltime_forecast(
#     new_data    = artifacts$data$future_tbl,
#     actual_data = artifacts$data$data_prepared_tbl, 
#     keep_data = TRUE
#   )
# 
# # Toggle OFF parallel processing
# plan(sequential)
# 
# lforecasts <- lapply(X = 1:length(materia), FUN = function(x){
#   forecast_calibration_tbl %>%
#     filter(materia == materia[x]) %>%
#     #group_by(materia) %>%
#     mutate(across(.value:.conf_hi,
#                   .fns = ~standardize_inv_vec(x = .,
#                                               mean = artifacts$standardize$std_mean[x],
#                                               sd = artifacts$standardize$std_sd[x]))) %>%
#     mutate(across(.value:.conf_hi,
#                   .fns = ~expm1(x = .)))
# })
# 
# forecast_calibration_tbl <- bind_rows(lforecasts)
# 
# forecast_calibration_tbl %>%
#   group_by(materia) %>%
#   plot_modeltime_forecast(.title = "Sentencias: Predicción 1 año",     
#                           .facet_ncol         = 1, 
#                           .conf_interval_show = FALSE,
#                           .interactive        = TRUE)


multilevelstack <- list(
  
  resamples_kfold = resamples_kfold,
  
  submodels_resamples_kfold_tbl = submodels_resamples_kfold_tbl, 
  
  ensemble_fit_ranger_kfold =  ensemble_fit_ranger_kfold, 
  ensemble_fit_xgboost_kfold = ensemble_fit_xgboost_kfold,
  ensemble_fit_svm_kfold =  ensemble_fit_svm_kfold,
  loadings_tbl = loadings_tbl,
  stacking_fit_wt = stacking_fit_wt,
  calibration_stacking = calibration_stacking,
  #refit_stacking_tbl = refit_stacking_tbl,
  #forecast_stacking_tbl = forecast_stacking_tbl
  refit_calibration_tbl = refit_calibration_tbl, 
  forecast_calibration_tbl = forecast_calibration_tbl
  
)

archivo_salida  <-  paste0(HOME_DIR,"/exp/001/multilevelstack.rds")

multilevelstack %>%
  write_rds(archivo_salida)
