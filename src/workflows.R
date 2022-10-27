# PART 2 WORKFLOWS
#remotes::install_github("curso-r/treesnip")
#install.packages("modeltime")
#install.packages("ranger")

pacman::p_load(tidyverse,
               timetk,
               tsibble,
               tsibbledata,
               fastDummies,
               skimr, 
               tidymodels, 
               ranger,
               lightgbm,
               modeltime, 
               treesnip # lightgbm
               )

# inspiration
# https://blog.bguarisma.com/time-series-forecasting-lab-part-2-feature-engineering-with-recipes
# https://www.rstudio.com/blog/update-your-machine-learning-pipeline-with-vetiver-and-quarto/
# https://pins.rstudio.com/articles/pins.html#versioning

artifacts <- read_rds(paste0(HOME_DIR, "/exp/001/feature_engineering_artifacts_list.rds"))

splits            <- artifacts$splits
recipe_spec       <- artifacts$recipes$recipe_spec
materia        <- artifacts$data$materia

# ML model specification and engine
# You will find in the figure below the specification and engine(s) for each of the 4 models used in this article.
# Random Forest modeltime function rand_forest() uses ranger from parsnip package as default engine.
# XGBoost modeltime function boost_tree() uses xgboost from parsnip package as default engine.
# Prophet modeltime function prophet_reg() uses prophet from modeltime package as default engine.
# Prophet Boost modeltime function prophet_boost() uses prophet_xgboost from modeltime package as default engine.

# RANDOM FOREST ----
wflw_fit_rf <- workflow() %>%
  add_model(
    spec = rand_forest(
      mode = "regression"
    ) %>% 
      set_engine("ranger")
  ) %>%
  add_recipe(recipe_spec %>% 
               step_rm(mes)) %>%
               fit(training(splits))
             
             
# XGBOOST ----
wflw_fit_xgboost <- workflow() %>%
  add_model(
    spec = boost_tree(
      mode = "regression"
    ) %>%
      set_engine("xgboost")
  ) %>%
  add_recipe(recipe_spec %>% 
               step_rm(mes)) %>%
               fit(training(splits))

# PROPHET ----
wflw_fit_prophet <- workflow() %>%
  add_model(
    spec = prophet_reg(
      seasonality_daily  = FALSE, 
      seasonality_weekly = FALSE, 
      seasonality_yearly = TRUE
    ) %>% 
      set_engine("prophet")
  ) %>%
  add_recipe(recipe_spec) %>%
  fit(training(splits))

# PROPHET BOOST ----
wflw_fit_prophet_boost <- workflow() %>%
  add_model(
    spec = prophet_boost(
      seasonality_daily  = FALSE, 
      seasonality_weekly = FALSE, 
      seasonality_yearly = FALSE
    ) %>% 
      set_engine("prophet_xgboost")
  ) %>%
  add_recipe(recipe_spec) %>%
  fit(training(splits))

# LIGHTGBM --------------
wflw_fit_ligthgbm <- workflow() %>%
  add_model(
    spec =  parsnip::boost_tree(
      mtry = 5,
      trees = 1000
    ) %>% 
      set_mode("regression") %>%
      set_engine("lightgbm", 
                 objective = "root_mean_squared_error")) %>%
  add_recipe(recipe_spec) %>%
  fit(training(splits))
    
# Models evaluation
# Modeltime table

submodels_tbl <- modeltime_table(
  wflw_fit_rf,
  wflw_fit_xgboost,
  wflw_fit_prophet,
  wflw_fit_prophet_boost,
  wflw_fit_ligthgbm
)

submodels_tbl


# Calibration table
# Calibration sets the stage for accuracy and forecast confidence by computing predictions and residuals from out of sample data.
# 
# Two columns are added to the previous modeltime table:
#   
# .type: Indicates the sample type: "Test" if predicted, or "Fitted" if residuals were stored during modeling.
# 
# .calibration_data: it contains a tibble with Timestamps, Actual Values, Predictions and Residuals calculated from new_data (Test Data)

calibrated_wflws_tbl <- submodels_tbl %>%
  modeltime_calibrate(new_data = testing(splits))


# Model Evaluation

calibrated_wflws_tbl %>%
  modeltime_accuracy(testing(splits)) %>%
  arrange(rmse)


# Save your work
workflow_artifacts <- list(
  
  workflows = list(
    
    wflw_random_forest = wflw_fit_rf,
    wflw_xgboost = wflw_fit_xgboost,
    wflw_prophet = wflw_fit_prophet,
    wflw_prophet_boost = wflw_fit_prophet_boost,
    wflw_lightgbm = wflw_fit_ligthgbm
    
  ),
  
  calibration = list(calibration_tbl = calibrated_wflws_tbl)
  
)


archivo_salida  <-  paste0(HOME_DIR,"/exp/001/workflows_artifacts_list.rds")

workflow_artifacts %>% 
  write_rds(archivo_salida)



