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

submodels_tbl <- workflows_tunednot$workflows

calibration_tbl %>% 
  modeltime_accuracy() %>%
  arrange(rmse)

# Average ensembles
# Average ensembles is a fairly simple concept, its main advantage is that it 
# does not require retraining. If you use the mean be aware of the individual 
# performance of each model since the "average model" performance will be sensitive 
# to "bad models" and outliers. To cope with this situation you can use the median 
# (more robust than the mean). Do not use the median when all models are doing 
# pretty well.
# 
# To create these two ensembles is pretty simple, all you have to do is to pipe 
# the modeltime table with all workflows into the ensemble_average() and specify 
# the type, either mean or median.

# Two ensembles: mean and median

ensemble_fit_mean <- calibration_tbl %>%
  ensemble_average(type = "mean")

ensemble_fit_median <- calibration_tbl %>%
  ensemble_average(type = "median")


# Weigthed ensembles
# Weighted ensembles can improve performance compared to average ensembles but 
# we must decide of the weight values. One solution is to use a simple rank 
# technique: create a rank column for which highest value is for the best
# model (lowest RMSE).
# 
# In the code snippet below we add the rank column and then we select .model_id 
# and rank columns to define a loadings tibble which will be used for defining 
# the weighted ensemble.

calibration_tbl %>%
  modeltime_accuracy() %>%
  mutate(rank = min_rank(-rmse)) 

loadings_tbl <- calibration_tbl %>%
  modeltime_accuracy() %>%
  mutate(rank = min_rank(-rmse)) %>%
  select(.model_id, rank)

# Then the weigths are created by passing models rank to the ensemble_weighted() 
# function. The newly created 
# ".loadings" column are the weights, the sum of all weigths is equal to 1.

ensemble_fit_wt <- calibration_tbl %>%
  ensemble_weighted(loadings = loadings_tbl$rank)

# Models evaluation
# Finally, let us display the accuracy results by adding previous ensemble models 
# to a modeltime table, calibrating the latter and piping it into the 
# modeltime_accuracy() function. We sort by ascending RMSE.


modeltime_table(
  ensemble_fit_mean,
  ensemble_fit_median,
  ensemble_fit_wt
  ) %>%
  modeltime_calibrate(testing(splits)) %>%
  modeltime_accuracy(testing(splits)) %>%
  arrange(rmse)

# We add the ensembles' calibration table to the non-tuned and tuned models' 
# calibration table (calibration_tbl).

calibration_all_tbl <- modeltime_table(
  ensemble_fit_mean,
  ensemble_fit_median,
  ensemble_fit_wt
  ) %>% 
  modeltime_calibrate(testing(splits)) %>%
  combine_modeltime_tables(calibration_tbl)

calibration_all_tbl %>%
  modeltime_calibrate(testing(splits)) %>%
  modeltime_accuracy() %>%
  arrange(desc(rsq))

# Since one of the average ensemble was the best model, we did not bother with 
# improving ensemble models performance with model selection e.g., selecting the
# top 5 models and/or suppressing redundant models such as the tuned and non-tuned 
# Prophet models which have the same performance.
