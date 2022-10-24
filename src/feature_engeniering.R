# PART 1 -  FEATURE ENGINEERING
pacman::p_load(tidyverse,
               timetk,
               tsibble,
               tsibbledata,
               fastDummies,
               skimr, 
               tidymodels)

options(scipen = 999)


# Load and transform
produccion_tbl <- readRDS(paste0(RAW_DATA_DIR, "/produccion.rds")) %>% 
  mutate(mes = ymd(str_c(año, mes, "-01"))) %>% 
  filter(instancia == "Primera_Instancia", tproducto == "sentencias") %>% 
  group_by(mes, materia) %>% 
  summarise(sentencias_dictadas = sum(cantidad, na.rm = T)) %>% 
  mutate(materia = as.factor(materia)) %>% 
  group_by(materia) %>% 
  tk_tbl()


# augment dataset
materia <- unique(produccion_tbl$materia)

groups <- lapply(X = 1:length(materia), FUN = function(x){
  
  produccion_tbl %>%
    filter(materia == materia[x]) %>%
    arrange(mes) %>%
    mutate(sentencias_dictadas =  log1p(x = sentencias_dictadas)) %>%
    mutate(sentencias_dictadas =  standardize_vec(sentencias_dictadas)) %>%
    future_frame(mes, .length_out = "12 month", .bind_data = TRUE) %>%
    mutate(materia = materia[x]) %>%
    tk_augment_fourier(.date_var = mes, .periods = 12, .K = 1) %>%
    tk_augment_lags(.value = sentencias_dictadas, .lags = c(12, 13 )) %>%
    tk_augment_slidify(.value   = c(sentencias_dictadas_lag12, sentencias_dictadas_lag13),
                       .f       = ~ mean(.x, na.rm = TRUE), 
                       .period  = c(3, 6, 9, 12),
                       .partial = TRUE,
                       .align   = "center")
})

groups_fe_tbl <- bind_rows(groups) %>%
  rowid_to_column(var = "rowid")


tmp <- produccion_tbl %>%
  group_by(materia) %>% 
  arrange(mes) %>%
  mutate(sentencias_dictadas = log1p(x = sentencias_dictadas)) %>% 
  group_map(~ c(mean = mean(.x$sentencias_dictadas, na.rm = TRUE),
                sd = sd(.x$sentencias_dictadas, na.rm = TRUE))) %>% 
  bind_rows()

std_mean <- tmp$mean
std_sd <- tmp$sd
rm('tmp')

# preparo datasets futuro
data_prepared_tbl <- groups_fe_tbl %>%
  filter(!is.na(sentencias_dictadas)) %>%
  drop_na()

future_tbl <- groups_fe_tbl %>%
  filter(is.na(sentencias_dictadas))


# Train and test datasets split
# cantidad de observaciones en test dataset: como mínimo tan largo como 
# el horizonte de predicción 

splits <- data_prepared_tbl %>%
  time_series_split(mes, 
                    assess = "12 months", 
                    cumulative = TRUE)
# Create recipe

recipe_spec <- recipe(sentencias_dictadas ~ ., data = training(splits)) %>%
  update_role(rowid, new_role = "indicator") %>%  
  step_other(materia) %>%
  step_timeseries_signature(mes) %>%
  step_rm(matches("(.xts$)|(.iso$)|(hour)|(minute)|(second)|(day)|(week)|(am.pm)")) %>%
  step_dummy(all_nominal(), one_hot = TRUE) %>%
  step_normalize(mes_index.num, mes_year)

#recipe_spec

# Recipe summary:
recipe_spec %>% 
  prep() %>%
  juice() %>% 
  glimpse() 

# mes_index.num and yar normalization Parameters
myskim <- skim_with(numeric = sfl(max, min), append = TRUE)

norm_param = recipe(sentencias_dictadas ~ ., data = training(splits)) %>%
  step_timeseries_signature(mes) %>% prep() %>% juice() %>%  myskim() %>% 
  dplyr::filter(skim_variable == "mes_index.num" | 
                  skim_variable == "mes_year") %>% 
  yank("numeric") %>% as_tibble()

# mes index
mes_index.num_limit_lower = norm_param$min[norm_param$skim_variable == "mes_index.num"]
mes_index.num_limit_upper = norm_param$max[norm_param$skim_variable == "mes_index.num"]
# mes_year normalization Parameters
mes_year_limit_lower = norm_param$min[norm_param$skim_variable == "mes_year"]
mes_year_limit_upper = norm_param$max[norm_param$skim_variable == "mes_year"]

  
feature_engineering_artifacts_list <- list(
  # Data
  data = list(
    data_prepared_tbl = data_prepared_tbl,
    future_tbl      = future_tbl,
    materia = materia
  ),
  
  # Recipes
  recipes = list(
    recipe_spec = recipe_spec
  ),
  
  # Splits
  splits = splits,
  
  # Inversion Parameters
  standardize = list(
    std_mean = std_mean,
    std_sd   = std_sd
  ),
  
  normalize = list(
    mes_index.num_limit_lower = mes_index.num_limit_lower, 
    mes_index.num_limit_upper = mes_index.num_limit_upper,
    mes_year_limit_lower = mes_year_limit_lower,
    mes_year_limit_upper = mes_year_limit_upper
  )  
)

dir.create(paste0(HOME_DIR, "/exp/001/"), showWarnings = FALSE )
archivo_salida  <-  paste0(HOME_DIR,"/exp/001/feature_engineering_artifacts_list.rds")

feature_engineering_artifacts_list %>% 
  write_rds(archivo_salida)



