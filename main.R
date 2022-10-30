# principal
if (!require("pacman")) install.packages("pacman")
pacman::p_load(tidyverse, data.table, dtplyr, here,
               lubridate, Rcpp, yaml, rlist, ggplot2, rmarkdown,
               ROCR, methods, Matrix, caret, rsample, renv)


HOME_DIR = here::here()
RAW_DATA_DIR = paste0(HOME_DIR,  "/data")
PROCESSED_DATA_DIR = paste0(HOME_DIR,  "/data/processed")
LOGS_DIR = paste0(HOME_DIR,  "/logs")
TRACKING_FILE = paste0(HOME_DIR,  "/tracking")
EXP_DIR = paste0(HOME_DIR,  "/exp")


# Free data
# url = "http://datos.jus.gob.ar/dataset/42720e56-2274-4ad5-820d-c366d784bc8c/resource/21b615fc-001d-43d1-9396-e61f804a32cc/download/llamados-atendidos-violencia-familiar-unificado-201701-202208.csv"
# download.file(url, paste0(RAW_DATA_DIR , "/violencia_familiar.csv"))
