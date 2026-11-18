PY := .venv/bin/python
R  := Rscript

.PHONY: all setup data models analysis r clean

all: data models analysis r

setup:
	python3 -m venv .venv
	$(PY) -m pip install -r requirements.txt
	$(R) -e 'install.packages(c("readr","dplyr","tidyr","ggplot2"), repos = "https://cloud.r-project.org")'

data:
	$(PY) python/01_get_data.py

models:
	$(PY) python/02_models.py

analysis:
	$(PY) python/03_error_analysis.py

r:
	$(R) R/04_temperature_effect.R

# Removes generated outputs; keeps the cached raw data (daily_demand.csv, weather_daily.csv).
clean:
	rm -f data/daily.csv results/* figures/*
