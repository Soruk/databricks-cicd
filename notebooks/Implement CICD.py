# Databricks notebook source
# MAGIC  %sh
# MAGIC  rm -r /dbfs/FileStore
# MAGIC  mkdir /dbfs/FileStore
# MAGIC  wget -O /dbfs/FileStore/sample_sales.csv https://github.com/MicrosoftLearning/mslearn-databricks/raw/main/data/sample_sales.csv
