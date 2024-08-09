# Databricks notebook source
# MAGIC  %sh
# MAGIC  rm -r /dbfs/delta_lab
# MAGIC  mkdir /dbfs/delta_lab
# MAGIC  wget -O /dbfs/delta_lab/covid_data.csv https://github.com/MicrosoftLearning/mslearn-databricks/raw/main/data/covid_data.csv

# COMMAND ----------

display(dbutils.fs.ls("dbfs:/pipelines/delta_lab"))

# COMMAND ----------

df = spark.read.format("delta").load('/pipelines/delta_lab/tables/aggregated_covid_data')
display(df)

# COMMAND ----------


