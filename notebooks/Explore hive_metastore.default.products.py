# Databricks notebook source
# MAGIC %sql
# MAGIC SELECT * FROM `hive_metastore`.`default`.`products`;

# COMMAND ----------

 df = spark.sql("SELECT * FROM products")
 df = df.filter("Category == 'Road Bikes'")
 display(df)
