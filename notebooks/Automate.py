# Databricks notebook source
# MAGIC  %sh
# MAGIC  rm -r /dbfs/FileStore
# MAGIC  mkdir /dbfs/FileStore
# MAGIC  wget -O /dbfs/FileStore/sample_sales_data.csv https://github.com/MicrosoftLearning/mslearn-databricks/raw/main/data/sample_sales_data.csv

# COMMAND ----------

 # Load the sample dataset into a DataFrame
 df = spark.read.csv('/FileStore/*.csv', header=True, inferSchema=True)
 df.show()

# COMMAND ----------

 from pyspark.sql.functions import col, sum

 # Aggregate sales data by product category
 sales_by_category = df.groupBy('product_category').agg(sum('transaction_amount').alias('total_sales'))
 sales_by_category.show()
