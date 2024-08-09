# Databricks notebook source
# Use dbutils.widget define a "folder" variable with a default value
dbutils.widgets.text("folder", "data")
   
# Now get the parameter value (if no value was passed, the default set above will be used)
folder = dbutils.widgets.get("folder")

# COMMAND ----------

import urllib3
   
# Download product data from GitHub
response = urllib3.PoolManager().request('GET', 'https://raw.githubusercontent.com/MicrosoftLearning/mslearn-databricks/main/data/products.csv')
data = response.data.decode("utf-8")
   
# Save the product data to the specified folder
path = "dbfs:/{0}/products.csv".format(folder)
dbutils.fs.put(path, data, True)
