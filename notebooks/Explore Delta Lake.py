# Databricks notebook source
# MAGIC  %sh
# MAGIC  rm -r /dbfs/delta_lab
# MAGIC  mkdir /dbfs/delta_lab
# MAGIC  wget -O /dbfs/delta_lab/products.csv https://raw.githubusercontent.com/MicrosoftLearning/mslearn-databricks/main/data/products.csv

# COMMAND ----------

df = spark.read.load('/delta_lab/products.csv', format='csv', header=True)
display(df.limit(10))

# COMMAND ----------

delta_table_path = "/delta/products-delta"
df.write.format("delta").save(delta_table_path)

# COMMAND ----------

# MAGIC  %sh
# MAGIC  ls /dbfs/delta/products-delta

# COMMAND ----------

from delta.tables import *
from pyspark.sql.functions import *
   
# Create a deltaTable object
deltaTable = DeltaTable.forPath(spark, delta_table_path)
# Update the table (reduce price of product 771 by 10%)
deltaTable.update(
    condition = "ProductID == 771",
    set = { "ListPrice": "ListPrice * 0.9" })
# View the updated data as a dataframe
deltaTable.toDF().show(10)

# COMMAND ----------

new_df = spark.read.format("delta").load(delta_table_path)
new_df.show(10)

# COMMAND ----------

new_df = spark.read.format("delta").option("versionAsOf", 0).load(delta_table_path)
new_df.show(10)

# COMMAND ----------

deltaTable.history(10).show(10, False, True)

# COMMAND ----------

spark.sql("CREATE DATABASE AdventureWorks")
spark.sql("CREATE TABLE AdventureWorks.ProductsExternal USING DELTA LOCATION '{0}'".format(delta_table_path))
spark.sql("DESCRIBE EXTENDED AdventureWorks.ProductsExternal").show(truncate=False)

# COMMAND ----------

# MAGIC %sql
# MAGIC USE AdventureWorks;
# MAGIC SELECT * FROM ProductsExternal;

# COMMAND ----------

df.write.format("delta").saveAsTable("AdventureWorks.ProductsManaged")
spark.sql("DESCRIBE EXTENDED AdventureWorks.ProductsManaged").show(truncate=False)

# COMMAND ----------

# MAGIC %sql
# MAGIC USE AdventureWorks;
# MAGIC SELECT * FROM ProductsManaged;

# COMMAND ----------

# MAGIC %sql
# MAGIC USE AdventureWorks;
# MAGIC SHOW TABLES;

# COMMAND ----------

# MAGIC  %sh
# MAGIC  echo "External table:"
# MAGIC  ls /dbfs/delta/products-delta
# MAGIC  echo
# MAGIC  echo "Managed table:"
# MAGIC  ls /dbfs/user/hive/warehouse/adventureworks.db/productsmanaged

# COMMAND ----------

# MAGIC %sql
# MAGIC USE AdventureWorks;
# MAGIC DROP TABLE IF EXISTS ProductsExternal;
# MAGIC DROP TABLE IF EXISTS ProductsManaged;
# MAGIC SHOW TABLES;

# COMMAND ----------

# MAGIC  %sh
# MAGIC  echo "External table:"
# MAGIC  ls /dbfs/delta/products-delta
# MAGIC  echo
# MAGIC  echo "Managed table:"
# MAGIC  ls /dbfs/user/hive/warehouse/adventureworks.db/productsmanaged

# COMMAND ----------

# MAGIC %sql
# MAGIC USE AdventureWorks;
# MAGIC CREATE TABLE Products
# MAGIC USING DELTA
# MAGIC LOCATION '/delta/products-delta';

# COMMAND ----------

# MAGIC %sql
# MAGIC USE AdventureWorks;
# MAGIC SELECT * FROM Products;

# COMMAND ----------

# MAGIC  %sh
# MAGIC  rm -r /dbfs/device_stream
# MAGIC  mkdir /dbfs/device_stream
# MAGIC  wget -O /dbfs/device_stream/devices1.json https://raw.githubusercontent.com/MicrosoftLearning/mslearn-databricks/main/data/devices1.json

# COMMAND ----------

from pyspark.sql.types import *
from pyspark.sql.functions import *
   
# Create a stream that reads data from the folder, using a JSON schema
inputPath = '/device_stream/'
jsonSchema = StructType([
StructField("device", StringType(), False),
StructField("status", StringType(), False)
])
iotstream = spark.readStream.schema(jsonSchema).option("maxFilesPerTrigger", 1).json(inputPath)
print("Source stream created...")

# COMMAND ----------

# Write the stream to a delta table
delta_stream_table_path = '/delta/iotdevicedata'
checkpointpath = '/delta/checkpoint'
deltastream = iotstream.writeStream.format("delta").option("checkpointLocation", checkpointpath).start(delta_stream_table_path)
print("Streaming to delta sink...")

# COMMAND ----------

# Read the data in delta format into a dataframe
df = spark.read.format("delta").load(delta_stream_table_path)
display(df)

# COMMAND ----------

# create a catalog table based on the streaming sink
spark.sql("CREATE TABLE IotDeviceData USING DELTA LOCATION '{0}'".format(delta_stream_table_path))

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM IotDeviceData;

# COMMAND ----------

# MAGIC  %sh
# MAGIC  wget -O /dbfs/device_stream/devices2.json https://raw.githubusercontent.com/MicrosoftLearning/mslearn-databricks/main/data/devices2.json

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM IotDeviceData;

# COMMAND ----------

deltastream.stop()
