# Databricks notebook source
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
