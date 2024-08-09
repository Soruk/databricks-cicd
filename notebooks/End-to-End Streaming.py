# Databricks notebook source
# MAGIC  %sh
# MAGIC  rm -r /dbfs/device_stream
# MAGIC  mkdir /dbfs/device_stream
# MAGIC  wget -O /dbfs/device_stream/device_data.csv https://github.com/MicrosoftLearning/mslearn-databricks/raw/main/data/device_data.csv

# COMMAND ----------

 from pyspark.sql.functions import *
 from pyspark.sql.types import *

 # Define the schema for the incoming data
 schema = StructType([
     StructField("device_id", StringType(), True),
     StructField("timestamp", TimestampType(), True),
     StructField("temperature", DoubleType(), True),
     StructField("humidity", DoubleType(), True)
 ])

 # Read streaming data from folder
 inputPath = '/device_stream/'
 iotstream = spark.readStream.schema(schema).option("header", "true").csv(inputPath)
 print("Source stream created...")

 # Write the data to a Delta table
 query = (iotstream
          .writeStream
          .format("delta")
          .option("checkpointLocation", "/tmp/checkpoints/iot_data")
          .start("/tmp/delta/iot_data"))

# COMMAND ----------

 display(dbutils.fs.ls("dbfs:/pipelines/device_stream/tables"))

# COMMAND ----------

df = spark.read.format("delta").load('/pipelines/device_stream/tables/transformed_iot_data')
display(df)

# COMMAND ----------

query.stop()
